#!/usr/bin/env python3
import itertools
import time
import json
import numpy as np

# ======================
# 配置参数
# ======================
TOL = 0.015
TOP_N = 10
MAX_SEEDS = 150              # 限制种子数量，防止 Refine 时间爆炸
PRIORITY_ERROR_THRESHOLD = 0.0005
NON_PRIORITY_ERROR_THRESHOLD = 0.005
D_RANGE = (0.5, 10.0)
X4_RATIO_LIMIT = 4.0
MAX_X4 = 6.0

GROUP_SIZES = np.array([1, 2, 2, 2, 1])  
BUCKET_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
BUCKET_TO_GROUP = [0, 1, 2, 2, 3, 3, 1, 4]

# ======================
# 核心搜索函数
# ======================

def fast_backtrack(targets, weights, D, json_data, pos_constraints):
    """高度优化的回溯，减少内部内存分配"""
    num_colors = len(targets)
    results = []
    current_sums = [0.0] * num_colors
    current_assign = [None] * 5

    def dfs(g_idx):
        if g_idx == 5:
            # 基础误差检查
            total_err = 0.0
            for i in range(num_colors):
                err = abs(current_sums[i]/D - targets[i])
                total_err += err
                # 种子阶段稍微放宽，但如果太大就剔除
                if err > 0.01: return 
            
            flat = [None] * 8
            flat[0] = current_assign[0][0]
            flat[1], flat[6] = current_assign[1]
            flat[2], flat[3] = current_assign[2]
            flat[4], flat[5] = current_assign[3]
            flat[7] = current_assign[4][0]
            results.append((flat, total_err))
            return

        size = GROUP_SIZES[g_idx]
        w = weights[g_idx]
        
        # 使用 combinations_with_replacement 遍历颜色分配方案
        for combo in itertools.combinations_with_replacement(range(num_colors), size):
            # POSITION 约束
            if g_idx in pos_constraints:
                valid = True
                for req in pos_constraints[g_idx]:
                    if req not in combo: # 简化版检查，假设一组内位置相同或不冲突
                        valid = False; break
                if not valid: continue

            # 更新 current_sums
            for c_idx in combo: current_sums[c_idx] += w
            
            # 剪枝检查
            pass_check = True
            for i in range(num_colors):
                if current_sums[i]/D > targets[i] + 0.02: # 宽松剪枝
                    pass_check = False; break
            
            if pass_check:
                current_assign[g_idx] = combo
                dfs(g_idx + 1)
            
            # 回溯
            for c_idx in combo: current_sums[c_idx] -= w

    dfs(0)
    return results

def refine_vectorized(seeds, targets, json_data):
    """NumPy 矢量化精修 - 优化计算规模"""
    if not seeds: return []
    
    # 缩小网格到 7^4 = 2401
    offsets = np.linspace(-0.03, 0.03, 7) 
    d1, d2, d3, d4 = np.meshgrid(offsets, offsets, offsets, offsets)
    d1, d2, d3, d4 = d1.ravel(), d2.ravel(), d3.ravel(), d4.ravel()
    
    target_arr = np.array(targets)
    is_priority = np.array([item["PRIORITY"] for item in json_data])
    refined_results = []
    
    # 只对前 50 个最优秀的种子进行精修，节省 75% 的 Refine 时间
    for sol in seeds[:50]:
        mX1, mX2, mX3, mX4 = sol['X1']+d1, sol['X2']+d2, sol['X3']+d3, sol['X4']+d4
        
        # 快速过滤无效参数
        mask = (mX1 >= 1.0) & (mX2 >= 1.0) & (mX3 >= 1.0) & (mX4 > mX1) & (mX4 > mX3) & \
               (mX4 <= MAX_X4) & (mX4/mX1 <= X4_RATIO_LIMIT) & (mX4/mX3 <= X4_RATIO_LIMIT)
        if not np.any(mask): continue
        mX1, mX2, mX3, mX4 = mX1[mask], mX2[mask], mX3[mask], mX4[mask]
        
        # 矢量化计算
        W = np.array([1/mX1, 1/mX4, np.ones_like(mX1), 1/mX2, 1/mX3])
        D = np.dot(GROUP_SIZES, W)
        
        # 提取当前方案的颜色权重
        c_weights = np.zeros((len(targets), 5))
        for b_idx, c_idx in enumerate(sol['assign']):
            c_weights[c_idx, BUCKET_TO_GROUP[b_idx]] += 1
        
        # 最终比例 = (颜色组权重 @ 组速度) / D
        final_pcts = np.dot(c_weights, W) / D
        errors = np.abs(final_pcts - target_arr[:, None])
        
        # 优先级检查
        valid_mask = np.all(errors <= np.where(is_priority[:, None], PRIORITY_ERROR_THRESHOLD, NON_PRIORITY_ERROR_THRESHOLD), axis=0)
        
        if np.any(valid_mask):
            total_errs = np.sum(errors, axis=0)
            best_idx = np.argmin(np.where(valid_mask, total_errs, 999))
            refined_results.append({
                'X1': mX1[best_idx], 'X2': mX2[best_idx], 'X3': mX3[best_idx], 'X4': mX4[best_idx],
                'assign': sol['assign'], 'dev': total_errs[best_idx], 'D': D[best_idx],
                'final_pcts': final_pcts[:, best_idx].tolist()
            })
        else:
            refined_results.append(sol)
            
    return refined_results

# ======================
# 主接口
# ======================

def linkrun(json_str):
    data = json.loads(json_str)["data"]
    raw_targets = np.array([item["MATRATCALC"] for item in data])
    targets = raw_targets / raw_targets.sum()
    
    # 预处理位置约束
    pos_constraints = {}
    for i, item in enumerate(data):
        if item["POSITION"]:
            g_idx = BUCKET_TO_GROUP[BUCKET_NAMES.index(item["POSITION"])]
            pos_constraints.setdefault(g_idx, []).append(i)

    seeds = []
    # 恢复分阶段搜索：由窄到宽
    stages = [
        (1.1, 2.2, 0.2, 1.1, 3.2, 0.2), # 快速粗扫
        (1.0, 3.5, 0.1, 1.1, 4.5, 0.1)  # 细扫
    ]
    
    start_time = time.time()
    print("Stage 1: Searching for seeds...")
    
    for x_min, x_max, x_step, x4_min, x4_max, x4_step in stages:
        if len(seeds) >= MAX_SEEDS: break
        
        x_vals = np.arange(x_min, x_max, x_step)
        x4_vals = np.arange(x4_min, x4_max, x4_step)
        
        for x1, x2, x3, x4 in itertools.product(x_vals, x_vals, x_vals, x4_vals):
            if not (x4 > x1 + 0.05 and x4 > x3 + 0.05): continue
            if x4/x1 > X4_RATIO_LIMIT or x4/x3 > X4_RATIO_LIMIT: continue
            
            weights = np.array([1/x1, 1/x4, 1.0, 1/x2, 1/x3])
            D = np.sum(weights * GROUP_SIZES)
            if not (D_RANGE[0] < D < D_RANGE[1]): continue
            
            # 进入回溯
            res = fast_backtrack(targets, weights, D, data, pos_constraints)
            for af, dev in res:
                seeds.append({'X1':x1, 'X2':x2, 'X3':x3, 'X4':x4, 'assign':af, 'dev':dev})
            
            if len(seeds) >= MAX_SEEDS: break

    print(f"Found {len(seeds)} seeds in {time.time()-start_time:.2f}s. Refining...")
    
    # 精修
    refined = refine_vectorized(sorted(seeds, key=lambda x: x['dev']), targets, data)
    
    # 结果输出格式化 (同前)
    results = sorted(refined, key=lambda x: x['dev'])[:TOP_N]
    output = []
    for s in results:
        w = [1/s['X1'], 1/s['X4'], 1.0, 1/s['X2'], 1/s['X3']]
        output.append({
            'X1': round(s['X1'], 3), 'X2': round(s['X2'], 3), 'X3': round(s['X3'], 3), 'X4': round(s['X4'], 3),
            'cum_error': round(s['dev'], 6), 'total_feed_speed_D': round(s['D'], 6),
            'assign': [{
                'bucket': BUCKET_NAMES[i], 'color': data[s['assign'][i]]['MFMLIN'],
                'x': round([s['X1'], s['X4'], 1.0, s['X2'], s['X3']][BUCKET_TO_GROUP[i]], 2),
                'speed': round(w[BUCKET_TO_GROUP[i]], 6)
            } for i in range(8)],
            'colors': [{
                'color': data[i]['MFMLIN'], 'target': round(targets[i]*100, 2),
                'final': round(s['final_pcts'][i]*100, 2), 'error': round(abs(s['final_pcts'][i]-targets[i])*100, 4)
            } for i in range(len(data))]
        })
    
    print(f"Total time: {time.time()-start_time:.2f}s")
    return json.dumps({'results': output}, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    json_input = """{
    "data": [
        {"MFMLIN": 10, "MATRATCALC": 1.5, "PRIORITY": 0, "POSITION": ""},
        {"MFMLIN": 40, "MATRATCALC": 6.43, "PRIORITY": 0, "POSITION": ""},
        {"MFMLIN": 50, "MATRATCALC": 5.0, "PRIORITY": 0, "POSITION": ""},
        {"MFMLIN": 60, "MATRATCALC": 9.32, "PRIORITY": 0, "POSITION": ""},
        {"MFMLIN": 99999, "MATRATCALC": 4.0, "PRIORITY": 0, "POSITION": ""}
    ]
    }"""
    print(linkrun(json_input))