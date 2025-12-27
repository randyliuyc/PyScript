#!/usr/bin/env python3
import itertools
import time
import json
import numpy as np

# ======================
# 算法核心参数
# ======================
TOL = 0.015                  # 粗搜容差
TOP_N = 10                  
MAX_SEEDS = 150              # 限制种子数量，平衡速度与精度
PRIORITY_ERROR_THRESHOLD = 0.0005   # 0.05%
NON_PRIORITY_ERROR_THRESHOLD = 0.005 # 0.5%
D_RANGE = (0.5, 10.0)
X4_RATIO_LIMIT = 4.0
MAX_X4 = 6.0

# 物理结构定义
BUCKETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
# 速度组映射：A:G0, B:G1, C:G2, D:G2, E:G3, F:G3, G:G1, H:G4
GROUP_SIZES = np.array([1, 2, 2, 2, 1])  
BUCKET_TO_GROUP = np.array([0, 1, 2, 2, 3, 3, 1, 4])

# ======================
# 核心计算函数
# ======================

def fast_backtrack(targets, weights, D, num_colors, pos_constraints):
    """
    高度优化的回溯算法
    """
    results = []
    current_sums = [0.0] * num_colors
    current_assign = [None] * 5

    def dfs(g_idx):
        if g_idx == 5:
            total_err = 0.0
            final_pcts = []
            for i in range(num_colors):
                pct = current_sums[i] / D
                err = abs(pct - targets[i])
                total_err += err
                final_pcts.append(pct)
                if err > 0.015: return # 种子阶段单色误差阈值，略微放宽
            
            # 还原为 8 桶位分配 (存储颜色索引)
            flat = [None] * 8
            flat[0] = current_assign[0][0] # A
            flat[1], flat[6] = current_assign[1] # B, G
            flat[2], flat[3] = current_assign[2] # C, D
            flat[4], flat[5] = current_assign[3] # E, F
            flat[7] = current_assign[4][0] # H
            results.append((flat, total_err, final_pcts))
            return

        size = GROUP_SIZES[g_idx]
        w = weights[g_idx]
        
        for combo in itertools.combinations_with_replacement(range(num_colors), size):
            # POSITION 约束检查
            if g_idx in pos_constraints:
                valid = True
                for req in pos_constraints[g_idx]:
                    if req not in combo:
                        valid = False; break
                if not valid: continue

            # 更新当前各颜色速度总和
            for c_idx in combo: current_sums[c_idx] += w
            
            # 剪枝：单色占比不能显著超过目标值
            pass_check = True
            for i in range(num_colors):
                if current_sums[i]/D > targets[i] + 0.03:
                    pass_check = False; break
            
            if pass_check:
                current_assign[g_idx] = combo
                dfs(g_idx + 1)
            
            # 回溯
            for c_idx in combo: current_sums[c_idx] -= w

    dfs(0)
    return results

def refine_vectorized(seeds, targets, json_data):
    """
    NumPy 矢量化精修：并行计算小范围内的最优解
    """
    if not seeds: return []
    
    # 定义微调网格 7^4 = 2401 组
    offsets = np.linspace(-0.03, 0.03, 7) 
    d1, d2, d3, d4 = np.meshgrid(offsets, offsets, offsets, offsets)
    d1, d2, d3, d4 = d1.ravel(), d2.ravel(), d3.ravel(), d4.ravel()
    
    target_arr = np.array(targets)
    is_priority = np.array([item["PRIORITY"] for item in json_data])
    refined_results = []
    
    # 仅精修前 50 个优质种子
    for sol in seeds[:50]:
        mX1, mX2, mX3, mX4 = sol['X1']+d1, sol['X2']+d2, sol['X3']+d3, sol['X4']+d4
        
        # 物理与工艺约束过滤
        mask = (mX1 >= 1.0) & (mX2 >= 1.0) & (mX3 >= 1.0) & (mX4 > mX1 + 0.01) & \
               (mX4 > mX3 + 0.01) & (mX4 <= MAX_X4) & \
               (mX4/mX1 <= X4_RATIO_LIMIT) & (mX4/mX3 <= X4_RATIO_LIMIT)
        
        if not np.any(mask): 
            refined_results.append(sol)
            continue
            
        mX1, mX2, mX3, mX4 = mX1[mask], mX2[mask], mX3[mask], mX4[mask]
        
        # 矢量化计算总牵伸 D
        W = np.array([1/mX1, 1/mX4, np.ones_like(mX1), 1/mX2, 1/mX3]) # 5 x N
        D_vec = np.dot(GROUP_SIZES, W) # N
        
        # 计算该分配方案下各颜色的权重矩阵
        c_weights = np.zeros((len(targets), 5))
        for b_idx, color_idx in enumerate(sol['assign']):
            c_weights[color_idx, BUCKET_TO_GROUP[b_idx]] += 1
        
        # 各颜色比例 = (颜色组权重 @ 组速度) / D
        final_pcts_matrix = np.dot(c_weights, W) / D_vec # K x N
        errors_matrix = np.abs(final_pcts_matrix - target_arr[:, None])
        
        # 优先级误差过滤
        valid_mask = np.all(errors_matrix <= np.where(is_priority[:, None], PRIORITY_ERROR_THRESHOLD, NON_PRIORITY_ERROR_THRESHOLD), axis=0)
        
        if np.any(valid_mask):
            total_errs = np.sum(errors_matrix, axis=0)
            best_idx = np.argmin(np.where(valid_mask, total_errs, 9.9))
            refined_results.append({
                'X1': mX1[best_idx], 'X2': mX2[best_idx], 'X3': mX3[best_idx], 'X4': mX4[best_idx],
                'assign': sol['assign'], 'dev': total_errs[best_idx], 'D': D_vec[best_idx],
                'final_pcts': final_pcts_matrix[:, best_idx].tolist()
            })
        else:
            refined_results.append(sol)
            
    return refined_results

# ======================
# 主运行接口
# ======================

def linkrun(json_str):
    linkargs = json.loads(json_str)
    json_data = linkargs["data"]
    
    # 目标比例归一化
    raw_targets = np.array([item["MATRATCALC"] for item in json_data])
    targets = raw_targets / raw_targets.sum()
    targets_pct = targets * 100.0

    # 预处理位置约束
    pos_constraints = {}
    for i, item in enumerate(json_data):
        if item.get("POSITION"):
            try:
                g_idx = BUCKET_TO_GROUP[BUCKETS.index(item["POSITION"])]
                pos_constraints.setdefault(g_idx, []).append(i)
            except ValueError: pass

    # Stage 1: 粗搜种子
    seeds = []
    search_stages = [
        (1.1, 2.5, 0.2, 1.1, 3.5, 0.2), 
        (1.0, 4.0, 0.1, 1.1, 6.0, 0.1)  
    ]
    
    start_time = time.time()
    for x_min, x_max, x_step, x4_min, x4_max, x4_step in search_stages:
        if len(seeds) >= MAX_SEEDS: break
        x_vals = np.arange(x_min, x_max, x_step)
        x4_vals = np.arange(x4_min, x4_max, x4_step)
        
        for x1, x2, x3, x4 in itertools.product(x_vals, x_vals, x_vals, x4_vals):
            if not (x4 > x1 + 0.05 and x4 > x3 + 0.05): continue
            if x4/x1 > X4_RATIO_LIMIT or x4/x3 > X4_RATIO_LIMIT: continue
            
            w_seed = np.array([1/x1, 1/x4, 1.0, 1/x2, 1/x3])
            D_seed = np.sum(w_seed * GROUP_SIZES)
            if not (D_RANGE[0] < D_seed < D_RANGE[1]): continue
            
            res = fast_backtrack(targets, w_seed, D_seed, len(json_data), pos_constraints)
            for af, dev, pcts in res:
                seeds.append({
                    'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4, 
                    'assign': af, 'dev': dev, 'D': D_seed, 'final_pcts': pcts
                })
            
            if len(seeds) >= MAX_SEEDS: break

    # Stage 2: 矢量化精修
    refined = refine_vectorized(sorted(seeds, key=lambda x: x['dev']), targets, json_data)
    
    # 结果排序并输出
    top_results = sorted(refined, key=lambda x: x['dev'])[:TOP_N]
    
    final_results = []
    for s in top_results:
        w_map = [1/s['X1'], 1/s['X4'], 1.0, 1/s['X2'], 1/s['X3']]
        
        assign_list = []
        for i, b_name in enumerate(BUCKETS):
            g_idx = BUCKET_TO_GROUP[i]
            color_idx = s['assign'][i]
            
            # X 值显示逻辑
            if i == 0: x_val = s['X1']
            elif i in [1, 6]: x_val = s['X4']
            elif i in [4, 5]: x_val = s['X2']
            elif i == 7: x_val = s['X3']
            else: x_val = 1.0
            
            assign_list.append({
                'bucket': b_name,
                'color': json_data[color_idx]['MFMLIN'],
                'colordes': json_data[color_idx].get('MFMDES', ''),
                'x': round(x_val, 2),
                'speed': round(w_map[g_idx], 6)
            })
            
        final_results.append({
            'X1': round(s['X1'], 4),
            'X2': round(s['X2'], 4),
            'X3': round(s['X3'], 4),
            'X4': round(s['X4'], 4),
            'cum_error': round(s['dev'], 6),
            'total_feed_speed_D': round(s['D'], 6),
            'assign': assign_list,
            'colors': [{
                'color': json_data[i]['MFMLIN'],
                'target': round(targets_pct[i], 2),
                'final': round(s['final_pcts'][i]*100.0, 2),
                'error': round(abs(s['final_pcts'][i]*100.0 - targets_pct[i]), 4)
            } for i in range(len(json_data))]
        })

    return json.dumps({'results': final_results}, indent=2, ensure_ascii=False)

# ======================
# 程序入口
# ======================
if __name__ == "__main__":
    json_str = """{
    "pyFile": "bt8",
    "data": [
    {
        "MFMLIN": 10,
        "MFMDES": "SPW ER007 UGOV-8666 V1",
        "MATRATCALC": 1.500000,
        "PRIORITY": 0,
        "POSITION": ""
    },
    {
        "MFMLIN": 40,
        "MFMDES": "MT WP白棉 UCB196A-3 ",
        "MATRATCALC": 6.430000,
        "PRIORITY": 0,
        "POSITION": ""
    },
    {
        "MFMLIN": 50,
        "MFMDES": "MT W白棉 VG054ABY ",
        "MATRATCALC": 5.000000,
        "PRIORITY": 0,
        "POSITION": ""
    },
    {
        "MFMLIN": 60,
        "MFMDES": "MT SW本白 V-11388A ",
        "MATRATCALC": 9.320000,
        "PRIORITY": 0,
        "POSITION": ""
    },
    {
        "MFMLIN": 99999,
        "MFMDES": "HY 条子",
        "MATRATCALC": 4.000000,
        "PRIORITY": 0,
        "POSITION": ""
    }
    ]
    }"""

    result = linkrun(json_str)
    print(result)