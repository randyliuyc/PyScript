#!/usr/bin/env python3
import itertools
import time
import json
import numpy as np

# ======================
# 算法核心参数
# 2026年3月3日, 优化计算速度，当前使用版本
# ======================
TOL = 0.015                  # 粗搜容差 1.5%
TOP_N = 100                  # 结果返回的数量
MAX_SEEDS = 150              # 限制种子数量，平衡速度与精度
PRIORITY_ERROR_THRESHOLD = 0.0005   # 0.05%
NON_PRIORITY_ERROR_THRESHOLD = 0.005 # 0.5%
D_RANGE = (0.5, 10.0)
X4_RATIO_LIMIT = 4.0
MAX_X4 = 6.0
MAX_RUNTIME = 60             # 最大运行时间60秒

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
            
            # 将 total_err 保留两位小数，便于后续通过 total_feed_speed_D 判断更优解
            total_err = round(total_err, 2)
            
            # 还原为 8 桶位分配 (存储颜色索引)
            # 对于大小为2的组，按固定顺序展开（小索引在前，大索引在后）
            flat = [None] * 8
            flat[0] = current_assign[0][0] # A (组0, 大小1)
            # 组1 (BG): 按顺序展开，确保 B < G 的索引顺序
            c1, c2 = current_assign[1]
            flat[1], flat[6] = (c1, c2) if c1 <= c2 else (c2, c1)
            # 组2 (CD): 按顺序展开，确保 C < D 的索引顺序
            c1, c2 = current_assign[2]
            flat[2], flat[3] = (c1, c2) if c1 <= c2 else (c2, c1)
            # 组3 (EF): 按顺序展开，确保 E < F 的索引顺序
            c1, c2 = current_assign[3]
            flat[4], flat[5] = (c1, c2) if c1 <= c2 else (c2, c1)
            flat[7] = current_assign[4][0] # H (组4, 大小1)
            results.append((flat, total_err, final_pcts))
            return

        size = GROUP_SIZES[g_idx]
        w = weights[g_idx]
        
        # 对于大小为2的组(BG, CD, EF)，使用 combinations 避免组内顺序重复
        # 对于大小为1的组(A, H)，直接迭代
        if size == 2:
            combo_iter = itertools.combinations(range(num_colors), size)
        else:
            combo_iter = itertools.combinations_with_replacement(range(num_colors), size)
        
        for combo in combo_iter:
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
    
    # 仅精修前 TOP_N 个优质种子
    for sol in seeds[:TOP_N]:
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
                'final_pcts': final_pcts_matrix[:, best_idx].tolist(),
                'stage_label': sol.get('stage_label', 'Unknown')
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
    # 支持多字符 POSITION，如 "ACF" 表示该颜色需要出现在 A、C、F 三个位置
    pos_constraints = {}
    # bucket_constraints: 记录每个颜色被约束到的具体桶位 {color_idx: [bucket_indices]}
    bucket_constraints = {}
    for i, item in enumerate(json_data):
        position = item.get("POSITION")
        if position:
            for bucket in position:
                try:
                    bucket_idx = BUCKETS.index(bucket)
                    g_idx = BUCKET_TO_GROUP[bucket_idx]
                    pos_constraints.setdefault(g_idx, []).append(i)
                    # 记录该颜色被约束到的具体桶位
                    bucket_constraints.setdefault(i, []).append(bucket_idx)
                except ValueError:
                    pass

    # Stage 1: 粗搜种子
    seeds = []
    
    # 从输入参数获取搜索范围，使用默认值
    xmin = linkargs.get("xmin", 1.1)
    x1_3max = linkargs.get("x1_3max", 4.0)
    x4max = linkargs.get("x4max", 6.0)
    xstep = linkargs.get("xstep", 0.1)
    
    search_stages = [
        {
            'label': f'{xmin}-{x1_3max}/{x4max}',
            'X1_range': (xmin, x1_3max, xstep),
            'X2_range': (xmin, x1_3max, xstep),
            'X3_range': (xmin, x1_3max, xstep),
            'X4_range': (xmin, x4max, xstep)
        }
    ]
    
    start_time = time.time()
    for stage in search_stages:
        # 检查是否超时
        if time.time() - start_time > MAX_RUNTIME:
            print(f"运行超时，已运行 {time.time() - start_time:.1f} 秒，强制停止")
            break
        
        x_min, x_max, x_step = stage['X1_range']
        x4_min, x4_max, x4_step = stage['X4_range']
        if len(seeds) >= MAX_SEEDS: break
        x_vals = np.arange(x_min, x_max, x_step)
        x4_vals = np.arange(x4_min, x4_max, x4_step)
        
        for x1, x2, x3, x4 in itertools.product(x_vals, x_vals, x_vals, x4_vals):
            # 每次迭代检查超时
            if time.time() - start_time > MAX_RUNTIME:
                print(f"运行超时，已运行 {time.time() - start_time:.1f} 秒，强制停止")
                break
            
            if not (x4 > x1 + 0.05 and x4 > x3 + 0.05): continue
            if x4/x1 > X4_RATIO_LIMIT or x4/x3 > X4_RATIO_LIMIT: continue
            
            w_seed = np.array([1/x1, 1/x4, 1.0, 1/x2, 1/x3])
            D_seed = np.sum(w_seed * GROUP_SIZES)
            if not (D_RANGE[0] < D_seed < D_RANGE[1]): continue
            
            res = fast_backtrack(targets, w_seed, D_seed, len(json_data), pos_constraints)
            for af, dev, pcts in res:
                seeds.append({
                    'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4, 
                    'assign': af, 'dev': dev, 'D': D_seed, 'final_pcts': pcts,
                    'stage_label': stage['label']
                })
            
            if len(seeds) >= MAX_SEEDS: break
        else:
            continue  # 如果内层循环正常结束，继续外层循环
        break  # 如果内层循环因超时或种子数限制而break，则跳出外层循环
    
    # 检查是否有有效结果
    if not seeds:
        return json.dumps({'error': '运行超时或未找到有效解', 'results': [], 'runtime': time.time() - start_time})
    else:
        print(f"搜索完成，找到 {len(seeds)} 个种子，运行时间 {time.time() - start_time:.1f} 秒")

    # Stage 2: 矢量化精修
    # 排序：误差越小越好，total_feed_speed_D 越大越好
    refined = refine_vectorized(sorted(seeds, key=lambda x: (x['dev'], -x['D'])), targets, json_data)
    
    # 结果排序并输出：误差越小越好，total_feed_speed_D 越大越好
    top_results = sorted(refined, key=lambda x: (x['dev'], -x['D']))[:TOP_N]
    
    final_results = []
    for s in top_results:
        w_map = [1/s['X1'], 1/s['X4'], 1.0, 1/s['X2'], 1/s['X3']]
        
        # 根据 bucket_constraints 调整组内分配顺序
        # bucket_constraints: {color_idx: [bucket_indices]} - 记录每个颜色被约束到的桶位
        # 组1(BG): 桶位1(B)和6(G), 组2(CD): 桶位2(C)和3(D), 组3(EF): 桶位4(E)和5(F)
        adjusted_assign = s['assign'].copy()
        
        # 检查组1 (BG): 桶位1和6
        b_color = adjusted_assign[1]  # B位置当前颜色索引
        g_color = adjusted_assign[6]  # G位置当前颜色索引
        # 检查B位置的颜色是否被约束到G位置(桶位6)，或者G位置的颜色是否被约束到B位置(桶位1)
        b_constrained_to_g = b_color in bucket_constraints and 6 in bucket_constraints[b_color]
        g_constrained_to_b = g_color in bucket_constraints and 1 in bucket_constraints[g_color]
        if b_constrained_to_g or g_constrained_to_b:
            adjusted_assign[1], adjusted_assign[6] = adjusted_assign[6], adjusted_assign[1]
        
        # 检查组2 (CD): 桶位2和3
        c_color = adjusted_assign[2]
        d_color = adjusted_assign[3]
        c_constrained_to_d = c_color in bucket_constraints and 3 in bucket_constraints[c_color]
        d_constrained_to_c = d_color in bucket_constraints and 2 in bucket_constraints[d_color]
        if c_constrained_to_d or d_constrained_to_c:
            adjusted_assign[2], adjusted_assign[3] = adjusted_assign[3], adjusted_assign[2]
        
        # 检查组3 (EF): 桶位4和5
        e_color = adjusted_assign[4]
        f_color = adjusted_assign[5]
        e_constrained_to_f = e_color in bucket_constraints and 5 in bucket_constraints[e_color]
        f_constrained_to_e = f_color in bucket_constraints and 4 in bucket_constraints[f_color]
        if e_constrained_to_f or f_constrained_to_e:
            adjusted_assign[4], adjusted_assign[5] = adjusted_assign[5], adjusted_assign[4]
        
        assign_list = []
        for i, b_name in enumerate(BUCKETS):
            g_idx = BUCKET_TO_GROUP[i]
            color_idx = adjusted_assign[i]
            
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
                'colorsho': json_data[color_idx].get('MFMSHO', ''),
                'x': round(x_val, 2),
                'speed': round(w_map[g_idx] / s['D'] * 100.0, 4)
            })
            
        final_results.append({
            'X1': round(s['X1'], 4),
            'X2': round(s['X2'], 4),
            'X3': round(s['X3'], 4),
            'X4': round(s['X4'], 4),
            'cum_error': round(s['dev'], 2),
            'total_feed_speed_D': round(s['D'], 6),
            'stage_label': s.get('stage_label', 'Unknown'),
            'assign': assign_list,
            'colors': [
                f"{json_data[i]['MFMDES']}: {round(targets_pct[i], 2)} -> {round(s['final_pcts'][i]*100.0, 2)} ({format(abs(s['final_pcts'][i]*100.0 - targets_pct[i]), '.2f')}%)"
                for i in range(len(json_data))
            ]
        })

    # return json.dumps({'results': final_results}, indent=2, ensure_ascii=False)
    return json.dumps({'results': final_results}, ensure_ascii=False)

# ======================
# 程序入口
# ======================
if __name__ == "__main__":
    json_str = """{
    "pyFile": "bt8",
    "xmin": 1.05,
    "x1_3max": 2.5,
    "x4max": 5.0,
    "xstep": 0.1,
    "data": 
[
  {
    "MFMLIN": 62,
    "MFMDES": "SWP本白 VG010M ",
    "MFMSHO": "SWP本白",
    "MATRATCALC": 36.900000,
    "PRIORITY": 0,
    "POSITION": ""
  },
  {
    "MFMLIN": 70,
    "MFMDES": "BC02W VE001M-U-001 ",
    "MFMSHO": "BC02W",
    "MATRATCALC": 11.200000,
    "PRIORITY": 0,
    "POSITION": "A"
  },
  {
    "MFMLIN": 80,
    "MFMDES": "WJ 白棉 VG055M ",
    "MFMSHO": "WJ 白棉",
    "MATRATCALC": 29.800000,
    "PRIORITY": 0,
    "POSITION": ""
  },
  {
    "MFMLIN": 81,
    "MFMDES": "W 白棉 VG055M ",
    "MFMSHO": "W 白棉",
    "MATRATCALC": 22.100000,
    "PRIORITY": 0,
    "POSITION": ""
  }
]
,"data1":[
  {
    "MFMLIN": 50,
    "MFMDES": "W 白棉条 VB050M ",
    "MFMSHO": "W 白棉条",
    "MATRATCALC": 41.390000,
    "PRIORITY": 0,
    "POSITION": ""
  },
  {
    "MFMLIN": 60,
    "MFMDES": "K004W20 VE001M-001 ",
    "MFMSHO": "K004W20",
    "MATRATCALC": 42.250000,
    "PRIORITY": 0,
    "POSITION": ""
  },
  {
    "MFMLIN": 70,
    "MFMDES": "Y006W VE001M-001 ",
    "MFMSHO": "Y006W",
    "MATRATCALC": 7.920000,
    "PRIORITY": 0,
    "POSITION": "F"
  },
  {
    "MFMLIN": 80,
    "MFMDES": "K002W20 VE001M-001 ",
    "MFMSHO": "K002W20",
    "MATRATCALC": 3.610000,
    "PRIORITY": 0,
    "POSITION": ""
  },
  {
    "MFMLIN": 90,
    "MFMDES": "Y01W20 VE001M-001 ",
    "MFMSHO": "Y01W20",
    "MATRATCALC": 4.830000,
    "PRIORITY": 0,
    "POSITION": ""
  }
]
    }"""

    result = linkrun(json_str)
    print(result)