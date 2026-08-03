#!/usr/bin/env python3
import itertools
import time
import json
import numpy as np
import random
import scipy.optimize

# ======================
# 算法核心参数
# 2026年3月3日, 优化计算速度，当前使用版本
# 2026年5月29日，相同分布签名去除处理
# 2026年5月29日，相邻颜色相同时对调，打乱颜色
# ======================
TOL = 0.02                  # 粗搜容差 1.5%
TOP_N = 100                   # 结果返回的数量
MAX_SEEDS = 150              # 限制种子数量，平衡速度与精度
MAX_PER_X_SIGNATURE = 5      # 每种 X 区域签名最多保留的结果数
PRIORITY_ERROR_THRESHOLD = 0.0005   # 0.05%
NON_PRIORITY_ERROR_THRESHOLD = 0.005 # 0.5%
D_RANGE = (0.5, 10.0)
MAX_RUNTIME = 60             # 最大运行时间60秒

# 物理结构定义
BUCKETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
# 速度组映射：A:G0, B:G0, C:G1, D:G1, E:G2, F:G2, G:G3, H:G3
# AB同速(1/X1), CD固定(1.0), EF同速(1/X2), GH同速(1/X3)
GROUP_SIZES = np.array([2, 2, 2, 2])  
BUCKET_TO_GROUP = np.array([0, 0, 1, 1, 2, 2, 3, 3])

# ======================
# 核心计算函数
# ======================

def fast_backtrack(targets, weights, D, num_colors, pos_constraints):
    """
    高度优化的回溯算法
    """
    results = []
    current_sums = [0.0] * num_colors
    current_assign = [None] * 4

    def dfs(g_idx):
        if g_idx == 4:
            total_err = 0.0
            final_pcts = []
            for i in range(num_colors):
                pct = current_sums[i] / D
                err = abs(pct - targets[i])
                total_err += err
                final_pcts.append(pct)
                if err > 0.015: return # 种子阶段单色误差阈值，略微放宽
            
            # 还原为 8 桶位分配 (存储颜色索引)
            # 每组大小为2，按顺序展开到具体桶位
            flat = [None] * 8
            # 组0 (AB): 桶位0=A, 1=B
            c1, c2 = current_assign[0]
            flat[0], flat[1] = (c1, c2) if c1 <= c2 else (c2, c1)
            # 组1 (CD): 桶位2=C, 3=D
            c1, c2 = current_assign[1]
            flat[2], flat[3] = (c1, c2) if c1 <= c2 else (c2, c1)
            # 组2 (EF): 桶位4=E, 5=F
            c1, c2 = current_assign[2]
            flat[4], flat[5] = (c1, c2) if c1 <= c2 else (c2, c1)
            # 组3 (GH): 桶位6=G, 7=H
            c1, c2 = current_assign[3]
            flat[6], flat[7] = (c1, c2) if c1 <= c2 else (c2, c1)
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
            
            # 剪枝：单色占比不能偏离目标值太远（上限+下限）
            pass_check = True
            for i in range(num_colors):
                pct = current_sums[i] / D
                if pct > targets[i] + 0.03:
                    pass_check = False; break
            if pass_check:
                remaining = sum(weights[g] for g in range(g_idx + 1, 4))
                for i in range(num_colors):
                    max_possible = (current_sums[i] + remaining) / D
                    if max_possible < targets[i] - 0.03:
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
    
    # 定义微调网格 7^3 = 343 组
    offsets = np.linspace(-0.03, 0.03, 7) 
    d1, d2, d3 = np.meshgrid(offsets, offsets, offsets)
    d1, d2, d3 = d1.ravel(), d2.ravel(), d3.ravel()
    
    target_arr = np.array(targets)
    is_priority = np.array([item["PRIORITY"] for item in json_data])
    refined_results = []
    
    # 仅精修前 TOP_N 个优质种子
    for sol in seeds[:TOP_N]:
        mX1, mX2, mX3 = sol['X1']+d1, sol['X2']+d2, sol['X3']+d3
        
        # 物理与工艺约束过滤
        mask = (mX1 >= 1.0) & (mX2 >= 1.0) & (mX3 >= 1.0)
        
        if not np.any(mask): 
            refined_results.append(sol)
            continue
            
        mX1, mX2, mX3 = mX1[mask], mX2[mask], mX3[mask]
        
        # 矢量化计算总牵伸 D
        # 组速度: G0(AB)=1/X1, G1(CD)=1.0, G2(EF)=1/X2, G3(GH)=1/X3
        W = np.array([1/mX1, np.ones_like(mX1), 1/mX2, 1/mX3]) # 4 x N
        D_vec = np.dot(GROUP_SIZES, W) # N
        
        # 计算该分配方案下各颜色的权重矩阵
        c_weights = np.zeros((len(targets), 4))
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
                'X1': mX1[best_idx], 'X2': mX2[best_idx], 'X3': mX3[best_idx],
                'assign': sol['assign'], 'dev': total_errs[best_idx], 'D': D_vec[best_idx],
                'final_pcts': final_pcts_matrix[:, best_idx].tolist(),
                'stage_label': sol.get('stage_label', 'Unknown')
            })
        else:
            refined_results.append(sol)
            
    return refined_results


def select_diverse_top(results, top_n=TOP_N):
    """
    多样性感知的 Top-N 选择：
    按误差排序，但在最终输出时跳过 (X区域, 分配方案) 都相同的结果，
    确保输出涵盖不同牵伸倍数组合和不同颜色分配方案。
    """
    if not results:
        return []
    
    sorted_results = sorted(results, key=lambda x: (round(x['dev'], 2), -x['D']))
    
    selected = []
    used_x_zones = set()       # (X1~0.3, X2~0.3, X3~0.3)
    used_assign_sigs = set()   # tuple of 8 assignments
    
    for s in sorted_results:
        # X 区域签名：~0.3 步长聚类
        x_zone = (round(s['X1'] / 0.3) * 0.3,
                  round(s['X2'] / 0.3) * 0.3,
                  round(s['X3'] / 0.3) * 0.3)
        # 分配方案签名
        assign_sig = tuple(s['assign'])
        
        # (X区域, 分配) 同时相同才跳过
        if x_zone in used_x_zones and assign_sig in used_assign_sigs:
            continue
        
        selected.append(s)
        used_x_zones.add(x_zone)
        used_assign_sigs.add(assign_sig)
        
        if len(selected) >= top_n:
            break
    
    # 不足 TOP_N 则从剩余补充
    if len(selected) < top_n:
        for s in sorted_results:
            if s not in selected:
                selected.append(s)
                if len(selected) >= top_n:
                    break
    
    return selected

def find_seeds_scipy(targets, num_colors, pos_constraints, bounds, max_seeds=MAX_SEEDS):
    """
    使用 scipy.optimize.minimize (Nelder-Mead) 进行多起点局部优化搜索
    """
    eval_cache = {}
    x_signature_count = {}
    seeds = []
    start_time = time.time()

    def objective(x):
        if time.time() - start_time > MAX_RUNTIME:
            return 100.0

        x1, x2, x3 = x
        penalty = 0.0

        # 连续惩罚：违反越多惩罚越大，为 Nelder-Mead 提供梯度方向
        if x1 < 1.0: penalty += (1.0 - x1) * 0.5
        if x2 < 1.0: penalty += (1.0 - x2) * 0.5
        if x3 < 1.0: penalty += (1.0 - x3) * 0.5

        # 组速度: G0(AB)=1/X1, G1(CD)=1.0, G2(EF)=1/X2, G3(GH)=1/X3
        w = np.array([1/max(x1,1e-6), 1.0, 1/max(x2,1e-6), 1/max(x3,1e-6)])
        D = np.sum(w * GROUP_SIZES)
        if D <= D_RANGE[0]: penalty += (D_RANGE[0] - D) * 0.1
        if D >= D_RANGE[1]: penalty += (D - D_RANGE[1]) * 0.1

        # 连续惩罚 > 0 时，返回惩罚值（始终大于有效解的最大误差 ~0.06）
        if penalty > 0:
            return 0.1 + penalty

        key = (round(x1, 2), round(x2, 2), round(x3, 2))
        if key in eval_cache:
            return eval_cache[key][0]

        res = fast_backtrack(targets, w, D, num_colors, pos_constraints)
        if not res:
            eval_cache[key] = (0.5, [], D)
            return 0.5

        min_err = min(r[1] for r in res)
        eval_cache[key] = (min_err, res, D)
        return min_err

    scipy_bounds = [
        (max(1.0, bounds['x1'][0]), bounds['x1'][1]),
        (max(1.0, bounds['x2'][0]), bounds['x2'][1]),
        (max(1.0, bounds['x3'][0]), bounds['x3'][1])
    ]

    n_explore = 200
    for i in range(n_explore):
        if len(seeds) >= max_seeds:
            break
        if time.time() - start_time > MAX_RUNTIME * 0.7:
            print(f"搜索超时，已运行 {time.time() - start_time:.1f} 秒，共 {i} 轮探索")
            break

        x0 = [round(random.uniform(*b), 2) for b in scipy_bounds]

        scipy.optimize.minimize(
            objective, x0, method='Nelder-Mead',
            bounds=scipy_bounds,
            options={'maxiter': 150, 'xatol': 0.01, 'fatol': 0.0001}
        )

    n_local = 100
    for i in range(n_local):
        if len(seeds) >= max_seeds:
            break
        if time.time() - start_time > MAX_RUNTIME * 0.9:
            break

        good_keys = [k for k, v in eval_cache.items() if v[0] < 0.015]
        if not good_keys:
            break

        base = random.choice(good_keys)
        x0 = [round(base[j] + random.uniform(-0.08, 0.08), 2) for j in range(3)]
        for j in range(3):
            x0[j] = max(scipy_bounds[j][0], min(scipy_bounds[j][1], x0[j]))

        scipy.optimize.minimize(
            objective, x0, method='Nelder-Mead',
            bounds=scipy_bounds,
            options={'maxiter': 80, 'xatol': 0.01, 'fatol': 0.0001}
        )

    for key, (err, assignments, D) in eval_cache.items():
        if err >= 0.015:
            continue
        x_sig = (round(key[0], 1), round(key[1], 1), round(key[2], 1))
        if x_signature_count.get(x_sig, 0) >= MAX_PER_X_SIGNATURE:
            continue
        x_signature_count[x_sig] = x_signature_count.get(x_sig, 0) + 1
        for af, dev, pcts in assignments:
            seeds.append({
                'X1': key[0], 'X2': key[1], 'X3': key[2],
                'assign': af, 'dev': dev, 'D': D, 'final_pcts': pcts,
                'stage_label': 'scipy'
            })

    # 诊断信息：无种子时帮助定位问题
    if not seeds:
        errors = [v[0] for v in eval_cache.values()]
        valid_errors = [e for e in errors if e < 0.5]
        good_errors = [e for e in valid_errors if e < 0.015]
        print(f"[诊断] 评估点数: {len(eval_cache)}, 有效X组合: {len(valid_errors)}, 误差<1.5%: {len(good_errors)}")
        if valid_errors:
            print(f"[诊断] 最佳误差: {min(valid_errors):.4f}, 最差误差: {max(valid_errors):.4f}")
        else:
            print(f"[诊断] 所有X组合均未通过物理约束或无法分配颜色")

    return seeds

def find_seeds_grid(targets, num_colors, pos_constraints, bounds, max_seeds=MAX_SEEDS):
    """
    确定性网格搜索，作为 scipy 搜索的兜底方案，确保结果稳定可复现
    """
    start_time = time.time()
    seeds = []
    x_sig_count = {}

    grid_step = 0.03
    x1_vals = np.arange(bounds['x1'][0], bounds['x1'][1] + grid_step/2, grid_step)
    x2_vals = np.arange(bounds['x2'][0], bounds['x2'][1] + grid_step/2, grid_step)
    x3_vals = np.arange(bounds['x3'][0], bounds['x3'][1] + grid_step/2, grid_step)

    total = len(x1_vals) * len(x2_vals) * len(x3_vals)
    checked = 0

    for x1 in x1_vals:
        if len(seeds) >= max_seeds or time.time() - start_time > MAX_RUNTIME * 0.8:
            break
        for x2 in x2_vals:
            if len(seeds) >= max_seeds or time.time() - start_time > MAX_RUNTIME * 0.8:
                break
            for x3 in x3_vals:
                checked += 1
                if len(seeds) >= max_seeds or time.time() - start_time > MAX_RUNTIME * 0.8:
                    break

                x1r, x2r, x3r = round(x1, 2), round(x2, 2), round(x3, 2)

                if x1r < 1.0 or x2r < 1.0 or x3r < 1.0:
                    continue

                # 组速度: G0(AB)=1/X1, G1(CD)=1.0, G2(EF)=1/X2, G3(GH)=1/X3
                w = np.array([1/x1r, 1.0, 1/x2r, 1/x3r])
                D = np.sum(w * GROUP_SIZES)
                if not (D_RANGE[0] < D < D_RANGE[1]):
                    continue

                res = fast_backtrack(targets, w, D, num_colors, pos_constraints)
                if not res:
                    continue

                for af, dev, pcts in res:
                    if dev >= 0.015:
                        continue
                    x_sig = (round(x1r, 1), round(x2r, 1), round(x3r, 1))
                    if x_sig_count.get(x_sig, 0) >= 2:
                        continue
                    x_sig_count[x_sig] = x_sig_count.get(x_sig, 0) + 1
                    seeds.append({
                        'X1': x1r, 'X2': x2r, 'X3': x3r,
                        'assign': af, 'dev': dev, 'D': D, 'final_pcts': pcts,
                        'stage_label': 'grid'
                    })

    print(f"网格兜底: 遍历 {checked}/{total} 个点, 找到 {len(seeds)} 个种子")
    return seeds

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

    # Stage 1: 使用 scipy.optimize.minimize 进行多起点局部优化搜索
    start_time = time.time()

    xmin = linkargs.get("xmin", 1.03)
    xmax = linkargs.get("xmax", 3.5)

    bounds = {
        'x1': (xmin, xmax),
        'x2': (xmin, xmax),
        'x3': (xmin, xmax)
    }

    seeds = find_seeds_scipy(targets, len(json_data), pos_constraints, bounds)

    # 兜底：scipy 随机搜索不稳定时，用确定性网格搜索保证覆盖率
    MIN_SEEDS_FALLBACK = 10
    if len(seeds) < MIN_SEEDS_FALLBACK:
        print(f"scipy 仅找到 {len(seeds)} 个种子，启动网格搜索兜底...")
        grid_seeds = find_seeds_grid(targets, len(json_data), pos_constraints, bounds)
        # 合并去重（按 X 签名）
        seen_sigs = set()
        for s in seeds:
            seen_sigs.add((round(s['X1'], 1), round(s['X2'], 1), round(s['X3'], 1)))
        for gs in grid_seeds:
            sig = (round(gs['X1'], 1), round(gs['X2'], 1), round(gs['X3'], 1))
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                seeds.append(gs)

    # 检查是否有有效结果
    if not seeds:
        elapsed = round(time.time() - start_time, 1)
        if elapsed >= MAX_RUNTIME * 0.7:
            reason = f'搜索超时 ({elapsed:.0f}s >= {MAX_RUNTIME*0.7:.0f}s)'
        else:
            reason = f'未找到误差<1.5%的有效解 (搜索{elapsed:.0f}s)，请放宽xmin/xmax范围或检查POSITION约束是否过严'
        return json.dumps({'error': reason, 'results': [], 'runtime': elapsed}, ensure_ascii=False)
    else:
        print(f"搜索完成，找到 {len(seeds)} 个种子，运行时间 {time.time() - start_time:.1f} 秒")

    # Stage 2: 按分配方案去重（同颜色分布保留误差最小的 1 个牵伸倍数）
    seeds_sorted = sorted(seeds, key=lambda x: x['dev'])
    deduped_seeds = []
    seen_assign = set()
    for s in seeds_sorted:
        sig = tuple(s['assign'])
        if sig not in seen_assign:
            seen_assign.add(sig)
            deduped_seeds.append(s)

    # Stage 3: 矢量化精修
    # 排序：误差越小越好，total_feed_speed_D 越大越好（误差四舍五入到两位小数后比较）
    refined = refine_vectorized(sorted(deduped_seeds, key=lambda x: (round(x['dev'], 2), -x['D'])), targets, json_data)

    # Stage 3: 多样性感知的 Top-N 选择
    # 按误差排序，但跳过 (X区域, 分配方案) 完全相同的冗余解
    top_results = select_diverse_top(refined, TOP_N)
    
    final_results = []
    for s in top_results:
        # 组速度: G0(AB)=1/X1, G1(CD)=1.0, G2(EF)=1/X2, G3(GH)=1/X3
        w_map = [1/s['X1'], 1.0, 1/s['X2'], 1/s['X3']]
        
        # 根据 bucket_constraints 调整组内分配顺序
        # 组0(AB): 桶位0和1, 组1(CD): 桶位2和3, 组2(EF): 桶位4和5, 组3(GH): 桶位6和7
        adjusted_assign = s['assign'].copy()
        
        # 检查组0 (AB): 桶位0和1
        a_color = adjusted_assign[0]
        b_color = adjusted_assign[1]
        a_constrained_to_b = a_color in bucket_constraints and 1 in bucket_constraints[a_color]
        b_constrained_to_a = b_color in bucket_constraints and 0 in bucket_constraints[b_color]
        if a_constrained_to_b or b_constrained_to_a:
            adjusted_assign[0], adjusted_assign[1] = adjusted_assign[1], adjusted_assign[0]
        
        # 检查组1 (CD): 桶位2和3
        c_color = adjusted_assign[2]
        d_color = adjusted_assign[3]
        c_constrained_to_d = c_color in bucket_constraints and 3 in bucket_constraints[c_color]
        d_constrained_to_c = d_color in bucket_constraints and 2 in bucket_constraints[d_color]
        if c_constrained_to_d or d_constrained_to_c:
            adjusted_assign[2], adjusted_assign[3] = adjusted_assign[3], adjusted_assign[2]
        
        # 检查组2 (EF): 桶位4和5
        e_color = adjusted_assign[4]
        f_color = adjusted_assign[5]
        e_constrained_to_f = e_color in bucket_constraints and 5 in bucket_constraints[e_color]
        f_constrained_to_e = f_color in bucket_constraints and 4 in bucket_constraints[f_color]
        if e_constrained_to_f or f_constrained_to_e:
            adjusted_assign[4], adjusted_assign[5] = adjusted_assign[5], adjusted_assign[4]
        
        # 检查组3 (GH): 桶位6和7
        g_color = adjusted_assign[6]
        h_color = adjusted_assign[7]
        g_constrained_to_h = g_color in bucket_constraints and 7 in bucket_constraints[g_color]
        h_constrained_to_g = h_color in bucket_constraints and 6 in bucket_constraints[h_color]
        if g_constrained_to_h or h_constrained_to_g:
            adjusted_assign[6], adjusted_assign[7] = adjusted_assign[7], adjusted_assign[6]
        
        # 尝试跨组+组内交换组合，减少圆环相邻同色
        # X 值跟随组一起移动
        orig_x = [s['X1'], s['X2'], s['X3']]
        final_assign, final_x = optimize_arrangement(adjusted_assign, bucket_constraints, orig_x)
        
        assign_list = []
        for i, b_name in enumerate(BUCKETS):
            g_idx = BUCKET_TO_GROUP[i]
            color_idx = final_assign[i]
            
            # X 值显示逻辑
            if i in [0, 1]: x_val = final_x[0]          # AB: X1
            elif i in [2, 3]: x_val = 1.0                # CD: 固定1.0
            elif i in [4, 5]: x_val = final_x[1]          # EF: X2
            elif i in [6, 7]: x_val = final_x[2]          # GH: X3
            
            assign_list.append({
                'bucket': b_name,
                'color': json_data[color_idx]['MFMLIN'],
                'colordes': json_data[color_idx].get('MFMDES', ''),
                'colorsho': json_data[color_idx].get('MFMSHO', ''),
                'x': round(x_val, 2),
                'speed': round(w_map[g_idx] / s['D'] * 100.0, 4)
            })
            
        final_results.append({
            'X1': round(final_x[0], 2),
            'X2': round(final_x[1], 2),
            'X3': round(final_x[2], 2),
            'cum_error': round(s['dev'], 2),
            'total_feed_speed_D': round(s['D'], 6),
            'stage_label': s.get('stage_label', 'Unknown'),
            'assign': assign_list,
            'colors': [
                f"{json_data[i]['MFMDES']}: {round(targets_pct[i], 2)} -> {round(s['final_pcts'][i]*100.0, 2)} ({format(abs(s['final_pcts'][i]*100.0 - targets_pct[i]), '.2f')}%)"
                for i in range(len(json_data))
            ]
        })

    return json.dumps({'results': final_results}, indent=2, ensure_ascii=False)
    # return json.dumps({'results': final_results}, ensure_ascii=False)

def optimize_arrangement(assign, bucket_constraints, x_vals=None):
    """
    双层优化：
    1. 跨组交换：AB(X1)、EF(X2)、GH(X3) 三个可变组整体交换位置，
       牵伸倍数跟随颜色一起移动。CD 组(固定1.0)不参与跨组交换。
    2. 组内交换：A↔B(0↔1)、C↔D(2↔3)、E↔F(4↔5)、G↔H(6↔7)。
    
    选择圆环 A-B-C-D-E-F-G-H-A 上相邻同色最少的排布。
    返回 (best_assign, best_x_vals)。
    """
    # 先检查是否有相邻同色，没有则直接返回
    adj_init = sum(1 for i in range(8) if assign[i] == assign[(i + 1) % 8])
    if adj_init == 0:
        return assign, x_vals

    # 可变组：AB(0,1, X1), EF(4,5, X2), GH(6,7, X3)
    movable_groups = [
        [(0, 1), 'X1'],
        [(4, 5), 'X2'],
        [(6, 7), 'X3'],
    ]
    
    best_assign = assign[:]
    best_x = x_vals[:] if x_vals else None
    best_adj = adj_init
    
    # 组内交换：每组内部 A↔B / C↔D / E↔F / G↔H
    group_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    import itertools
    # 可变组跨组置换：3! = 6 种
    for perm in itertools.permutations(range(3)):
        # 构建新分配：CD 固定不动
        test = [None] * 8
        test[2] = assign[2]  # CD 固定
        test[3] = assign[3]
        
        # 可变组按 perm 重新排列
        for new_gi, old_gi in enumerate(perm):
            (a, b), _ = movable_groups[old_gi]
            (new_a, new_b), _ = movable_groups[new_gi]
            test[new_a] = assign[a]
            test[new_b] = assign[b]
        
        # 检查跨组约束
        valid = True
        for i in range(8):
            ci = test[i]
            if ci in bucket_constraints:
                if i not in bucket_constraints[ci]:
                    valid = False; break
        if not valid:
            continue
        
        # 组内交换：16 种
        for mask in range(16):
            inner_test = test[:]
            inner_valid = True
            for pi in range(4):
                if mask & (1 << pi):
                    a, b = group_pairs[pi]
                    ca, cb = inner_test[a], inner_test[b]
                    if ca in bucket_constraints and b not in bucket_constraints[ca]:
                        inner_valid = False; break
                    if cb in bucket_constraints and a not in bucket_constraints[cb]:
                        inner_valid = False; break
                    inner_test[a], inner_test[b] = cb, ca
            
            if not inner_valid:
                continue
            
            adj = sum(1 for i in range(8) if inner_test[i] == inner_test[(i + 1) % 8])
            if adj < best_adj:
                best_adj = adj
                best_assign = inner_test[:]
                # X 值跟随可变组移动
                if x_vals:
                    new_x = list(x_vals)  # 默认不变
                    for new_gi, old_gi in enumerate(perm):
                        if new_gi == old_gi:
                            continue
                        old_key = movable_groups[old_gi][1]
                        new_key = movable_groups[new_gi][1]
                        old_idx = ['X1', 'X2', 'X3'].index(old_key)
                        new_idx = ['X1', 'X2', 'X3'].index(new_key)
                        new_x[new_idx] = x_vals[old_idx]
                    best_x = new_x
                if best_adj == 0:
                    return best_assign, best_x

    return best_assign, best_x

# ======================
# 程序入口
# ======================
if __name__ == "__main__":
    json_str = """{
    "pyFile": "bt8",
    "xmin": 1.03,
    "xmax": 3.5,
    "xstep": 0.1,
    "data": 
[
  {
    "MFMLIN": 62,
    "MFMDES": "SWP本白 VG010M ",
    "MFMSHO": "SWP本白",
    "MATRATCALC": 36.900000,
    "PRIORITY": 0,
    "POSITION": "AD"
  },
  {
    "MFMLIN": 70,
    "MFMDES": "BC02W VE001M-U-001 ",
    "MFMSHO": "BC02W",
    "MATRATCALC": 11.200000,
    "PRIORITY": 0,
    "POSITION": ""
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
,"data2":[
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
    "POSITION": "ACE"
  },
  {
    "MFMLIN": 70,
    "MFMDES": "Y006W VE001M-001 ",
    "MFMSHO": "Y006W",
    "MATRATCALC": 7.920000,
    "PRIORITY": 0,
    "POSITION": ""
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
],"data3":[
  {
    "MFMLIN": 10,
    "MFMDES": "K001W20 VE001M-001 ",
    "MFMSHO": "K001W20",
    "MATRATCALC": 2.530000,
    "PRIORITY": 0,
    "POSITION": ""
  },
  {
    "MFMLIN": 20,
    "MFMDES": "K003W20 VE001M-001 ",
    "MFMSHO": "K003W20",
    "MATRATCALC": 1.020000,
    "PRIORITY": 0,
    "POSITION": ""
  },
  {
    "MFMLIN": 30,
    "MFMDES": "W 白棉条 VB050M ",
    "MFMSHO": "W 白棉条",
    "MATRATCALC": 8.950000,
    "PRIORITY": 0,
    "POSITION": ""
  }
]
    }"""

    result = linkrun(json_str)
    print(result)