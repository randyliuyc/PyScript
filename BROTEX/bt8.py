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
# 单色误差统一按「相对目标色比」计算：|实际比例 - 目标比例| / 目标比例。
# 小色比用绝对百分点判定会被放过：例如目标 0.71%、实际 0，只差 0.71 个百分点，
# 绝对阈值下仍算合格；换成相对误差后该色是 100%，必然不合格。
SEED_REL_TOL = 0.10          # 种子阶段单色相对误差容差 10%（粗筛，需大于验收阈值）
PRUNE_REL_MARGIN = 0.20      # 回溯剪枝的相对余量 20%，必须大于 SEED_REL_TOL
TOP_N = 100                   # 结果返回的数量
MAX_PER_X_SIGNATURE = 5      # 每种 X 区域签名最多保留的种子数
PRIORITY_REL_THRESHOLD = 0.02    # 优先色相对误差阈值 2%
NORMAL_REL_THRESHOLD = 0.05      # 普通色相对误差阈值 5%
INFEASIBLE_COST = 10.0       # 搜索中「无可行分配」的目标值，必须高于任何可行解
D_RANGE = (0.5, 10.0)
MAX_RUNTIME = 30             # 默认最大运行时间（秒），可由输入 maxtime 覆盖

# 物理结构定义
BUCKETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
# 速度组映射：A:G0, B:G0, C:G1, D:G1, E:G2, F:G2, G:G3, H:G3
# AB同速(1/X1), CD固定(1.0), EF同速(1/X2), GH同速(1/X3)
GROUP_SIZES = np.array([2, 2, 2, 2])  
BUCKET_TO_GROUP = np.array([0, 0, 1, 1, 2, 2, 3, 3])

# ======================
# 核心计算函数
# ======================

def relative_error(actual, target):
    """单色相对误差 = |实际比例 - 目标比例| / 目标比例。

    目标为 0 时：该色不参与分配（实际也为 0）记 0；一旦被放入桶位则视为不可接受。
    """
    if target > 0:
        return abs(actual - target) / target
    return 0.0 if abs(actual) <= 1e-12 else float('inf')


def _min_single_slot_share(xmin, xmax):
    """当前 8 桶位结构下单色只占 1 个桶位时可能达到的最小占比（0-1）。

    组速度 G0=1/X1、G1=1.0、G2=1/X2、G3=1/X3，D=Σ(组大小×速度)。
    可变组取 1/xmax、其余组取最大速度 1/xmin 时，单桶占比最小。
    目标色比低于该值的颜色无法被分配，相对误差必然无法满足。
    """
    speed = 1.0 / xmax
    total = 2.0 * (speed + 1.0 + 2.0 / xmin)
    return speed / total


def fast_backtrack(targets, weights, D, num_colors, pos_constraints):
    """
    高度优化的回溯算法（单色误差按相对目标色比判定）
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
                err = relative_error(pct, targets[i])
                total_err += err
                final_pcts.append(pct)
                if err > SEED_REL_TOL: return  # 种子阶段单色相对误差阈值
            
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
        remaining = sum(GROUP_SIZES[g] * weights[g] for g in range(g_idx + 1, 4))

        # 所有组大小都为2，使用 combinations_with_replacement 允许同一颜色在组内重复
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

            # 剪枝：单色占比不能偏离目标值太远（相对余量，必须比种子容差更宽）
            pass_check = True
            for i in range(num_colors):
                pct = current_sums[i] / D
                if pct > targets[i] * (1.0 + PRUNE_REL_MARGIN):
                    pass_check = False; break
            if pass_check:
                for i in range(num_colors):
                    max_possible = (current_sums[i] + remaining) / D
                    if max_possible < targets[i] * (1.0 - PRUNE_REL_MARGIN):
                        pass_check = False; break
            
            if pass_check:
                current_assign[g_idx] = combo
                dfs(g_idx + 1)
            
            # 回溯
            for c_idx in combo: current_sums[c_idx] -= w

    dfs(0)
    return results

def refine_vectorized(seeds, targets, json_data, bounds=None):
    """
    NumPy 矢量化精修：并行计算小范围内的最优解
    """
    if not seeds: return []
    
    # 定义微调网格 3^3 = 27 组（±0.01）
    offsets = np.linspace(-0.01, 0.01, 3)
    d1, d2, d3 = np.meshgrid(offsets, offsets, offsets)
    d1, d2, d3 = d1.ravel(), d2.ravel(), d3.ravel()
    
    target_arr = np.array(targets)
    is_priority = np.array([bool(item.get("PRIORITY")) for item in json_data])
    refined_results = []
    
    # 获取边界约束
    x_min = bounds['x1'][0] if bounds else 1.0
    x_max = bounds['x1'][1] if bounds else 10.0
    
    # 仅精修前 TOP_N 个优质种子
    for sol in seeds[:TOP_N]:
        mX1, mX2, mX3 = sol['X1']+d1, sol['X2']+d2, sol['X3']+d3
        
        # 物理与工艺约束过滤：受输入 xmin/xmax 约束
        mask = (mX1 >= x_min) & (mX1 <= x_max) & \
               (mX2 >= x_min) & (mX2 <= x_max) & \
               (mX3 >= x_min) & (mX3 <= x_max)
        
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
        abs_err = np.abs(final_pcts_matrix - target_arr[:, None])
        safe_target = np.where(target_arr[:, None] > 0, target_arr[:, None], 1.0)
        # 单色相对误差；目标为 0 却分到桶位时视为不可接受（inf）
        errors_matrix = np.where((target_arr[:, None] <= 0) & (abs_err > 1e-12),
                                 np.inf, abs_err / safe_target)

        # 优先级误差过滤（相对误差）
        valid_mask = np.all(errors_matrix <= np.where(is_priority[:, None], PRIORITY_REL_THRESHOLD, NORMAL_REL_THRESHOLD), axis=0)
        
        if np.any(valid_mask):
            total_errs = np.sum(errors_matrix, axis=0)
            best_idx = np.argmin(np.where(valid_mask, total_errs, INFEASIBLE_COST))
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
    1. 跨组等效去重：将 assign 按 (AB段, CD段, EF段, GH段) 分组后排序，
       相同的颜色分布签名视为等效，只保留误差最小的。
    2. 按误差排序，输出 TOP_N 个不同颜色分布。
    """
    if not results:
        return []
    
    # 生成归一化签名：将4组排序后拼接，消除组间交换的等效性
    def color_dist_sig(assign):
        groups = [
            tuple(sorted(assign[0:2])),   # AB
            tuple(sorted(assign[2:4])),   # CD
            tuple(sorted(assign[4:6])),   # EF
            tuple(sorted(assign[6:8])),   # GH
        ]
        # 固定 CD 在位置1，其余3组排序
        movable = [groups[0], groups[2], groups[3]]
        movable.sort()
        return (movable[0], groups[1], movable[1], movable[2])
    
    # 按颜色分布签名分组，每组只保留误差最小的
    sig_best = {}
    for s in results:
        sig = color_dist_sig(s['assign'])
        if sig not in sig_best or s['dev'] < sig_best[sig]['dev']:
            sig_best[sig] = s
    
    sorted_results = sorted(sig_best.values(), key=lambda x: (x['dev'], -x['D']))
    
    return sorted_results[:top_n]

def find_seeds_scipy(targets, num_colors, pos_constraints, bounds, rng=None, deadline=None, verbose=False):
    """
    使用 scipy.optimize.minimize (Nelder-Mead) 进行多起点局部优化搜索

    rng: random.Random 实例；不传时使用固定种子，保证同一输入结果可复现
    deadline: 绝对时间戳（time.time() 基准）；不传时使用 MAX_RUNTIME 默认预算
    verbose: 是否输出诊断信息
    """
    rng = rng or random.Random(0)
    eval_cache = {}
    x_signature_count = {}
    seeds = []
    start_time = time.time()
    if deadline is None:
        deadline = start_time + MAX_RUNTIME
    explore_deadline = start_time + max(deadline - start_time, 0.0) * 0.7

    def objective(x):
        if time.time() > deadline:
            return 100.0

        # 先取整到 2 位小数再计算：缓存值、输出 X 与 dev 天然一一对应
        x1, x2, x3 = (round(float(v), 2) for v in x)
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

        # 连续惩罚 > 0 时，返回惩罚值（必须始终大于任何可行解的相对误差之和）
        if penalty > 0:
            return INFEASIBLE_COST + penalty

        key = (x1, x2, x3)
        if key in eval_cache:
            return eval_cache[key][0]

        res = fast_backtrack(targets, w, D, num_colors, pos_constraints)
        if not res:
            eval_cache[key] = (INFEASIBLE_COST, [], D)
            return INFEASIBLE_COST

        min_err = min(r[1] for r in res)
        eval_cache[key] = (min_err, res, D)
        return min_err

    scipy_bounds = [
        (max(1.0, bounds['x1'][0]), bounds['x1'][1]),
        (max(1.0, bounds['x2'][0]), bounds['x2'][1]),
        (max(1.0, bounds['x3'][0]), bounds['x3'][1])
    ]

    # 确定性粗网格：每维 5 个点（含边界），共 125 点。
    # 只在盒子的角落/边缘有解时（例如 8 色等分），随机起点容易漏掉。
    grid_axes = [np.linspace(b[0], b[1], 5) for b in scipy_bounds]
    for gx1 in grid_axes[0]:
        for gx2 in grid_axes[1]:
            for gx3 in grid_axes[2]:
                objective([round(float(gx1), 2), round(float(gx2), 2), round(float(gx3), 2)])

    n_explore = 200
    for i in range(n_explore):
        if time.time() > explore_deadline:
            if verbose:
                print(f"[bt8] 探索阶段超时，已运行 {time.time() - start_time:.1f} 秒，共 {i} 轮")
            break

        x0 = [round(rng.uniform(*b), 2) for b in scipy_bounds]

        scipy.optimize.minimize(
            objective, x0, method='Nelder-Mead',
            bounds=scipy_bounds,
            options={'maxiter': 150, 'xatol': 0.01, 'fatol': 0.0001}
        )

    n_local = 100
    for i in range(n_local):
        if time.time() > deadline:
            if verbose:
                print(f"[bt8] 局部搜索超时，已运行 {time.time() - start_time:.1f} 秒，共 {i} 轮")
            break

        good_keys = [k for k, v in eval_cache.items() if v[1]]
        if not good_keys:
            break

        base = rng.choice(good_keys)
        x0 = [round(base[j] + rng.uniform(-0.08, 0.08), 2) for j in range(3)]
        for j in range(3):
            x0[j] = max(scipy_bounds[j][0], min(scipy_bounds[j][1], x0[j]))

        scipy.optimize.minimize(
            objective, x0, method='Nelder-Mead',
            bounds=scipy_bounds,
            options={'maxiter': 80, 'xatol': 0.01, 'fatol': 0.0001}
        )

    for key, (_err, assignments, D_key) in eval_cache.items():
        # fast_backtrack 返回的每条分配都已满足单色相对容差 SEED_REL_TOL；
        # 这里只排除"无可行分配"的点。累计误差不能用它与单色容差比较：
        # 颜色多时累计误差天然会超过单色容差。
        if not assignments:
            continue
        x1, x2, x3 = key
        x_sig = (round(x1, 1), round(x2, 1), round(x3, 1))
        if x_signature_count.get(x_sig, 0) >= MAX_PER_X_SIGNATURE:
            continue
        x_signature_count[x_sig] = x_signature_count.get(x_sig, 0) + 1
        for af, dev, pcts in assignments:
            seeds.append({
                'X1': x1, 'X2': x2, 'X3': x3,
                'assign': af, 'dev': dev, 'D': D_key, 'final_pcts': pcts,
                'stage_label': 'scipy'
            })

    # 诊断信息：无种子时帮助定位问题
    if verbose and not seeds:
        with_assign = [v for v in eval_cache.values() if v[1]]
        print(f"[bt8] 评估点数: {len(eval_cache)}, 有可行分配: {len(with_assign)}")
        if with_assign:
            errs = [min(r[1] for r in v[1]) for v in with_assign]
            print(f"[bt8] 最佳累计误差: {min(errs):.4f}, 最差累计误差: {max(errs):.4f}")
        else:
            print("[bt8] 所有 X 组合均未通过物理约束或无法分配颜色")

    return seeds

# ======================
# 输入校验与结果契约
# ======================

def _error_result(message, runtime=0.0, code=None, params=None):
    """统一的失败返回（与其他 LinkPython 程序的 isSuccess 契约一致）。

    message 保留中文，便于日志与旧调用方；页面用 errorCode/errorParams
    按当前语言渲染提示。
    """
    payload = {
        'isSuccess': 0,
        'error': message,
        'message': message,
        'results': [],
        'runtime': round(runtime, 1),
    }
    if code:
        payload['errorCode'] = code
        payload['errorParams'] = params or {}
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def _validate_input(linkargs):
    """校验并抽取输入；不合法时抛出 ValueError（消息可直接展示给用户）"""
    if not isinstance(linkargs, dict):
        raise ValueError('输入必须是 JSON 对象')

    data = linkargs.get('data')
    if not isinstance(data, list) or not data:
        raise ValueError('data 必须是非空数组')

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f'data[{idx}] 必须是对象')
        if 'MATRATCALC' not in item:
            raise ValueError(f'data[{idx}] 缺少 MATRATCALC')
        if 'MFMLIN' not in item:
            raise ValueError(f'data[{idx}] 缺少 MFMLIN')
        try:
            value = float(item['MATRATCALC'])
        except (TypeError, ValueError):
            raise ValueError(f'data[{idx}].MATRATCALC 必须是数字')
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'data[{idx}].MATRATCALC 必须是有限的非负数')

        position = item.get('POSITION') or ''
        if not isinstance(position, str):
            raise ValueError(f'data[{idx}].POSITION 必须是字符串')
        illegal = sorted({ch for ch in position if ch not in BUCKETS})
        if illegal:
            raise ValueError(f'data[{idx}].POSITION 含非法桶位 {illegal}，只允许 A-H')

    if sum(float(item['MATRATCALC']) for item in data) <= 0:
        raise ValueError('所有 MATRATCALC 之和必须大于 0')

    try:
        xmin = float(linkargs.get('xmin', 1.03))
        xmax = float(linkargs.get('xmax', 3.5))
    except (TypeError, ValueError):
        raise ValueError('xmin/xmax 必须是数字')
    if not (np.isfinite(xmin) and np.isfinite(xmax)):
        raise ValueError('xmin/xmax 必须是有限数字')
    if xmin < 1.0:
        raise ValueError('xmin 必须 ≥ 1.0')
    if xmax <= xmin:
        raise ValueError('xmax 必须大于 xmin')

    try:
        seed = int(linkargs.get('seed', 0))
    except (TypeError, ValueError):
        raise ValueError('seed 必须是整数')

    try:
        maxtime = float(linkargs.get('maxtime', MAX_RUNTIME))
    except (TypeError, ValueError):
        raise ValueError('maxtime 必须是数字')
    if not np.isfinite(maxtime) or maxtime <= 0:
        raise ValueError('maxtime 必须是正数')

    verbose = bool(linkargs.get('debug', False))

    return data, xmin, xmax, seed, maxtime, verbose


# ======================
# 桶位约束工具
# ======================

def _slot_ok(color, bucket_idx, bucket_constraints):
    """颜色 color 是否允许放在 bucket_idx；未出现在约束表中的颜色不受限"""
    allowed = bucket_constraints.get(color)
    return allowed is None or bucket_idx in allowed


def satisfies_bucket_constraints(assign, bucket_constraints):
    """检查整条分配是否满足所有具体桶位约束"""
    return all(_slot_ok(color, i, bucket_constraints) for i, color in enumerate(assign))


def order_in_groups(assign, bucket_constraints):
    """
    在每个 2 格组内选择满足桶位约束的顺序（组内两格速度相同，交换不影响比例）。
    无法满足时返回 None。
    """
    out = list(assign)
    for a, b in ((0, 1), (2, 3), (4, 5), (6, 7)):
        if _slot_ok(out[a], a, bucket_constraints) and _slot_ok(out[b], b, bucket_constraints):
            continue
        out[a], out[b] = out[b], out[a]
        if not (_slot_ok(out[a], a, bucket_constraints) and _slot_ok(out[b], b, bucket_constraints)):
            return None
    return out


# ======================
# 主运行接口
# ======================

def linkrun(json_str):
    try:
        linkargs = json.loads(json_str)
        json_data, xmin, xmax, seed, maxtime, verbose = _validate_input(linkargs)
    except (json.JSONDecodeError, ValueError) as exc:
        return _error_result(f'输入错误: {exc}', code='input_invalid',
                             params={'detail': str(exc)})

    start_time = time.time()
    deadline = start_time + maxtime
    rng = random.Random(seed)

    # 目标比例归一化
    raw_targets = np.array([float(item["MATRATCALC"]) for item in json_data])
    targets = raw_targets / raw_targets.sum()
    targets_pct = targets * 100.0

    # 结构下界预检：单色至少占 1 个桶位，占比不可能低于当前结构的单桶最小值。
    # 低于它的颜色（如 0.71%）无论怎么搜索都放不进去，先给出明确结论，
    # 避免用绝对误差把它当成「只差 0.71 个百分点」而返回无意义方案。
    min_share = _min_single_slot_share(xmin, xmax)
    too_small = [
        (i, item) for i, item in enumerate(json_data)
        if targets[i] > 0 and targets[i] < min_share - 1e-12
    ]
    if too_small:
        too_small_text = [
            f"{item.get('MFMDES', '').strip() or item.get('MFMLIN', '')} {targets[i] * 100:.2f}%"
            for i, item in too_small
        ]
        detail = '、'.join(too_small_text)
        return _error_result(
            f'无法配色：{detail} 的目标色比低于该结构可实现的最小单色比例 '
            f'{min_share * 100:.2f}%（8 桶位、牵伸 {xmin:g}–{xmax:g}）。'
            f'这些颜色无法分配到桶位，单色相对误差必然超过 '
            f'{NORMAL_REL_THRESHOLD * 100:g}%，请调整色比或牵伸范围',
            code='color_below_structural_min',
            params={
                'colors': too_small_text,
                'min_pct': round(float(min_share) * 100, 2),
                'xmin': float(xmin),
                'xmax': float(xmax),
                'threshold_pct': round(float(NORMAL_REL_THRESHOLD) * 100, 2),
            })

    # 预处理位置约束
    # 支持多字符 POSITION，如 "ACF" 表示该颜色需要出现在 A、C、F 三个位置
    pos_constraints = {}
    # bucket_constraints: 记录每个颜色被约束到的具体桶位 {color_idx: [bucket_indices]}
    bucket_constraints = {}
    for i, item in enumerate(json_data):
        position = item.get("POSITION")
        if position:
            for bucket in position:
                bucket_idx = BUCKETS.index(bucket)
                g_idx = BUCKET_TO_GROUP[bucket_idx]
                pos_constraints.setdefault(g_idx, []).append(i)
                # 记录该颜色被约束到的具体桶位
                bucket_constraints.setdefault(i, []).append(bucket_idx)

    # Stage 1: 使用 scipy.optimize.minimize 进行多起点局部优化搜索
    bounds = {
        'x1': (xmin, xmax),
        'x2': (xmin, xmax),
        'x3': (xmin, xmax)
    }

    seeds = find_seeds_scipy(
        targets, len(json_data), pos_constraints, bounds,
        rng=rng, deadline=deadline, verbose=verbose)

    # 检查是否有有效结果
    if not seeds:
        elapsed = round(time.time() - start_time, 1)
        if time.time() >= deadline:
            return _error_result(
                f'搜索超时（已达到 maxtime={maxtime:g}s），请放宽 xmin/xmax 或增大 maxtime',
                elapsed, code='search_timeout', params={'maxtime': maxtime})
        return _error_result(
            f'未找到单色相对误差<{SEED_REL_TOL * 100:.1f}%的有效解（搜索 {elapsed:g}s），'
            f'请放宽 xmin/xmax 范围或检查 POSITION 约束是否过严',
            elapsed, code='no_seed',
            params={'tol_pct': round(SEED_REL_TOL * 100, 1), 'seconds': elapsed})
    elif verbose:
        print(f"[bt8] 搜索完成，找到 {len(seeds)} 个种子，运行时间 {time.time() - start_time:.1f} 秒")

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
    # 排序：误差越小越好，total_feed_speed_D 越大越好
    refined = refine_vectorized(sorted(deduped_seeds, key=lambda x: (x['dev'], -x['D'])), targets, json_data, bounds)

    # Stage 3: 多样性感知的 Top-N 选择
    # 按误差排序，但跳过 (X区域, 分配方案) 完全相同的冗余解
    top_results = select_diverse_top(refined, TOP_N)
    
    final_results = []
    constraint_rejected = 0
    error_rejected = 0
    closest_error = None   # (相对误差, 名称, 目标%, 实际%)，取超标最少的一条说明原因
    for s in top_results:
        # 组内顺序：按桶位约束选择满足约束的排列（组内两格速度相同，交换不影响比例）
        ordered_assign = order_in_groups(s['assign'], bucket_constraints)
        if ordered_assign is None:
            constraint_rejected += 1
            continue

        # 尝试跨组+组内交换组合，减少圆环相邻同色；X 值跟随组一起移动
        orig_x = [s['X1'], s['X2'], s['X3']]
        final_assign, final_x = optimize_arrangement(ordered_assign, bucket_constraints, orig_x)

        if not satisfies_bucket_constraints(final_assign, bucket_constraints):
            constraint_rejected += 1
            continue

        # 组速度按最终位置计算：G0(AB)=1/X1, G1(CD)=1.0, G2(EF)=1/X2, G3(GH)=1/X3
        w_map = [1.0 / final_x[0], 1.0, 1.0 / final_x[1], 1.0 / final_x[2]]
        D_final = float(np.sum(np.array(w_map) * GROUP_SIZES))

        # 单色相对误差判定：
        # - 普通色超出阈值 → 该方案不可行，直接丢弃（小色比不会被放过）
        # - 优先色超出阈值 → 保留方案但记为「优先色未达标」，排到后面
        warnings = []
        priority_ok = True
        normal_over = False
        for i, item in enumerate(json_data):
            is_priority = bool(item.get("PRIORITY"))
            err = relative_error(s['final_pcts'][i], targets[i])
            threshold = PRIORITY_REL_THRESHOLD if is_priority else NORMAL_REL_THRESHOLD
            if is_priority and err > PRIORITY_REL_THRESHOLD:
                priority_ok = False
            if err > threshold + 1e-12:
                warnings.append(
                    f"{item.get('MFMDES', '')}: 相对误差 {err * 100:.2f}% 超过阈值 {threshold * 100:.2f}%"
                    + ("（优先色）" if is_priority else "")
                )
                if not is_priority:
                    normal_over = True
                    if closest_error is None or err < closest_error[0]:
                        closest_error = (
                            err,
                            item.get('MFMDES', '').strip() or item.get('MFMLIN', ''),
                            targets[i] * 100.0,
                            s['final_pcts'][i] * 100.0,
                        )
        if normal_over:
            error_rejected += 1
            continue

        # cum_error 保持原语义（绝对百分点差之和，写入 MFGMATBTS.CUMVAR）；
        # 另给出相对误差之和，供排序与页面展示。
        abs_sum = float(np.sum(np.abs(np.array(s['final_pcts']) - targets)))
        rel_sum = float(np.sum([
            relative_error(s['final_pcts'][i], targets[i])
            for i in range(len(json_data))
        ]))

        assign_list = []
        for i, b_name in enumerate(BUCKETS):
            g_idx = BUCKET_TO_GROUP[i]
            color_idx = final_assign[i]

            # X 值显示逻辑
            if i in [0, 1]: x_val = final_x[0]          # AB: X1
            elif i in [2, 3]: x_val = 1.0                # CD: 固定1.0
            elif i in [4, 5]: x_val = final_x[1]          # EF: X2
            else: x_val = final_x[2]                      # GH: X3

            assign_list.append({
                'bucket': b_name,
                'color': json_data[color_idx]['MFMLIN'],
                'colordes': json_data[color_idx].get('MFMDES', ''),
                'colorsho': json_data[color_idx].get('MFMSHO', ''),
                'x': round(x_val, 2),
                'speed': round(w_map[g_idx] / D_final * 100.0, 4)
            })

        final_results.append({
            'X1': round(final_x[0], 2),
            'X2': round(final_x[1], 2),
            'X3': round(final_x[2], 2),
            # cum_error 保持旧语义（绝对差之和，供 MFGMATBTS.CUMVAR）；
            # cum_rel_error 是各颜色相对误差之和，用于排序与页面「累计相对误差」。
            # 数值可能极小（如 4.8e-05），保留 4 位小数后即以普通小数输出，
            # 不会出现科学计数法（E 字符），下游 SQL 可直接转为数字
            'cum_error': round(abs_sum, 4),
            'cum_error_pct': round(abs_sum * 100.0, 4),
            'cum_rel_error': round(rel_sum, 4),
            'cum_rel_error_pct': round(rel_sum * 100.0, 4),
            'total_feed_speed_D': round(D_final, 6),
            'stage_label': s.get('stage_label', 'Unknown'),
            'priority_ok': priority_ok,
            'warnings': warnings,
            'assign': assign_list,
            'colors': [
                f"{json_data[i].get('MFMDES', '')}: {round(targets_pct[i], 2)} -> "
                f"{round(s['final_pcts'][i] * 100.0, 2)} "
                f"({format(relative_error(s['final_pcts'][i], targets[i]) * 100.0, '.2f')}%)"
                for i in range(len(json_data))
            ]
        })

    if not final_results:
        elapsed = time.time() - start_time
        if error_rejected and closest_error:
            err, name, target_pct, actual_pct = closest_error
            rel_pct = round(float(err) * 100, 2)
            if not np.isfinite(rel_pct):
                rel_pct = 9999.0
            return _error_result(
                f'没有方案满足单色相对误差阈值 {NORMAL_REL_THRESHOLD * 100:g}%：'
                f'已过滤 {error_rejected} 个候选方案，最接近通过的方案里 {name} 目标 {target_pct:.2f}%、'
                f'实际 {actual_pct:.2f}%（相对误差 {err * 100:.2f}%）',
                elapsed, code='no_plan_meets_threshold',
                params={
                    'threshold_pct': round(float(NORMAL_REL_THRESHOLD) * 100, 2),
                    'rejected': int(error_rejected),
                    'name': name,
                    'target_pct': round(float(target_pct), 2),
                    'actual_pct': round(float(actual_pct), 2),
                    'rel_pct': rel_pct,
                })
        if constraint_rejected:
            return _error_result(
                f'POSITION 桶位约束无法满足：已过滤 {constraint_rejected} 个违反约束的候选方案，'
                f'请检查 POSITION 设置是否与目标比例相容',
                elapsed, code='position_infeasible',
                params={'rejected': int(constraint_rejected)})
        return _error_result('未找到有效方案', elapsed, code='no_valid_plan')

    # 优先色达标的方案排在前面，其次按累计相对误差排序
    final_results.sort(key=lambda r: (not r['priority_ok'], r['cum_rel_error']))

    return json.dumps({'isSuccess': 1, 'results': final_results}, ensure_ascii=False, allow_nan=False)

def optimize_arrangement(assign, bucket_constraints, x_vals=None):
    """
    双层优化：
    1. 跨组交换：AB(X1)、EF(X2)、GH(X3) 三个可变组整体交换位置，
       牵伸倍数跟随颜色一起移动。CD 组(固定1.0)不参与跨组交换。
    2. 组内交换：A↔B(0↔1)、C↔D(2↔3)、E↔F(4↔5)、G↔H(6↔7)。
    
    选择圆环 A-B-C-D-E-F-G-H-A 上相邻同色最少的排布。
    返回 (best_assign, best_x_vals)。
    """
    # 先检查是否有相邻同色；没有相邻同色且桶位约束已满足时直接返回
    adj_init = sum(1 for i in range(8) if assign[i] == assign[(i + 1) % 8])
    if adj_init == 0 and satisfies_bucket_constraints(assign, bucket_constraints):
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
    "xstep": 0.02,
    "data": 
[
  {
    "MFMLIN": 20,
    "MFMDES": "B024W J ",
    "MFMSHO": "B024W",
    "MATRATCALC": 10.000000,
    "PRIORITY": 0,
    "POSITION": ""
  },
  {
    "MFMLIN": 30,
    "MFMDES": "G004W J ",
    "MFMSHO": "G004W",
    "MATRATCALC": 10.000000,
    "PRIORITY": 0,
    "POSITION": ""
  },
  {
    "MFMLIN": 40,
    "MFMDES": "G008W QUA JC ",
    "MFMSHO": "G008W",
    "MATRATCALC": 80.000000,
    "PRIORITY": 0,
    "POSITION": ""
  }
]
,"data2":[
  {
    "MFMLIN": 10,
    "MFMDES": "R004W  ",
    "MFMSHO": "R004W",
    "MATRATCALC": 1.600000,
    "PRIORITY": 0,
    "POSITION": ""
  },
  {
    "MFMLIN": 20,
    "MFMDES": "R016SW33  ",
    "MFMSHO": "R016SW33",
    "MATRATCALC": 3.600000,
    "PRIORITY": 0,
    "POSITION": ""
  },
  {
    "MFMLIN": 30,
    "MFMDES": "FW  ",
    "MFMSHO": "FW",
    "MATRATCALC": 16.400000,
    "PRIORITY": 0,
    "POSITION": ""
  },
  {
    "MFMLIN": 40,
    "MFMDES": "SW VF029 ",
    "MFMSHO": "SW",
    "MATRATCALC": 3.400000,
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