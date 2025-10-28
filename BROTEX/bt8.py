#!/usr/bin/env python3
"""
search_X4_refine.py
带局部补偿搜索的四牵伸倍数算法。
Stage 1: 粗搜索（分阶段逐步扩大范围）
Stage 2: 对找到的解在 ±0.03 范围内，以 0.01 步长精修，寻找更小误差
输出前10个误差最小方案（JSON格式）
"""

import itertools, time, math, json

# ======================
# 全局常量
# ======================
# 算法核心参数
TOL = 0.015                  # 允许的误差阈值
TOP_N = 10                   # 输出的最优解数量
MAX_SOLUTIONS_TO_STOP = 200  # 找到足够解时停止搜索的阈值
PRIORITY_ERROR_THRESHOLD = 0.0005  # 优先级颜色误差阈值
D_RANGE = (0.5, 10)          # 牵伸比例 D 的有效范围
X4_RATIO_LIMIT = 4           # X4/X1 和 X4/X3 的最大允许比值
MIN_X4 = 1.1                 # X4 的最小值
MAX_X4 = 6.0                 # X4 的最大值
EPSILON = 1e-12              # 浮点数比较的容差

# 微调补偿参数
REFINE_DELTA = 0.03
REFINE_STEP = 0.01
REFINE_DELTAS = [-REFINE_DELTA, -0.02, -0.01, 0, 0.01, 0.02, REFINE_DELTA]

# 搜索限制
MAX_X_CHECKS = 200000        # 最大检查组合数
MAX_ASSIGN_PER_X = 1000      # 每个 X 组合的最大分配尝试数
MAX_SOLUTIONS_COLLECT = 2000 # 最大收集解数量

# 桶位定义
BUCKETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# 搜索阶段定义
SEARCH_STAGES = [
    {
        "X1_range": (1.1, 2.0, 0.1),
        "X2_range": (1.1, 2.0, 0.1),
        "X3_range": (1.1, 2.0, 0.1),
        "X4_range": (1.1, 3.0, 0.1),
        "label": "Stage 1 (X1,X2,X3: 1.1-2.0, X4: 1.1-3.0)"
    },
    {
        "X1_range": (1.1, 3.0, 0.1),
        "X2_range": (1.1, 3.0, 0.1),
        "X3_range": (1.1, 3.0, 0.1),
        "X4_range": (1.1, 4.0, 0.1),
        "label": "Stage 2 (X1,X2,X3: 1.1-3.0, X4: 1.1-4.0)"
    },
    {
        "X1_range": (1.0, 4.0, 0.1),
        "X2_range": (1.0, 4.0, 0.1),
        "X3_range": (1.0, 4.0, 0.1),
        "X4_range": (1.0, 6.0, 0.1),
        "label": "Stage 3 (X1,X2,X3: 1.0-4.0, X4: 1.0-6.0)"
    }
]

# ======================
# 辅助函数
# ======================
def frange(start, stop, step):
    vals = []
    v = start
    while v <= stop + EPSILON:
        vals.append(round(v, 10))
        v += step
    return vals

def speeds_from_X(X1, X2, X3, X4):
    return [1/X1, 1/X4, 1, 1, 1/X2, 1/X2, 1/X4, 1/X3]

def compute_D(X1, X2, X3, X4):
    return sum(speeds_from_X(X1, X2, X3, X4))

def canonical_signature(assign, speeds):
    groups = {}
    for i, s in enumerate(speeds):
        key = round(s, 8)
        groups.setdefault(key, []).append(assign[i])
    return tuple(sorted((k, tuple(sorted(v))) for k, v in groups.items()))

def evaluate_assignment(assign, contribs, targets, json_data):
    K = len(targets)
    sums = [0] * K
    for i, c in enumerate(assign):
        sums[c] += contribs[i]
    devs = [abs(sums[i] - targets[i]) for i in range(K)]
    total = sum(sums)
    final_pcts = [s / total * 100 for s in sums]
    
    # 检查优先级误差
    for i, item in enumerate(json_data):
        if item["PRIORITY"] and devs[i] >= PRIORITY_ERROR_THRESHOLD:
            return sums, devs, float('inf'), final_pcts
    
    # 检查位置约束
    for i, item in enumerate(json_data):
        if item["POSITION"]:
            found = False
            for bucket_idx, color_idx in enumerate(assign):
                if color_idx == i and BUCKETS[bucket_idx] == item["POSITION"]:
                    found = True
                    break
            if not found:
                return sums, devs, float('inf'), final_pcts
    
    return sums, devs, sum(devs), final_pcts

def backtracking_find(contribs, targets, json_data):
    n = len(contribs)
    K = len(targets)
    priority_order = sorted(range(K), key=lambda i: (
        -json_data[i]["PRIORITY"],
        json_data[i]["POSITION"] != ""
    ))
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + contribs[i]
    
    assign = [None] * n
    curr = [0] * K
    lower = [t - TOL for t in targets]
    upper = [t + TOL for t in targets]
    res = []
    
    def dfs(pos):
        if len(res) >= MAX_ASSIGN_PER_X:
            return True
        if pos == n:
            s, d, c, final_pcts = evaluate_assignment(assign, contribs, targets, json_data)
            if c <= TOL:
                res.append((assign.copy(), s, d, c, final_pcts))
            return False
        
        v = contribs[pos]
        rem = suffix[pos + 1]
        
        for k in range(K):
            if curr[k] + v > upper[k] + EPSILON:
                continue
            
            feas = True
            for j in range(K):
                if curr[j] + rem + (v if j == k else 0) < lower[j] - EPSILON:
                    feas = False
                    break
            
            if not feas:
                continue
            
            assign[pos] = k
            curr[k] += v
            stop = dfs(pos + 1)
            curr[k] -= v
            assign[pos] = None
            
            if stop:
                return True
        return False
    
    dfs(0)
    return res

# ======================
# 主搜索函数
# ======================
def search_stage1(json_data, targets):
    sols = []
    sigs = set()
    checked = 0
    start = time.time()

    for stage in SEARCH_STAGES:
        X1_vals = frange(*stage["X1_range"])
        X2_vals = frange(*stage["X2_range"])
        X3_vals = frange(*stage["X3_range"])
        X4_vals = frange(*stage["X4_range"])

        print(f"\nStarting {stage['label']}...")
        stage_checked = 0
        stage_sols = []

        for X1, X2, X3, X4 in itertools.product(X1_vals, X2_vals, X3_vals, X4_vals):
            checked += 1
            stage_checked += 1

            if not (X4 > X1 and X4 > X3):
                continue
            if X4 / X1 >= X4_RATIO_LIMIT or X4 / X3 >= X4_RATIO_LIMIT:
                continue
            D = compute_D(X1, X2, X3, X4)
            if not (D_RANGE[0] < D < D_RANGE[1]):
                continue

            sp = speeds_from_X(X1, X2, X3, X4)
            contrib = [s / D for s in sp]
            assigns = backtracking_find(contrib, targets, json_data)

            if not assigns:
                continue

            for af, s, d, c, final_pcts in assigns:
                sig = (
                    round(X1, 3),
                    round(X2, 3),
                    round(X3, 3),
                    round(X4, 3),
                    canonical_signature(af, sp)
                )
                if sig in sigs:
                    continue
                sigs.add(sig)
                stage_sols.append({
                    'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4,
                    'assign': af, 'dev': c, 'speeds': sp, 'D': D,
                    'final_pcts': final_pcts
                })

                if len(stage_sols) >= MAX_SOLUTIONS_COLLECT:
                    break

            if len(stage_sols) >= MAX_SOLUTIONS_COLLECT or checked >= MAX_X_CHECKS:
                break

        sols.extend(stage_sols)
        print(f"Found {len(stage_sols)} solutions in {stage['label']}, checked {stage_checked} combinations.")

        if len(sols) >= MAX_SOLUTIONS_TO_STOP:
            print(f"Found enough solutions (>= {MAX_SOLUTIONS_TO_STOP}), stopping further stages.")
            break

    print(f"\nStage1 done: {len(sols)} solutions total, checked {checked}, time {time.time() - start:.2f}s")
    return sols

# ======================
# 微调阶段
# ======================
def refine_solutions(base_sols, json_data, targets):
    refined = []
    for sol in base_sols:
        best = sol.copy()
        best_dev = sol['dev']
        for d1, d2, d3, d4 in itertools.product(REFINE_DELTAS, REFINE_DELTAS, REFINE_DELTAS, REFINE_DELTAS):
            X1 = sol['X1'] + d1
            X2 = sol['X2'] + d2
            X3 = sol['X3'] + d3
            X4 = sol['X4'] + d4
            if not (1.0 <= X1 <= 4.0 and 1.0 <= X2 <= 4.0 and 1.0 <= X3 <= 4.0 and MIN_X4 <= X4 <= MAX_X4):
                continue
            if not (X4 > X1 and X4 > X3):
                continue
            if X4 / X1 >= X4_RATIO_LIMIT or X4 / X3 >= X4_RATIO_LIMIT:
                continue
            D = compute_D(X1, X2, X3, X4)
            if not (D_RANGE[0] < D < D_RANGE[1]):
                continue
            sp = speeds_from_X(X1, X2, X3, X4)
            contrib = [s / D for s in sp]
            s, d, c, final_pcts = evaluate_assignment(sol['assign'], contrib, targets, json_data)
            if c < best_dev:
                best = {
                    'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4,
                    'assign': sol['assign'], 'dev': c, 'speeds': sp, 'D': D,
                    'final_pcts': final_pcts
                }
                best_dev = c
        refined.append(best)
    return refined

# ======================
# 主执行函数
# ======================
def linkrun(json_data):
    def normalize_targets(targets):
        total = sum(targets)
        if total == 100:
            return targets
        else:
            factor = 100.0 / total
            return [round(p * factor, 2) for p in targets]

    targets_pct = [item["MATRATCALC"] for item in json_data]
    targets_pct = normalize_targets(targets_pct)
    print("Normalized targets_pct:", targets_pct)

    targets = [p / 100.0 for p in targets_pct]

    stage1 = search_stage1(json_data, targets)
    refined = refine_solutions(stage1, json_data, targets)
    refined_sorted = sorted(refined, key=lambda s: s['dev'])
    top = refined_sorted[:TOP_N]
    results = []
    
    for s in top:
        assign_list = []
        for bucket_idx, color_idx in enumerate(s['assign']):
            if bucket_idx == 0:
                x_value = s['X1']
            elif bucket_idx in [1, 6]:
                x_value = s['X4']
            elif bucket_idx in [4, 5]:
                x_value = s['X2']
            elif bucket_idx == 7:
                x_value = s['X3']
            else:
                x_value = 1.0
                
            assign_list.append({
                'bucket': BUCKETS[bucket_idx],
                'color': json_data[color_idx]['MFMLIN'],
                'x': round(x_value, 2),
                'speed': round(s['speeds'][bucket_idx], 6)
            })
        
        colors_list = []
        for color_idx in range(len(json_data)):
            target_pct = targets_pct[color_idx]
            final_pct = s['final_pcts'][color_idx]
            error = abs(final_pct - target_pct)
            colors_list.append({
                'color': json_data[color_idx]['MFMLIN'],
                'target': round(target_pct, 2),
                'final': round(final_pct, 2),
                'error': round(error, 2)
            })
        
        results.append({
            'X1': round(s['X1'], 4),
            'X2': round(s['X2'], 4),
            'X3': round(s['X3'], 4),
            'X4': round(s['X4'], 4),
            'cum_error': round(s['dev'], 6),
            'total_feed_speed_D': round(s['D'], 6),
            'assign': assign_list,
            'colors': colors_list
        })
    
    meta = {
        'targets_pct': targets_pct,
        'tol': TOL,
        'refine_step': REFINE_STEP,
        'returned': len(results)
    }
    print(json.dumps({'meta': meta, 'results': results}, indent=2, ensure_ascii=False))

# 程序启动
if __name__ == "__main__":
    json_data = [
        {"MFMLIN": 10, "MATRATCALC": 1.5, "PRIORITY": False, "POSITION": "B"},
        {"MFMLIN": 20, "MATRATCALC": 6.43, "PRIORITY": False, "POSITION": ""},
        {"MFMLIN": 30, "MATRATCALC": 5, "PRIORITY": False, "POSITION": ""},
        {"MFMLIN": 40, "MATRATCALC": 9.32, "PRIORITY": False, "POSITION": ""},
        {"MFMLIN": 50, "MATRATCALC": 4, "PRIORITY": False, "POSITION": ""}
    ]
    linkrun(json_data)
