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
# 参数设置
# ======================
# 从 JSON 数据中提取 MATRATCALC
json_data = [
    {"MFMLIN": 10, "MATRATCALC": 17, "PRIORITY": False, "POSITION": ""},
    {"MFMLIN": 20, "MATRATCALC": 18, "PRIORITY": False, "POSITION": ""},
    {"MFMLIN": 30, "MATRATCALC": 19.00, "PRIORITY": False, "POSITION": ""},
    {"MFMLIN": 40, "MATRATCALC": 20, "PRIORITY": False, "POSITION": ""},
    {"MFMLIN": 50, "MATRATCALC": 26, "PRIORITY": False, "POSITION": ""}
]
targets_pct = [item["MATRATCALC"] for item in json_data]
targets = [p / 100.0 for p in targets_pct]
tol = 0.015
top_n = 10

# 归一化 targets_pct
total_pct = sum(targets_pct)
if total_pct != 100:
    targets_pct = [p / total_pct * 100 for p in targets_pct]
print("Normalized targets_pct:", targets_pct)

# 定义三个阶段的搜索范围
search_stages = [
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

# 微调补偿参数
refine_delta = 0.03
refine_step = 0.01

max_X_checks = 200000
max_assign_per_X = 1000
max_solutions_collect = 2000

buckets = ['A','B','C','D','E','F','G','H']

# ======================
# 辅助函数
# ======================
def frange(start, stop, step):
    vals = []
    v = start
    while v <= stop + 1e-9:
        vals.append(round(v, 10))
        v += step
    return vals

def speeds_from_X(X1, X2, X3, X4):
    return [1/X1, 1/X4, 1, 1, 1/X2, 1/X2, 1/X4, 1/X3]

def compute_D(X1, X2, X3, X4):
    return sum(speeds_from_X(X1,X2,X3,X4))

def canonical_signature(assign, speeds):
    groups = {}
    for i, s in enumerate(speeds):
        key = round(s, 8)
        groups.setdefault(key, []).append(assign[i])
    return tuple(sorted((k, tuple(sorted(v))) for k,v in groups.items()))

def evaluate_assignment(assign, contribs, targets):
    K = len(targets)
    sums = [0] * K
    for i, c in enumerate(assign):
        sums[c] += contribs[i]
    devs = [abs(sums[i] - targets[i]) for i in range(K)]
    # 计算每个颜色的最终比例（百分比）
    total = sum(sums)
    final_pcts = [s / total * 100 for s in sums]
    return sums, devs, sum(devs), final_pcts

def backtracking_find(contribs, tol, targets, max_collect):
    n = len(contribs)
    K = len(targets)
    order = sorted(range(n), key=lambda i: -contribs[i])
    ordered = [contribs[i] for i in order]
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + ordered[i]
    assign = [None] * n
    curr = [0] * K
    lower = [t - tol for t in targets]
    upper = [t + tol for t in targets]
    res = []
    def dfs(pos):
        if len(res) >= max_collect:
            return True
        if pos == n:
            af = [None] * n
            for i, b in enumerate(order):
                af[b] = assign[i]
            s, d, c, final_pcts = evaluate_assignment(af, contribs, targets)
            if c <= tol:
                res.append((af, s, d, c, final_pcts))
            return False
        v = ordered[pos]
        rem = suffix[pos + 1]
        for k in range(K):
            if curr[k] + v > upper[k] + 1e-12:
                continue
            feas = True
            for j in range(K):
                if curr[j] + rem + (v if j == k else 0) < lower[j] - 1e-12:
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
# 主搜索函数（分阶段逐步扩大范围）
# ======================
def search_stage1():
    sols = []
    sigs = set()
    checked = 0
    start = time.time()

    for stage in search_stages:
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

            # 跳过不符合条件的组合
            if not (X4 > X1 and X4 > X3):
                continue
            if X4 / X1 >= 4 or X4 / X3 >= 4:
                continue
            D = compute_D(X1, X2, X3, X4)
            if not (0.5 < D < 10):
                continue

            sp = speeds_from_X(X1, X2, X3, X4)
            contrib = [s / D for s in sp]
            assigns = backtracking_find(contrib, tol, targets, max_assign_per_X)

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

                if len(stage_sols) >= max_solutions_collect:
                    break

            if len(stage_sols) >= max_solutions_collect or checked >= max_X_checks:
                break

        sols.extend(stage_sols)
        print(f"Found {len(stage_sols)} solutions in {stage['label']}, checked {stage_checked} combinations.")

        if len(sols) >= 200:
            print("Found enough solutions (>=200), stopping further stages.")
            break

    print(f"\nStage1 done: {len(sols)} solutions total, checked {checked}, time {time.time() - start:.2f}s")
    return sols

# ======================
# 局部微调阶段
# ======================
def refine_solutions(base_sols):
    refined = []
    deltas = [-refine_delta, -0.02, -0.01, 0, 0.01, 0.02, refine_delta]
    for sol in base_sols:
        best = sol.copy()
        best_dev = sol['dev']
        for d1, d2, d3, d4 in itertools.product(deltas, deltas, deltas, deltas):
            X1 = sol['X1'] + d1
            X2 = sol['X2'] + d2
            X3 = sol['X3'] + d3
            X4 = sol['X4'] + d4
            if not (1.0 <= X1 <= 4.0 and 1.0 <= X2 <= 4.0 and 1.0 <= X3 <= 4.0 and 1.1 <= X4 <= 6.0):
                continue
            if not (X4 > X1 and X4 > X3):
                continue
            if X4 / X1 >= 4 or X4 / X3 >= 4:
                continue
            D = compute_D(X1, X2, X3, X4)
            if not (0.5 < D < 10):
                continue
            sp = speeds_from_X(X1, X2, X3, X4)
            contrib = [s / D for s in sp]
            s, d, c, final_pcts = evaluate_assignment(sol['assign'], contrib, targets)
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
# 主函数
# ======================
def main():
    stage1 = search_stage1()
    refined = refine_solutions(stage1)
    refined_sorted = sorted(refined, key=lambda s: s['dev'])
    top = refined_sorted[:top_n]
    results = []
    for s in top:
        results.append({
            'X1': round(s['X1'], 4),
            'X2': round(s['X2'], 4),
            'X3': round(s['X3'], 4),
            'X4': round(s['X4'], 4),
            'cum_error': round(s['dev'], 6),
            'total_feed_speed_D': round(s['D'], 6),
            'speeds': [round(v, 6) for v in s['speeds']],
            'assign': {b: f"C{c+1}" for b, c in zip(buckets, s['assign'])},
            'final_pcts': [round(p, 2) for p in s['final_pcts']]
        })
    meta = {
        'targets_pct': targets_pct,
        'tol': tol,
        'refine_step': refine_step,
        'returned': len(results)
    }
    print(json.dumps({'meta': meta, 'results': results}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
