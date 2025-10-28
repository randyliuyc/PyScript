#!/usr/bin/env python3
"""
search_X4_json.py

四牵伸倍数模型的可运行脚本（在终端输出 JSON 结果）
- 输入：targets_pct（百分比，和为100，2..8个颜色）
- 输出：最多 top_n 个满足 cum_error < tol 的方案，以 JSON 打印到 stdout
- 运行：python3 search_X4_json.py
"""

import itertools
import time
import math
import json
from collections import OrderedDict

# --------------------------
# User-editable parameters
# --------------------------
targets_pct = [17, 18, 19, 20, 26]   # <-- 修改为你的配比（2..8 entries）
assert 2 <= len(targets_pct) <= 8, "Number of colors must be between 2 and 8."
assert abs(sum(targets_pct) - 100.0) < 1e-9, "Targets must sum to 100."

targets = [p / 100.0 for p in targets_pct]

# cumulative error tolerance (fraction), e.g. 1.5% -> 0.015
tol = 0.015

# output count
top_n = 10

# priority search grid (fast, recommended). Use step 0.1 or 0.05
X1_min, X1_max, X1_step = 1.1, 2.0, 0.1
X2_min, X2_max, X2_step = 1.1, 2.0, 0.1
X3_min, X3_max, X3_step = 1.1, 2.0, 0.1

# X4 full range
X4_min, X4_max, X4_step = 1.1, 6.0, 0.1

# runtime caps (safety)
max_X_checks = 200000    # stop checking after this many X combinations (to avoid endless runs)
max_assign_per_X = 2000  # cap assignments collected per X
max_solutions_collect = 2000

# buckets order
buckets = ['A','B','C','D','E','F','G','H']

# --------------------------
# Helper functions
# --------------------------
def frange(start, stop, step):
    vals = []
    v = start
    while v <= stop + 1e-9:
        vals.append(round(v, 10))
        v += step
    return vals

def speeds_from_X(X1, X2, X3, X4):
    # mapping as specified:
    # C,D = 1
    # A = 1/X1
    # E,F = 1/X2
    # H = 1/X3
    # B,G = 1/X4
    return [1.0/X1, 1.0/X4, 1.0, 1.0, 1.0/X2, 1.0/X2, 1.0/X4, 1.0/X3]

def compute_D(X1, X2, X3, X4):
    return sum(speeds_from_X(X1, X2, X3, X4))

def canonical_signature(assign, speeds):
    """
    Deduplicate by grouping buckets with identical speeds (rounded) and capturing
    the multiset of colors assigned to each equal-speed group.
    """
    groups = {}
    for idx, sp in enumerate(speeds):
        key = round(sp, 9)
        groups.setdefault(key, []).append(assign[idx])
    group_signs = []
    for k in sorted(groups.keys()):
        colors = tuple(sorted(groups[k]))
        group_signs.append((round(k,6), colors))
    return tuple(group_signs)

def evaluate_assignment(assign, contribs, targets):
    K = len(targets)
    sums = [0.0]*K
    for i, c in enumerate(assign):
        sums[c] += contribs[i]
    devs = [abs(sums[i] - targets[i]) for i in range(K)]
    cum = sum(devs)
    return sums, devs, cum

def backtracking_find_assignments(contribs, tol, targets, max_collect):
    """
    Backtracking to find assignments where cumulative deviation <= tol.
    Prunes by ordering buckets descending by contribution.
    Returns list of tuples: (assign_full, sums, devs, cum)
    """
    n = len(contribs)  # 8
    K = len(targets)
    order = sorted(range(n), key=lambda i: -contribs[i])
    ordered = [contribs[i] for i in order]
    suffix = [0.0]*(n+1)
    for i in range(n-1, -1, -1):
        suffix[i] = suffix[i+1] + ordered[i]

    assignment = [None]*n
    current = [0.0]*K
    lower = [targets[i] - tol for i in range(K)]
    upper = [targets[i] + tol for i in range(K)]
    results = []

    def backtrack(pos):
        if len(results) >= max_collect:
            return True
        if pos == n:
            assign_full = [None]*n
            for idx, bucket_idx in enumerate(order):
                assign_full[bucket_idx] = assignment[idx]
            sums, devs, cum = evaluate_assignment(assign_full, contribs, targets)
            if cum <= tol + 1e-12:
                results.append((assign_full, sums, devs, cum))
            return False
        val = ordered[pos]
        rem = suffix[pos+1]
        for color in range(K):
            # upper-bound prune
            if current[color] + val > upper[color] + 1e-12:
                continue
            # lower-bound feasibility
            feasible = True
            for k in range(K):
                poss_max = current[k] + rem + (val if k == color else 0.0)
                if poss_max < lower[k] - 1e-12:
                    feasible = False
                    break
            if not feasible:
                continue
            assignment[pos] = color
            current[color] += val
            stop = backtrack(pos+1)
            current[color] -= val
            assignment[pos] = None
            if stop:
                return True
        return False

    backtrack(0)
    return results

# --------------------------
# Main search
# --------------------------
def main():
    X1_vals = frange(X1_min, X1_max, X1_step)
    X2_vals = frange(X2_min, X2_max, X2_step)
    X3_vals = frange(X3_min, X3_max, X3_step)
    X4_vals = frange(X4_min, X4_max, X4_step)

    solutions = []   # collected solution dicts
    signatures = set()
    checked = 0
    start = time.time()

    # iterate over grid
    for X1 in X1_vals:
        for X2 in X2_vals:
            for X3 in X3_vals:
                for X4 in X4_vals:
                    checked += 1
                    # constraints: X4 > X1 and X4 > X3
                    if not (X4 > X1 and X4 > X3):
                        continue
                    # ratio constraints
                    if (X4 / X1) >= 4.0 or (X4 / X3) >= 4.0:
                        continue
                    D = compute_D(X1, X2, X3, X4)
                    if not (0.5 < D < 10.0):
                        continue
                    # contribution fractions
                    speeds = speeds_from_X(X1, X2, X3, X4)
                    contribs = [s / D for s in speeds]
                    # find assignments
                    assigns = backtracking_find_assignments(contribs, tol, targets, max_assign_per_X)
                    if not assigns:
                        continue
                    for assign_full, sums, devs, cum in assigns:
                        sig = (round(X1,4), round(X2,4), round(X3,4), round(X4,4),
                               canonical_signature(assign_full, speeds))
                        if sig in signatures:
                            continue
                        signatures.add(sig)
                        solutions.append({
                            'X1': round(X1,4), 'X2': round(X2,4), 'X3': round(X3,4), 'X4': round(X4,4),
                            'assign': assign_full,
                            'per_color_actual': [round(v, 8) for v in sums],
                            'per_color_dev': [round(d, 8) for d in devs],
                            'cum_error': round(cum, 10),
                            'speeds': [round(s, 8) for s in speeds],
                            'total_feed_speed_D': round(D, 8)
                        })
                        if len(solutions) >= max_solutions_collect:
                            break
                    if len(solutions) >= max_solutions_collect:
                        break
                    if checked >= max_X_checks:
                        break
                if len(solutions) >= max_solutions_collect or checked >= max_X_checks:
                    break
            if len(solutions) >= max_solutions_collect or checked >= max_X_checks:
                break
        if len(solutions) >= max_solutions_collect or checked >= max_X_checks:
            break

    elapsed = time.time() - start
    # sort by cum_error ascending and then by total_feed_speed
    solutions_sorted = sorted(solutions, key=lambda s: (s['cum_error'], s['total_feed_speed_D']))
    top = solutions_sorted[:top_n]

    # transform assignments from indices to labels for readability
    output = []
    for sol in top:
        assign_labels = ['C{}'.format(i+1) for i in sol['assign']]
        rec = {
            'X1': sol['X1'], 'X2': sol['X2'], 'X3': sol['X3'], 'X4': sol['X4'],
            'total_feed_speed_D': sol['total_feed_speed_D'],
            'cum_error': sol['cum_error'],
            'assign_buckets': dict(zip(buckets, assign_labels)),
            'per_color_actual': sol['per_color_actual'],
            'per_color_dev': sol['per_color_dev'],
            'speeds': sol['speeds']
        }
        output.append(rec)

    meta = {
        'targets_pct': targets_pct,
        'tol': tol,
        'top_n_requested': top_n,
        'collected_solutions': len(solutions),
        'returned_solutions': len(output),
        'checked_X_combinations': checked,
        'elapsed_seconds': round(elapsed, 3)
    }

    # print JSON to stdout (pretty)
    print(json.dumps({'meta': meta, 'results': output}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
