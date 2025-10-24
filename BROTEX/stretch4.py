# -*- coding: utf-8 -*-
# Solver for assigning colors to 8 buckets with variable speeds (X1,X2,X3).
# Requires: pip install pulp
# 2025年10月21日，测试代码可用

import itertools
import math
from collections import defaultdict
import pulp

# --- USER INPUT ---
# Example: 4 colors with targets (sum to 1.0)
targets = {1:0.17, 2:0.18, 3:0.19, 4:0.20, 5:0.13, 6:0.13}   # replace with your actual targets
epsilon = 0.01   # allowed absolute error in fraction (e.g. 0.01 = 1%)
K = len(targets)

# X grid (example: priority around 1.3..2.0)
def make_X_values():
    vals = []
    # priority fine grid in [1.3, 2.0]
    v = 1.3
    while v <= 2.0001:
        vals.append(round(v,3)); v += 0.05
    # extend outside range coarse steps
    v = 1.0
    while v < 1.3 - 1e-9:
        vals.append(round(v,3)); v += 0.1
    v = 2.2
    while v <= 3.2 + 1e-9:
        vals.append(round(v,3)); v += 0.1
    vals = sorted(set(vals))
    return vals

X1_values = make_X_values()
X2_values = make_X_values()
X3_values = make_X_values()

# Bucket order: A,B,C,D,E,F,G,H
def speeds_from_X(X1,X2,X3, N=1.0):
    s_C = 1.0
    s_D = 1.0
    s_B = 1.0 / X1
    s_A = s_B / 1.3
    s_E = 1.0 / X2
    s_F = 1.0 / X2
    s_G = 1.0 / X3
    s_H = s_G / 1.3
    return {'A':s_A,'B':s_B,'C':s_C,'D':s_D,'E':s_E,'F':s_F,'G':s_G,'H':s_H}

buckets = ['A','B','C','D','E','F','G','H']
S_tol = epsilon

# MILP solver function
def try_solve_milp(speeds, targets, epsilon):
    prob = pulp.LpProblem('mix', pulp.LpStatusOptimal)
    K = len(targets)
    colors = list(targets.keys())
    # variables
    y = pulp.LpVariable.dicts('y', ((b,c) for b in buckets for c in colors), 0,1, cat='Binary')
    d = pulp.LpVariable.dicts('d', (c for c in colors), lowBound=0, cat='Continuous')
    S_total = sum(speeds[b] for b in buckets)
    # each bucket assigned one color
    for b in buckets:
        prob += pulp.lpSum(y[(b,c)] for c in colors) == 1
    # linearized deviations
    for c in colors:
        a_c = pulp.lpSum(speeds[b]*y[(b,c)] for b in buckets) / S_total
        prob += a_c - targets[c] <= d[c]
        prob += targets[c] - a_c <= d[c]
        prob += d[c] <= epsilon
    # objective (feasibility): minimize sum of deviations (or max)
    prob += pulp.lpSum(d[c] for c in colors)
    # solve
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
    res = prob.solve(solver)
    if pulp.LpStatus[res] == 'Optimal' or pulp.LpStatus[res] == 'Feasible':
        # build assignment and actuals
        assign = {}
        for b in buckets:
            for c in colors:
                if pulp.value(y[(b,c)]) > 0.5:
                    assign[b] = c
        actuals = {}
        for c in colors:
            actuals[c] = sum(speeds[b] for b,v in assign.items() if v==c) / S_total
        return True, assign, actuals
    else:
        return False, None, None

# MAIN search (coarse)
solutions = []
limit_checks = 2000  # optional limit for runtime control
count = 0
for X1, X2, X3 in itertools.product(X1_values, X2_values, X3_values):
    count += 1
    if count > limit_checks:
        break
    speeds = speeds_from_X(X1,X2,X3)
    feasible, assign, actuals = try_solve_milp(speeds, targets, S_tol)
    if feasible:
        # 计算绝对差值
        absolute_diff = {
            color: round(abs(actuals[color] - targets[color]) *100, 2) 
            for color in targets
            }
        
        # 计算累计的绝对差值
        total_abs_diff = round(sum(abs(actuals[color] - targets[color]) for color in targets) * 100, 2)
        
        # 计算相对差值（百分比）
        relative_diff = {
            color: round((abs(actuals[color] - targets[color]) / targets[color]) * 100, 2)
            for color in targets
            }

        # 将 actuals 转换为百分比格式
        actuals_percent = {
            color: round(value * 100, 2)
            for color, value in actuals.items()
            }

        solutions.append({'X1':X1,'X2':X2,'X3':X3,'assign':assign,'actuals':actuals_percent,'diff':absolute_diff,'rela_diff':relative_diff,'total_diff':total_abs_diff})
        # optional early stop: break
        # break

print("Found", len(solutions), "solutions (coarse search).")
# 按累计绝对误差排序并输出前10个最优方案
sorted_solutions = sorted(solutions, key=lambda x: x['total_diff'])
print("\nTop 10 solutions with smallest total absolute difference (%):")
for s in sorted_solutions[:10]:
    print(f"X1={s['X1']}, X2={s['X2']}, X3={s['X3']}, TotalDiff={s['total_diff']}%")
    print(f"Assignments: {s['assign']}")
    print(f"Actuals(%): {s['actuals']}")
    print(f"Diffs(%): {s['diff']}\n")
