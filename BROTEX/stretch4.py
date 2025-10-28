import itertools
import math
from collections import defaultdict
import pulp

# --- USER INPUT ---
targets = {1: 0.17, 2: 0.18, 3: 0.19, 4: 0.20, 5: 0.13, 6: 0.13}  # 替换为实际目标
epsilon = 0.01  # 允许的绝对误差
K = len(targets)

# 生成 X 的取值范围
def make_X_values(start, end, step=0.1):
    vals = []
    v = start
    while v <= end + 1e-9:  # 避免浮点精度问题
        vals.append(round(v, 3))
        v += step
    return sorted(set(vals))  # 去重并排序

# X1, X2, X3 的取值范围：1.0 - 4.0，步长 0.1
X1_values = make_X_values(1.0, 4.0, 0.01)
X2_values = make_X_values(1.0, 4.0, 0.01)
X3_values = make_X_values(1.0, 4.0, 0.01)

# X4 的取值范围：1.0 - 6.0，步长 0.1
X4_values = make_X_values(1.0, 6.0, 0.01)

# 桶顺序: A, B, C, D, E, F, G, H
def speeds_from_X(X1, X2, X3, X4, N=1.0):
    s_C = 1.0
    s_D = 1.0
    s_B = 1.0 / X4
    s_A = 1.0 / X1
    s_E = 1.0 / X2
    s_F = 1.0 / X2
    s_G = 1.0 / X4
    s_H = 1.0 / X3
    return {'A': s_A, 'B': s_B, 'C': s_C, 'D': s_D, 'E': s_E, 'F': s_F, 'G': s_G, 'H': s_H}

buckets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
S_tol = epsilon

# MILP 求解器函数
def try_solve_milp(speeds, targets, epsilon):
    prob = pulp.LpProblem('mix', pulp.LpStatusOptimal)
    K = len(targets)
    colors = list(targets.keys())
    # 变量
    y = pulp.LpVariable.dicts('y', ((b, c) for b in buckets for c in colors), 0, 1, cat='Binary')
    d = pulp.LpVariable.dicts('d', (c for c in colors), lowBound=0, cat='Continuous')
    S_total = sum(speeds[b] for b in buckets)
    # 每个桶分配一种颜色
    for b in buckets:
        prob += pulp.lpSum(y[(b, c)] for c in colors) == 1
    # 线性化偏差
    for c in colors:
        a_c = pulp.lpSum(speeds[b] * y[(b, c)] for b in buckets) / S_total
        prob += a_c - targets[c] <= d[c]
        prob += targets[c] - a_c <= d[c]
        prob += d[c] <= epsilon
    # 目标函数：最小化总偏差
    prob += pulp.lpSum(d[c] for c in colors)
    # 求解
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
    res = prob.solve(solver)
    if pulp.LpStatus[res] == 'Optimal' or pulp.LpStatus[res] == 'Feasible':
        # 构建分配和实际值
        assign = {}
        for b in buckets:
            for c in colors:
                if pulp.value(y[(b, c)]) > 0.5:
                    assign[b] = c
        actuals = {}
        for c in colors:
            actuals[c] = sum(speeds[b] for b, v in assign.items() if v == c) / S_total
        return True, assign, actuals
    else:
        return False, None, None

# 主搜索逻辑
solutions = []
limit_checks = 2000  # 可选限制，控制运行时间
count = 0

for X1, X2, X3, X4 in itertools.product(X1_values, X2_values, X3_values, X4_values):
    count += 1
    if count > limit_checks:
        break
    speeds = speeds_from_X(X1, X2, X3, X4)
    feasible, assign, actuals = try_solve_milp(speeds, targets, S_tol)
    if feasible:
        # 计算绝对差值
        absolute_diff = {
            color: round(abs(actuals[color] - targets[color]) * 100, 2)
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

        solutions.append({
            'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4,
            'assign': assign,
            'actuals': actuals_percent,
            'diff': absolute_diff,
            'rela_diff': relative_diff,
            'total_diff': total_abs_diff
        })

# 输出结果
print("Found", len(solutions), "solutions (coarse search).")
sorted_solutions = sorted(solutions, key=lambda x: x['total_diff'])
print("\nTop 10 solutions with smallest total absolute difference (%):")
for s in sorted_solutions[:10]:
    print(f"X1={s['X1']}, X2={s['X2']}, X3={s['X3']}, X4={s['X4']}, TotalDiff={s['total_diff']}%")
    print(f"Assignments: {s['assign']}")
    print(f"Actuals(%): {s['actuals']}")
    print(f"Diffs(%): {s['diff']}\n")
