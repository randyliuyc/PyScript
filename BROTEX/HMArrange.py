import random
import numpy as np
import pandas as pd

# ======================
# 3. 评估函数
# ======================
def evaluate(matrix):
    """计算方案的均匀性得分和重复惩罚"""
    row_scores = []
    col_scores = []
    
    # 参数均衡性
    for i in range(num_rows):
        row = matrix[i]
        row_mean = {k: np.mean([x[k] for x in row]) for k in ["P1","P2","P3","P4"]}
        diff = sum(abs(row_mean[k] - global_mean[k]) for k in row_mean)
        row_scores.append(diff)
    
    for j in range(num_cols):
        col = [matrix[i][j] for i in range(num_rows)]
        col_mean = {k: np.mean([x[k] for x in col]) for k in ["P1","P2","P3","P4"]}
        diff = sum(abs(col_mean[k] - global_mean[k]) for k in col_mean)
        col_scores.append(diff)
    
    balance_score = np.mean(row_scores + col_scores)
    
    # 连续重复惩罚
    penalty = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if j > 0 and matrix[i][j]["group"] == matrix[i][j-1]["group"]:
                penalty += 1
            if i > 0 and matrix[i][j]["group"] == matrix[i-1][j]["group"]:
                penalty += 1
    
    # 综合目标
    return balance_score + 0.5 * penalty

# ======================
# 4. 启发式初排
# ======================
def initial_layout():
    """生成一个初始方案（蛇形分布避免重复）"""
    layout = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    shuffled = bales.copy()
    random.shuffle(shuffled)
    
    idx = 0
    for i in range(num_rows):
        cols = range(num_cols) if i % 2 == 0 else reversed(range(num_cols))
        for j in cols:
            layout[i][j] = shuffled[idx]
            idx += 1
    return layout

# ======================
# 5. 局部搜索优化
# ======================
def local_search(layout, iterations=1000):
    best = layout
    best_score = evaluate(best)
    
    for _ in range(iterations):
        # 随机交换两个位置
        i1, j1 = random.randint(0, num_rows-1), random.randint(0, num_cols-1)
        i2, j2 = random.randint(0, num_rows-1), random.randint(0, num_cols-1)
        if (i1, j1) == (i2, j2):
            continue
        
        layout[i1][j1], layout[i2][j2] = layout[i2][j2], layout[i1][j1]
        score = evaluate(layout)
        
        if score < best_score:
            best_score = score
            best = [row.copy() for row in layout]
        else:
            # 恢复
            layout[i1][j1], layout[i2][j2] = layout[i2][j2], layout[i1][j1]
    
    return best, best_score

# ======================
# 6. 主程序
# ======================
# layout = initial_layout()
# optimized, score = local_search(layout, iterations=2000)

# # ======================
# # 7. 输出结果
# # ======================
# df = pd.DataFrame([[optimized[i][j]["group"] for j in range(num_cols)] for i in range(num_rows)],
#                   columns=[f"列{j+1}" for j in range(num_cols)])
# print("最终配摊方案：")
# print(df)
# print("\n均衡性指标得分：", round(score, 3))

# 输出每行、列参数均值
def param_stats(matrix):
    print("\n每行参数均值：")
    for i in range(num_rows):
        vals = matrix[i]
        mean = {k: np.mean([x[k] for x in vals]) for k in ["P1","P2","P3","P4"]}
        print(f"行{i+1}: {mean}")
    print("\n每列参数均值：")
    for j in range(num_cols):
        vals = [matrix[i][j] for i in range(num_rows)]
        mean = {k: np.mean([x[k] for x in vals]) for k in ["P1","P2","P3","P4"]}
        print(f"列{j+1}: {mean}")

# param_stats(optimized)

def linkrun(json_str):
    """
    处理传入的 JSON 字符串，并完成棉包排列算法
    :param json_str: JSON 格式的字符串，例如 '{"num_cols": 4, "other": "value"}'
    :return: 返回处理后的信息字符串或排列结果
    """
    import json
    try:
        data = json.loads(json_str)
        if "num_cols" in data:
            num_cols = data["num_cols"]
            if not isinstance(num_cols, int) or num_cols <= 0:
                return "参数错误: num_cols 必须是正整数"
            
            data = [
                ["A", 2, 33.2, 1, 2, 3],
                ["B", 1, 35.2, 1, 2, 3],
                ["C", 5, 33.2, 1, 2, 3],
                ["D", 18, 36.2, 1, 2, 3],
                ["E", 14, 37.2, 1, 2, 3],
                ["F", 2, 33.2, 1, 2, 3],
                ["G", 2, 35.2, 1, 2, 3],
                ["H", 3, 35.2, 1, 2, 3],
                ["I", 3, 35.2, 1, 2, 3],
                ["J", 2, 35.2, 1, 2, 3]
            ]

            # 配摊列数（用户输入）
            # num_cols = 2
            # 计算行数
            total_bales = sum([r[1] for r in data])
            # num_rows = total_bales // num_cols

            # ======================
            # 2. 生成棉包列表
            # ======================
            bales = []
            for group, qty, p1, p2, p3, p4 in data:
                for _ in range(qty):
                    bales.append({"group": group, "P1": p1, "P2": p2, "P3": p3, "P4": p4})

            random.shuffle(bales)

            # 全局参数均值
            global_mean = {
                "P1": np.mean([b["P1"] for b in bales]),
                "P2": np.mean([b["P2"] for b in bales]),
                "P3": np.mean([b["P3"] for b in bales]),
                "P4": np.mean([b["P4"] for b in bales])
            }
    
            # 调用棉包排列算法
            global num_rows
            num_rows = total_bales // num_cols
            layout = initial_layout()
            optimized, score = local_search(layout, iterations=2000)
            
            # 生成结果
            df = pd.DataFrame([[optimized[i][j]["group"] for j in range(num_cols)] for i in range(num_rows)],
                              columns=[f"列{j+1}" for j in range(num_cols)])
            
            # param_stats(optimized)

            result = {
                "配摊方案": df,
                "均衡性得分": round(score, 3)
            }
            return result
        else:
            return f"收到参数: {data}"
    except json.JSONDecodeError as e:
        return f"JSON 解析错误: {e}"
    
if __name__ == "__main__":
    print("Starting ...")

    json_str = '{"num_cols": 4, "B": 4, "Value": [1, 2, 3, 4]}'
    result = linkrun(json_str)
    print(result)    