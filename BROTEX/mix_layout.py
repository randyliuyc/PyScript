import random
import numpy as np
import pandas as pd

def mix_cotton_layout(data, num_cols=4, iterations=2000, verbose=False):
    """
    混棉配摊算法（启发式 + 局部优化）
    
    参数：
        data: list
            每行格式为 [group, qty, P1, P2, P3, P4]
        num_cols: int
            配摊列数（2~6）
        iterations: int
            局部优化迭代次数
        verbose: bool
            是否打印统计结果
    
    返回：
        df: pd.DataFrame
            最终配摊矩阵
        score: float
            均衡性得分（越低越好）
    """

    # ======================
    # 1. 生成棉包列表
    # ======================
    total_bales = sum([r[1] for r in data])
    num_rows = (total_bales + num_cols - 1) // num_cols  # 向上取整
    
    bales = []
    for group, qty, p1, p2, p3, p4 in data:
        for _ in range(qty):
            bales.append({"group": group, "P1": p1, "P2": p2, "P3": p3, "P4": p4})

    random.shuffle(bales)

    # 全局均值
    global_mean = {
        "P1": np.mean([b["P1"] for b in bales]),
        "P2": np.mean([b["P2"] for b in bales]),
        "P3": np.mean([b["P3"] for b in bales]),
        "P4": np.mean([b["P4"] for b in bales])
    }

    # ======================
    # 2. 评估函数
    # ======================
    def evaluate(matrix):
        """计算方案的均匀性得分、重复惩罚和全局空间分布均匀性"""
        row_scores = []
        col_scores = []
        
        # 1. 行均衡
        for i in range(len(matrix)):
            row = [x for x in matrix[i] if x is not None]
            row_mean = {k: np.mean([x[k] for x in row]) for k in ["P1","P2","P3","P4"]}
            diff = sum(abs(row_mean[k] - global_mean[k]) for k in row_mean)
            row_scores.append(diff)
        
        # 2. 列均衡
        for j in range(num_cols):
            col = [matrix[i][j] for i in range(len(matrix)) if j < len(matrix[i]) and matrix[i][j] is not None]
            if not col:
                continue
            col_mean = {k: np.mean([x[k] for x in col]) for k in ["P1","P2","P3","P4"]}
            diff = sum(abs(col_mean[k] - global_mean[k]) for k in col_mean)
            col_scores.append(diff)
        
        balance_score = np.mean(row_scores + col_scores)
        
        # 3. 相邻重复惩罚
        penalty = 0
        for i in range(len(matrix)):
            for j in range(num_cols):
                if matrix[i][j] is None:
                    continue
                if j > 0 and matrix[i][j-1] is not None and matrix[i][j]["group"] == matrix[i][j-1]["group"]:
                    penalty += 1
                if i > 0 and j < len(matrix[i-1]) and matrix[i-1][j] is not None and matrix[i][j]["group"] == matrix[i-1][j]["group"]:
                    penalty += 1
        
        # 4. 全局空间分布均匀性（新增）
        regions = {
            "upper": lambda i, j: i < num_rows // 3,
            "lower": lambda i, j: i >= 2 * num_rows // 3,
            "left": lambda i, j: j < num_cols // 3,
            "right": lambda i, j: j >= 2 * num_cols // 3,
            "center": lambda i, j: (num_rows // 3 <= i < 2 * num_rows // 3) and (num_cols // 3 <= j < 2 * num_cols // 3)
        }
        
        group_region_counts = {}
        for i in range(len(matrix)):
            for j in range(num_cols):
                if matrix[i][j] is not None:
                    group = matrix[i][j]["group"]
                    if group not in group_region_counts:
                        group_region_counts[group] = {region: 0 for region in regions}
                    for region, condition in regions.items():
                        if condition(i, j):
                            group_region_counts[group][region] += 1
        
        distribution_score = 0
        for group, counts in group_region_counts.items():
            total = sum(counts.values())
            if total == 0:
                continue
            proportions = [count / total for count in counts.values()]
            distribution_score += np.std(proportions)
        
        # 综合得分
        total_score = balance_score + 0.5 * penalty + 0.5 * distribution_score
        return total_score
    
    # ======================
    # 3. 初始布局（蛇形）
    # ======================
    def initial_layout():
        layout = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        idx = 0
        for i in range(num_rows):
            cols = range(num_cols) if i % 2 == 0 else reversed(range(num_cols))
            for j in cols:
                if idx < len(bales):
                    layout[i][j] = bales[idx]
                    idx += 1
        return layout

    # ======================
    # 4. 局部搜索
    # ======================
    def local_search(layout):
        best = layout
        best_score = evaluate(best)
        for _ in range(iterations):
            i1, j1 = random.randint(0, num_rows-1), random.randint(0, num_cols-1)
            i2, j2 = random.randint(0, num_rows-1), random.randint(0, num_cols-1)
            
            # 跳过空位或相同
            if i1 == i2 and j1 == j2:
                continue
            if layout[i1][j1] is None or layout[i2][j2] is None:
                continue

            # 交换
            layout[i1][j1], layout[i2][j2] = layout[i2][j2], layout[i1][j1]
            score = evaluate(layout)
            
            if score < best_score:
                best_score = score
                best = [row.copy() for row in layout]
            else:
                layout[i1][j1], layout[i2][j2] = layout[i2][j2], layout[i1][j1]
        
        return best, best_score

    # ======================
    # 5. 执行优化
    # ======================
    layout = initial_layout()
    optimized, score = local_search(layout)

    # ======================
    # 6. 输出结果
    # ======================
    df = pd.DataFrame([[optimized[i][j]["group"] if optimized[i][j] else None for j in range(num_cols)] 
                        for i in range(num_rows)],
                      columns=[f"C{j+1}" for j in range(num_cols)])
    
    # 统计每个 group 被分配的次数
    group_counts = {}
    for i in range(num_rows):
        for j in range(num_cols):
            if optimized[i][j] is not None:
                group = optimized[i][j]["group"]
                group_counts[group] = group_counts.get(group, 0) + 1

    # 按 group 排序
    sorted_group_counts = dict(sorted(group_counts.items()))

    if verbose:
        print("最终配摊方案：")
        print(df)
        print("\n均衡性指标得分：", round(score, 3))

        # 输出行列均值
        print("\n每行参数均值：")
        for i in range(num_rows):
            vals = [x for x in optimized[i] if x is not None]
            mean = {k: np.mean([x[k] for x in vals]) for k in ["P1","P2","P3","P4"]}
            print(f"行{i+1}: {mean}")
        
        print("\n每列参数均值：")
        for j in range(num_cols):
            vals = [optimized[i][j] for i in range(num_rows) if j < len(optimized[i]) and optimized[i][j] is not None]
            if vals:
                mean = {k: np.mean([x[k] for x in vals]) for k in ["P1","P2","P3","P4"]}
                print(f"列{j+1}: {mean}")

        # 输出排序后的 group 分配次数
        print("\n每个 group 被分配的次数（按 group 排序）：")
        for group, count in sorted_group_counts.items():
            print(f"{group}: {count}")

    return df, score, sorted_group_counts
def linkrun(args):
    import json
    try:
        linkargs = json.loads(args)
        if "num_cols" not in linkargs:
            return json.dumps({"isSuccess": 0, "message": "参数错误: 缺少 num_cols"})
        
        num_cols = linkargs["num_cols"]
        if not isinstance(num_cols, int) or num_cols <= 0:
            return json.dumps({"isSuccess": 0, "message": "参数必须是正数."})

        # 从 args 中提取并转换 data
        if "data" not in linkargs:
            return json.dumps({"isSuccess": 0, "message": "参数错误: 缺少 data"})
        
        input_data = linkargs["data"]
        if not isinstance(input_data, list) or not all(isinstance(item, dict) for item in input_data):
            return json.dumps({"isSuccess": 0, "message": "data 格式错误: 必须为对象数组"})
        
        # 转换为 [group, qty, P1, P2, P3, P4] 格式
        data = []
        for item in input_data:
            if not all(key in item for key in ["OPECOD", "MATRAT", "P1", "P2", "P3", "P4"]):
                return json.dumps({"isSuccess": 0, "message": "data 字段缺失: 必须包含 OPECOD, MATRAT, P1-P4"})
            try:
                data.append([
                    str(item["OPECOD"]),  # group
                    int(item["MATRAT"]),   # qty
                    float(item["P1"]),     # P1
                    float(item["P2"]),     # P2
                    float(item["P3"]),     # P3
                    float(item["P4"])      # P4
                ])
            except (ValueError, TypeError) as e:
                return json.dumps({"isSuccess": 0, "message": f"数据类型错误: {e}"})

        # 调用混棉配摊算法
        df, score, sorted_group_counts = mix_cotton_layout(data, num_cols)

        result = {
            "isSuccess": 1,
            "data": df.to_dict(orient="records"),
            "score": float(round(score, 3)),
            "group_counts": sorted_group_counts
        }
        return json.dumps(result)

    except json.JSONDecodeError as e:
        return json.dumps({"isSuccess": 0, "message": f"JSON 解析错误: {e}"})

# ======================
# ✅ 示例调用
# ======================
if __name__ == "__main__":
    print("Starting ...")

    json_str = """{
  "num_cols": 4,
  "data": [
    {
      "OPECOD": "A",
      "CPNITMNO": "JKBM",
      "CPNITMVER": "巴西BCI棉",
      "CPNLOT": "B2411072",
      "MATRAT": 2,
      "P1": 0.000000,
      "P2": 0.000000,
      "P3": 0.000000,
      "P4": 0.000000
    },
    {
      "OPECOD": "B",
      "CPNITMNO": "JKBM",
      "CPNITMVER": "巴西棉",
      "CPNLOT": "BRZ/S50609-1",
      "MATRAT": 1,
      "P1": 0.000000,
      "P2": 0.000000,
      "P3": 0.000000,
      "P4": 0.000000
    },
    {
      "OPECOD": "C",
      "CPNITMNO": "JKBM",
      "CPNITMVER": "巴西BCI棉",
      "CPNLOT": "B241107B",
      "MATRAT": 5,
      "P1": 0.000000,
      "P2": 0.000000,
      "P3": 0.000000,
      "P4": 0.000000
    },
    {
      "OPECOD": "D",
      "CPNITMNO": "JKBM",
      "CPNITMVER": "巴西BCI棉",
      "CPNLOT": "B2411072",
      "MATRAT": 18,
      "P1": 0.000000,
      "P2": 0.000000,
      "P3": 0.000000,
      "P4": 0.000000
    },
    {
      "OPECOD": "E",
      "CPNITMNO": "JKBM",
      "CPNITMVER": "巴西BCI棉",
      "CPNLOT": "B241107A",
      "MATRAT": 14,
      "P1": 0.000000,
      "P2": 0.000000,
      "P3": 0.000000,
      "P4": 0.000000
    },
    {
      "OPECOD": "F",
      "CPNITMNO": "JKBM",
      "CPNITMVER": "巴西BCI棉",
      "CPNLOT": "B2409131",
      "MATRAT": 2,
      "P1": 0.000000,
      "P2": 0.000000,
      "P3": 0.000000,
      "P4": 0.000000
    },
    {
      "OPECOD": "G",
      "CPNITMNO": "JKBM",
      "CPNITMVER": "巴西BCI棉",
      "CPNLOT": "B241107B",
      "MATRAT": 2,
      "P1": 0.000000,
      "P2": 0.000000,
      "P3": 0.000000,
      "P4": 0.000000
    },
    {
      "OPECOD": "H",
      "CPNITMNO": "JKBM",
      "CPNITMVER": "巴西BCI棉",
      "CPNLOT": "B2411072",
      "MATRAT": 3,
      "P1": 0.000000,
      "P2": 0.000000,
      "P3": 0.000000,
      "P4": 0.000000
    },
    {
      "OPECOD": "I",
      "CPNITMNO": "JKBM",
      "CPNITMVER": "巴西BCI棉",
      "CPNLOT": "B2411075",
      "MATRAT": 3,
      "P1": 0.000000,
      "P2": 0.000000,
      "P3": 0.000000,
      "P4": 0.000000
    },
    {
      "OPECOD": "J",
      "CPNITMNO": "JKBM",
      "CPNITMVER": "土耳其棉",
      "CPNLOT": "BTX2308-98-1",
      "MATRAT": 2,
      "P1": 0.000000,
      "P2": 0.000000,
      "P3": 0.000000,
      "P4": 0.000000
    }
  ]
}"""
    result = linkrun(json_str)
    print(result)    
