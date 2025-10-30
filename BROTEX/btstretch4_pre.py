import pandas as pd
from itertools import product

def calculate_ratios(X1, X2, X3, X4):
  """计算 A-H 比例（与 bt8.py 中的 speeds_from_X 逻辑一致）"""
  inv_sum = (1/X1 + 2/X4 + 2 + 2/X2 + 1/X3)
  ratios = {
    'A': round((1/X1 / inv_sum) * 100, 2),
    'B': round((1/X4 / inv_sum) * 100, 2),
    'C': round((1 / inv_sum) * 100, 2),
    'D': round((1 / inv_sum) * 100, 2),
    'E': round((1/X2 / inv_sum) * 100, 2),
    'F': round((1/X2 / inv_sum) * 100, 2),
    'G': round((1/X4 / inv_sum) * 100, 2),
    'H': round((1/X3 / inv_sum) * 100, 2)
  }
  ratios['Total'] = round(sum(ratios.values()), 2)
  return ratios

# 生成所有有效组合
data = []
for X1, X2, X3, X4 in product(
  [round(x * 0.1, 1) for x in range(10, 41)],  # X1: 1.0-4.0
  [round(x * 0.1, 1) for x in range(10, 41)],  # X2: 1.0-4.0
  [round(x * 0.1, 1) for x in range(10, 41)],  # X3: 1.0-4.0
  [round(x * 0.1, 1) for x in range(10, 61)]   # X4: 1.0-6.0
):
  if X4 > X1 and X4 > X3 and (X4 / X1 < 4.0) and (X4 / X3 < 4.0):
    ratios = calculate_ratios(X1, X2, X3, X4)
    data.append({
      'stretch1': X1,
      'stretch2': X2,
      'stretch3': X3,
      'stretch4': X4,
      **ratios
    })

# 保存为 Excel
df = pd.DataFrame(data)
df.to_excel("btstretch4_precomputed_ratios.xlsx", index=False)
