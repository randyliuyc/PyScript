import gradio as gr
import pandas as pd
import numpy as np
from itertools import product
import re

EXCEL_PATH = "merged_ratios_4cols_1.xlsx"
SHEET_NAME = "Sheet1"

white_keys = [
    "W", "SW", "WP", "SWP", "FWP", "WJ", "WPJ", "SWJ", "SWPJ",
    "FW", "FWJ", "FWPJ", "WAO","WC","WB","WUS","WOC","WGEC",
    "WL","WN","WM","WTE","WT"
]

# ===== 加载数据 =====
def load_data():
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=0)
        required_cols = ['STT', '牵伸I', '牵伸II', '牵伸III', '牵伸IV', 'A', 'B', 'C', 'D']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Excel文件缺少以下列: {required_cols}")

        df = df.dropna(subset=['A','B','C','D'])

        for col in 'ABCD':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['A','B','C','D','STT'])

        rows = [{
            'Row': int(row['STT']),
            'Ratios': dict(zip('ABCD', [row[c] for c in 'ABCD'])),
            '牵伸I': row['牵伸I'], '牵伸II': row['牵伸II'], '牵伸III': row['牵伸III'], '牵伸IV': row['牵伸IV'],
            'STT': row['STT'],
            **{c: row[c] for c in 'ABCD'}
        } for _, row in df.iterrows()]
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"数据加载错误: {e}")
        return pd.DataFrame()

# ===== 拆分颜色部分 - 移除最大颜色 =====

def split_color_parts(color_ratios, min_split_value=2.6, max_parts_per_color=12, max_part_value=6):
    """
    将颜色比例拆分成满足条件的小部分:
      - 每部分 >= min_split_value 且 < max_part_value
      - 每种颜色最多拆分成 max_parts_per_color 部分
      - 不再自动移除最大颜色（只处理传入的内容）
    """
    split_dict = {}
    for color, value in color_ratios.items():
        parts = []
        for n in range(1, max_parts_per_color + 1):
            part_val = value / n
            # 条件：在区间 [min_split_value, max_part_value) 内
            if min_split_value <= part_val < max_part_value:
                parts.append((n, round(part_val, 2)))
            elif part_val < min_split_value:
                break

        # 只有在有有效拆分方式时才添加到 split_dict
        if parts:
            split_dict[color] = parts

    return split_dict

import itertools

def __distribute_overflow_parts(combo_parts, df_ratios, max_per_col=6, tolerance=0.5):
    """
    处理超过 max_per_col 的组合（例如：8WC）
    - 拆分为 6 + 余数
    - 将余数分配到其他空列中，使误差最小
    """
    adjusted_parts = []
    extra_assignments = {}

    for num, val, color in combo_parts:
        if num <= max_per_col:
            adjusted_parts.append((num, val, color))
        else:
            # 拆分出6个有效部分
            adjusted_parts.append((max_per_col, val, color))

            # 每个单位的实际值
            per_unit_val = val / num
            # 需要处理的余数
            leftover_units = num - max_per_col
            residual_value = round(leftover_units * per_unit_val, 4)

            # 可用于分配的空列
            candidate_cols = [c for c in "ABCD" if c not in extra_assignments]

            if not candidate_cols:
                continue  # 没有空列可分配

            best_plan, best_error = None, float("inf")

            # 尝试按所有可能方式拆分 leftover_units
            # 例如 leftover_units=2 => [(1,1)], [(2,)]
            for split in itertools.combinations_with_replacement(range(1, leftover_units+1), leftover_units):
                if sum(split) != leftover_units:
                    continue
                if len(split) > len(candidate_cols):
                    continue

                # 将拆分结果匹配到列
                for cols_perm in itertools.permutations(candidate_cols, len(split)):
                    error = 0
                    assignment = {}
                    for units, col in zip(split, cols_perm):
                        assign_val = units * per_unit_val
                        diff = abs(df_ratios[col] - assign_val)
                        error += diff
                        assignment[col] = f"{units}{color}"

                    if error < best_error:
                        best_error, best_plan = error, assignment

            # 如果误差可接受则保存结果
            if best_plan and best_error <= tolerance:
                for col, part in best_plan.items():
                    extra_assignments[col] = part
            else:
                # 如果找不到合适的分配方式，则跳过
                pass

    return adjusted_parts, extra_assignments

def __fill_largest_color_to_remaining(col_infos, largest_color, df_cols_ratios):
    """
    用最大颜色填充剩余位置
    """
    cols = ["A","B","C","D"]
    mapping_str_per_col = {}
    
    for c in cols:
        info = col_infos.get(c, {})
        parts = []
        
        if info.get("nw_color"):
            # 已有匹配的颜色
            parts.append(f"{info['nw_num']}{info['nw_color']}")
            remaining_spots = 6 - int(info['nw_num'])
        else:
            remaining_spots = 6
        
        # 用最大颜色填充剩余部分
        if remaining_spots > 0:
            parts.append(f"{remaining_spots}{largest_color}")
        
        mapping_str_per_col[c] = "+".join(parts) if parts else f"6{largest_color}"
    
    # 计算简单误差
    stats = {"white_error": 0.0, "color_error": 0.0, "total_error": 0.0, "assignment": {}}
    return mapping_str_per_col, stats

def match_colors_to_row_debug(combo, row, tolerance=0.5, priority_colors=None, largest_color=None, color_ratios=None):
    """
    将颜色与行匹配 - 只匹配小颜色，然后填充最大颜色
    """
    if priority_colors is None: 
        priority_colors = []
    if largest_color is None: 
        largest_color = "W"
    
    df_ratios = {c: float(row[c]) for c in "ABCD"}
    
    # 只处理组合中非最大颜色的颜色
    # 步骤1：将组合标准化为列表 (num, val, color)
    combo_parts = [(num, val, color) for num, val, color in combo]

    # 步骤2：调用处理溢出（>6）的函数
    combo_parts, overflow_assignments = __distribute_overflow_parts(combo_parts, df_ratios)

    # 步骤3：处理溢出后重新划分非最大颜色
    non_largest_parts = [part for part in combo_parts if part[2] != largest_color]

    cols, used_cols = "ABCD", set()
    col_infos = {c: {"nw_color": None, "nw_num": 0} for c in cols}

    # 步骤4：如果 overflow_assignments 有已分配的余数
    for col, expr in overflow_assignments.items():
        # 从表达式中提取数量和颜色，例如 "2WC" -> 数量=2, 颜色="WC"
        color_part = ""
        num_part = ""
        for i, char in enumerate(expr):
            if char.isdigit():
                num_part += char
            else:
                color_part = expr[i:]
                break
        
        if num_part and color_part:
            col_infos[col] = {"nw_color": color_part, "nw_num": int(num_part)}
            used_cols.add(col)

    match_nonwhite_error, priority_error = 0.0, 0.0

    # 先匹配小颜色
    for part in non_largest_parts:
        num_parts, val, color = part
        best_col, min_diff = None, float("inf")
        for col in cols:
            if col in used_cols: 
                continue
            diff = abs(df_ratios[col] - float(val))
            if diff <= tolerance and diff < min_diff:
                best_col, min_diff = col, diff
        
        if best_col is None:
            return None
        
        col_infos[best_col] = {"nw_color": color, "nw_num": int(num_parts)}
        used_cols.add(best_col)
        match_nonwhite_error += min_diff
        if color in priority_colors:
            priority_error += min_diff

    # 用最大颜色填充剩余位置
    mapping_str_per_col, stats = __fill_largest_color_to_remaining(col_infos, largest_color, df_ratios)

    # 只从已匹配的小颜色计算总误差
    final_total_error = round(match_nonwhite_error, 4)

    # 创建映射字符串
    mapping = {c: mapping_str_per_col.get(c, f"6{largest_color}") for c in cols}
    mapping_colors = [mapping[c] for c in cols]
    stretch_cols = ["牵伸I", "牵伸II", "牵伸III", "牵伸IV"]
    stretch_vals = [row.get(col, "") for col in stretch_cols]
    stretch_str = "(" + "/".join(str(v) for v in stretch_vals if v != "") + ")"
    mapping_str = "/".join(mapping_colors) + " " + stretch_str

    return {
        "Row": row.get("Row", row.name),
        "Mapping": mapping_str,
        "误差": round(final_total_error, 2),
        "白色误差": 0,
        "颜色误差": round(stats.get("color_error", 0.0), 2),
        "非白色匹配误差": round(match_nonwhite_error, 2),
        "优先误差": round(priority_error, 2),
        "MappingDict": mapping,
        "Ratios": df_ratios
    }

def preview_user_ratios(color_input):
    if not color_input.strip(): 
        return ""
    
    lines, ratios, log, total = color_input.strip().split("\n"), {}, ["📥 用户输入的颜色比例："], 0
    pattern = re.compile(r"^\s*([A-Za-z0-9_\-\(\)\u4e00-\u9fff]+)\s*:\s*([\d\.]+)\s*$")
    
    for line in lines:
        match = pattern.match(line)
        if not match: 
            return f"⚠️ 格式错误：'{line}'。正确格式：颜色名: 数值 (例如 W: 5.0)"
        k, v = match.groups()
        k, val = k.strip().upper(), float(v)
        ratios[k] = val
        log.append(f"- {k}: {val:.2f}%")
        total += val
    
    # 显示最大颜色
    if ratios:
        max_color = max(ratios, key=ratios.get)
        max_ratio = ratios[max_color]
        log.append(f"🎯 最大颜色: {max_color} ({max_ratio:.2f}%) - 将填充到剩余位置")
        
        other_colors = {k: v for k, v in ratios.items() if k != max_color}
        if other_colors:
            log.append(f"🔹 参与组合的颜色: {list(other_colors.keys())}")
    
    missing = 100.0 - total
    log.append(f"🎯 总计: {total:.2f}%")
    if missing > 0: 
        log.append(f"⚠️ 缺少比例: {missing:.2f}%")
    elif missing < 0: 
        log.append(f"⚠️ 总比例超过100%，多出 {-missing:.2f}%")
    
    return "\n".join(log)

def show_product_code_display(code): 
    return (f"### 📌 产品代码: **{code.strip()}**", True) if code.strip() else ("", False)

# ===== 渲染表格 =====
def render_result_table(results, page, page_size=10):
    start, end, page_results = page * page_size, page * page_size + page_size, results[page*page_size:page*page_size+page_size]
    if not page_results: 
        return "⚠️ 没有结果可显示。"
    
    data = []
    for i, r in enumerate(page_results, start=start+1):
        row_info = {"序号": i, "行号": r["Row"]}
        for col in 'ABCD': 
            row_info[col] = f"{r['Ratios'].get(col,0):.2f} → {r['MappingDict'].get(col,'XX')}"
        row_info.update({"误差": r["误差"],"优先误差": r["优先误差"],"排布": r["Mapping"]})
        data.append(row_info)
    
    return pd.DataFrame(data)[["序号","行号"]+list("ABCD")+["误差","优先误差","排布"]].to_markdown(index=False)

def prev_page(results,current,page_size=10): 
    return (render_result_table(results,max(0,current-1),page_size), max(0,current-1))

def next_page(results,current,page_size=10):
    max_page = len(results)//page_size
    return (render_result_table(results,current if current+1>max_page else current+1,page_size), current if current+1>max_page else current+1)

# ===== 运行应用 =====
def run_app(color_input, elongation_limit, priority_input, page_size=10):
    import traceback
    import re
    from itertools import product
    
    log = []
    try:
        # 1️⃣ 解析颜色输入
        ratios, pattern = {}, re.compile(r"^\s*([A-Za-z0-9_\-\(\)\u4e00-\u9fff]+)\s*:\s*([\d\.]+)\s*$")
        for line in color_input.strip().split("\n"):
            match = pattern.match(line)
            if match:
                k, v = match.groups()
                ratios[k.strip().upper()] = float(v.strip())
            else:
                log.append(f"⚠️ 格式错误：'{line}'")

        # 2️⃣ 预先检查总比例
        total_ratio = sum(ratios.values())
        if abs(total_ratio - 100) > 0.001:
            return f"⚠️ 输入的颜色比例总和不等于100% (总和={total_ratio})", "⚠️ 比例无效", [], 0

        # 3️⃣ 找到最大颜色并从组合中分离
        max_color = max(ratios, key=ratios.get)
        max_ratio = ratios[max_color]
        other_colors = {k: v for k, v in ratios.items() if k != max_color}
        
        log.append(f"🎨 最大颜色: {max_color} = {max_ratio:.2f}% (不参与组合)")
        log.append(f"🔹 参与组合的颜色: {list(other_colors.keys())}")

        # 4️⃣ 检查参与组合的颜色 >= 2.6
        min_ratio_threshold = 2.6
        invalid_colors = [f"{color}: {ratio}" for color, ratio in other_colors.items() if ratio < min_ratio_threshold]
        if invalid_colors:
            error_msg = f"⚠️ 由于存在小于 {min_ratio_threshold} 的颜色比例，未找到结果:\n"
            error_msg += "\n".join([f"  - {color_ratio}" for color_ratio in invalid_colors])
            error_msg += f"\n\n📋 要求: 所有颜色（除最大颜色外）必须 >= {min_ratio_threshold}%"
            return error_msg, "⚠️ 颜色比例无效", [], 0

        # 5️⃣ 加载数据并处理组合
        df_all = load_data()
        if df_all.empty:
            return "⚠️ 未找到数据", "⚠️ 无结果", [], 0

        priority_colors = [s.strip().upper() for s in priority_input.split(",") if s.strip()] if priority_input else []
        log.append(f"🔹 优先颜色: {priority_colors}")

        # 仅从小颜色创建组合（不包含最大颜色）
        split_dict = split_color_parts(other_colors)
        # 每种颜色的详细调试信息
        for color, ratio in other_colors.items():
            if color in split_dict:
                splits = split_dict[color]
            else:
                log.append(f"⚠️ {color} ({ratio}%): 无法拆分 - 没有满足 [2.6, 6) 的拆分方式")
        
        log.append(f"🔹 可拆分的颜色总数: {len(split_dict)}/{len(other_colors)}")
        
        if not split_dict:
            log.append("⚠️ 没有颜色可创建组合（只有最大颜色）")
            # 特殊情况：只有最大颜色
            all_results = []
            for idx, row in df_all.iterrows():
                mapping_str = f"6{max_color}/6{max_color}/6{max_color}/6{max_color}"
                stretch_vals = [row.get(f"E{i}", "") for i in range(1, 5)]
                stretch_str = "(" + "/".join(str(v) for v in stretch_vals if v != "") + ")"
                
                result = {
                    "Row": row.get("Row", idx),
                    "Mapping": mapping_str + " " + stretch_str,
                    "误差": 0.0,
                    "优先误差": 0.0,
                    "MappingDict": {c: f"6{max_color}" for c in "ABCD"},
                    "Ratios": {c: float(row[c]) for c in "ABCD"}
                }
                all_results.append(result)
        else:
            # 从 split_dict 创建所有组合
            all_color_parts = [[(num, val, color) for num, val in parts] for color, parts in split_dict.items()]
            all_combos = [list(combo) for combo in product(*all_color_parts) if len(combo) <= 4]
            log.append(f"🔢 可行组合总数: {len(all_combos)} (由于移除最大颜色，显著减少)")

            # 如果有伸长限制则进行过滤
            if elongation_limit:
                try:
                    elong_val = float(elongation_limit)
                    df_all = df_all[(df_all["牵伸I"] <= elong_val) & (df_all["牵伸II"] <= elong_val) & 
                                  (df_all["牵伸III"] <= elong_val) & (df_all["牵伸IV"] <= elong_val)]
                    log.append(f"🔹 过滤伸长 <= {elong_val}，剩余: {len(df_all)} 行")
                except: 
                    pass

            # 匹配组合
            all_results, skipped_combos = [], 0
            for i, combo in enumerate(all_combos, 1):
                combo_results, combo_str = [], " , ".join([f"{num}{color}:{val}" for (num, val, color) in combo])
                for idx, row in df_all.iterrows():
                    try:
                        res = match_colors_to_row_debug(
                            combo, row, tolerance=0.5, 
                            priority_colors=priority_colors, 
                            largest_color=max_color,
                            color_ratios=ratios
                        )
                        if res:
                            res["Combination"] = i
                            combo_results.append(res)
                    except Exception as e_row:
                        log.append(f"❌ 组合 {i} 行 {idx} 匹配错误: {e_row}")
                        
                if not combo_results:
                    skipped_combos += 1
                else:
                    all_results.extend(combo_results)
                    log.append(f"✅ 组合 {i} ({combo_str}) 匹配到 {len(combo_results)} 行")

            log.append(f"📊 总结: {len(all_combos)} 个组合，跳过 {skipped_combos} 个组合，剩余 {len(all_results)} 个结果")

        if not all_results:
            log.append("⚠️ 未找到匹配的结果。")
            return "\n".join(log), "⚠️ 无匹配结果", [], 0

        all_results = sorted(all_results, key=lambda x: (x.get("优先误差", 0), x.get("误差", 0)))
        return "\n".join(log), render_result_table(all_results, 0, page_size), all_results, 0

    except Exception as e:
        return "\n".join([*log, f"❌ 发生异常:\n{traceback.format_exc()}"]), "⚠️ 发生错误", []

# ===== Gradio 界面 =====
def get_four_stretch_app_zh():
    with gr.Blocks() as app:
        gr.Markdown("<h2 style='text-align: center;'>🎨 四项牵伸指标查询系统</h2>")
        with gr.Row():
            with gr.Column(scale=1):
                color_input = gr.Textbox(lines=6,label="🎨 输入颜色和比例",placeholder="B014: 15\nB020: 20\nWC: 65")
                elongation_input = gr.Textbox(label="🧪 过滤伸长指标 (例如: 2.5)")
                priority_input = gr.Textbox(label="🎯 优先颜色误差", placeholder="B014, B020")
            with gr.Column(scale=2):
                realtime_log = gr.Textbox(label="📥 用户输入的比例", lines=12, interactive=False)
                run_btn = gr.Button("🔍 查询")
            with gr.Column(scale=3):
                log_output = gr.Textbox(label="📋 处理信息", lines=15, interactive=False)
        with gr.Row():
            product_code_input = gr.Textbox(label="📦 产品代码名称", placeholder="输入产品代码...")
            product_code_display = gr.Markdown(value="", visible=False)

        table_output = gr.Markdown(label="📊 查询结果")
        results_state = gr.State([])
        current_page = gr.State(0)
        page_size = 10

        run_btn.click(
            fn=run_app,
            inputs=[color_input, elongation_input, priority_input, gr.State(page_size)],
            outputs=[log_output, table_output, results_state, current_page]
        )

        color_input.change(fn=preview_user_ratios, inputs=color_input, outputs=realtime_log)
        product_code_input.change(fn=show_product_code_display, inputs=product_code_input, outputs=[product_code_display, product_code_display])

        with gr.Row():
            prev_btn = gr.Button("⬅️ 上一页")
            next_btn = gr.Button("➡️ 下一页")
        prev_btn.click(fn=prev_page, inputs=[results_state, current_page], outputs=[table_output, current_page])
        next_btn.click(fn=next_page, inputs=[results_state, current_page], outputs=[table_output, current_page])

    return app

four_stretch_app_zh = get_four_stretch_app_zh()
__all__ = ["four_stretch_app_zh"]