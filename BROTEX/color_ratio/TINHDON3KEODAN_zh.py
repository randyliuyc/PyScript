import gradio as gr
import pandas as pd
import numpy as np
from itertools import combinations
from math import ceil
import os
import re
from collections import defaultdict

# ======= 配置 =========
EXCEL_PATH = "expanded_result.xlsx"
SHEET_NAME = "Sheet1"
PICKLE_PATH = "processed_ratios_all1.pkl"
white_keys = ["W", "SW", "WP", "SWP", "FWP", "WJ", "WPJ", "SWJ", "SWPJ", "FW", "FWJ", "FWPJ", "WAO","WC","WB","WUS","WOC","WGEC","WL","WN","WM","WTE","WT"]

# ======= 载入数据函数 =========
def load_data():
    if os.path.exists(PICKLE_PATH):
        return pd.read_pickle(PICKLE_PATH)
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME).iloc[1:]
    df = df[(df['牵伸倍数Ⅰ'] >= 1.1) & (df['牵伸倍数Ⅱ'] >= 1.1) & (df['牵伸倍数Ⅲ'] >= 1.1)]
    for col in 'ABCDEFGH':
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].max() <= 1:
            df[col] *= 100
    df[list('ABCDEFGH')] = df[list('ABCDEFGH')].round(3)
    df['牵伸倍数Ⅰ'] = df['牵伸倍数Ⅰ'].round(2)
    df['牵伸倍数Ⅱ'] = df['牵伸倍数Ⅱ'].round(2)
    df['牵伸倍数Ⅲ'] = df['牵伸倍数Ⅲ'].round(2)
    rows = []
    for idx, row in df.iterrows():
        vals = [row[c] for c in 'ABCDEFGH']
        if any(pd.isnull(vals)):
            continue
        rows.append({
            'Row': idx,
            'Ratios': vals,
            '牵伸倍数Ⅰ': row['牵伸倍数Ⅰ'],
            '牵伸倍数Ⅱ': row['牵伸倍数Ⅱ'],
            '牵伸倍数Ⅲ': row['牵伸倍数Ⅲ'],
            **{c: row[c] for c in 'ABCDEFGH'}
        })
    df_all = pd.DataFrame(rows)
    df_all.to_pickle(PICKLE_PATH)
    return df_all

def adjust_ratios(ratio_dict, num_units=None):
    log = []
    total_white = sum(ratio_dict.get(k, 0) for k in white_keys)
    white_ratios = {k: ratio_dict.get(k, 0) for k in white_keys if k in ratio_dict}
    max_white_color = max(white_ratios, key=white_ratios.get) if white_ratios else None
    max_white_ratio = white_ratios.get(max_white_color, 0) if max_white_color else 0

    max_color_initial = max(ratio_dict, key=ratio_dict.get)
    max_ratio_initial = ratio_dict[max_color_initial]

    log.append(f"🧊 白色总量: {total_white:.2f}")

    mapping_units = {6: 0.25, 5: 0.375, 4: 0.5, 3: 0.625, 2: 0.75, 0: 1.0}
    def get_units_and_percent(value):
        if value >= 90: return 6, 0.25
        elif value >= 85: return 5, 0.375
        elif value >= 75: return 4, 0.5
        elif value >= 70: return 3, 0.625
        elif value >= 65: return 2, 0.75
        else: return 0, 1.0

    if isinstance(num_units, (int, float)) and int(num_units) in mapping_units:
        num_units = int(num_units)
        color_percent = mapping_units[num_units]
        log.append(f"🖐️ 用户选择的单元数: {num_units} → 使用比例: {color_percent:.3f}")
        if color_percent != 1.0:
            log.append(f"🧮 结构: {8 - num_units} 混合单元 + {num_units} 单元拆分")
        else:
            log.append(f"🖐️ 保持原色比例")
    else:
        if total_white > max_ratio_initial:
            units, color_percent = get_units_and_percent(total_white)
            log.append(f"📊 白色总量最大 ({total_white:.2f}) → 使用比例: {color_percent:.3f}")
        else:
            units, color_percent = get_units_and_percent(max_ratio_initial)
            log.append(f"📊 最大比例: {max_ratio_initial:.2f} → 使用比例: {color_percent:.3f}")
        if color_percent == 1.0:
            log.append(f"🖐️ 保持原色比例")
        else:
            log.append(f"🧮 结构: {8 - units} 混合单元 + {units} 单元拆分")

    excluded_colors = set()
    excluded_colors.add(max_color_initial)

    temp_adjusted = {
        k: round(v / color_percent, 2)
        for k, v in ratio_dict.items()
        if k not in excluded_colors
    }
    for k, v in temp_adjusted.items():
        log.append(f"🔎 处理颜色 {k}: {v:.2f}")
    total_after = sum(temp_adjusted.values())
    removed_color = None
    max_color = None

    if total_after > 100:
        excess = total_after - 100
        log.append(f"⚠️ 总和超过100: {total_after:.2f}, 超出: {excess:.2f}")
        candidates = {k: v for k, v in temp_adjusted.items() if v >= excess}
        if candidates:
            removed_color = min(candidates, key=candidates.get)
            log.append(f"🗑️ 移除颜色 {removed_color} (≥ {excess:.2f})")
        else:
            removed_color = max(temp_adjusted, key=temp_adjusted.get)
            log.append(f"🗑️ 没有颜色≥{excess:.2f}，移除最大颜色: {removed_color}")
        
        removed_val = temp_adjusted.pop(removed_color)
        total_after -= removed_val
        excluded_colors.add(removed_color)
        log.append(f"📉 移除后总和: {total_after:.2f}")


    if abs(total_after - 100) < 2 and temp_adjusted:
        max_color = max(temp_adjusted, key=temp_adjusted.get)
    else:
        max_color = None

    no_split_major = max_color is not None

    return temp_adjusted, color_percent, log, max_color_initial, max_white_color, total_after, excluded_colors, no_split_major, max_color
def format_float_keep_one_decimal(x):
    s = f"{x:.2f}"        # giữ 2 chữ số thập phân
    s = s.rstrip('0').rstrip('.')  # bỏ số 0 và dấu chấm thừa
    return s

def match_colors_to_row_debug(color_ratios, row, tolerance=1.5, excluded_colors=None, priority_colors=None, split_threshold=21):
    if excluded_colors is None:
        excluded_colors = {"W", "SW", "FW"}

    df_ratios = {c: row[c] for c in 'ABCDEFGH'}
    log_lines = []
    mapping = {}
    used_cuis = set()

    max_color, max_val = max(color_ratios.items(), key=lambda x: x[1])
    total_ratio = sum(color_ratios.values())
    log_lines.append(f"=== 调试行 {row['Row']} ===")
    log_lines.append(f"🎨 需要匹配的颜色比例: {color_ratios}")
    log_lines.append(f"🔎 A–H 比例: {[round(df_ratios[c], 3) for c in 'ABCDEFGH']}")
    log_lines.append(f"🌈 最大颜色: {max_color} = {max_val:.2f}")
    log_lines.append(f"📊 总比例: {total_ratio:.2f}")

    all_colors = sorted(color_ratios.items(), key=lambda x: -x[1])
    for color, val in all_colors:
        if color == max_color and abs(total_ratio - 100) <= 2.0:
            log_lines.append(f"↪️ 跳过匹配 {color} 因为总比例≈100%")
            continue

        if val > split_threshold:
            best_combo = None
            min_error = float("inf")
            for r in [2, 3]:
                for combo in combinations([c for c in df_ratios if c not in used_cuis], r):
                    s = sum(df_ratios[cui] for cui in combo)
                    error = abs(s - val)
                    if error <= tolerance and error < min_error:
                        best_combo = combo
                        min_error = error
            if best_combo:
                for cui in best_combo:
                    mapping[cui] = color
                    used_cuis.add(cui)
                log_lines.append(f"✅ 分配组合 {best_combo} 给颜色 {color}, 误差 {min_error:.3f}")
            else:
                log_lines.append(f"❌ 未找到适合组合给颜色 {color}")
                return None, "\n".join(log_lines)
        else:
            best_cui = None
            min_diff = float("inf")
            for cui, cui_val in df_ratios.items():
                if cui in used_cuis:
                    continue
                diff = abs(cui_val - val)
                if diff <= tolerance and diff < min_diff:
                    best_cui = cui
                    min_diff = diff
            if best_cui:
                mapping[best_cui] = color
                used_cuis.add(best_cui)
                log_lines.append(f"✅ 分配单元 {best_cui} 给颜色 {color}, 误差 {min_diff:.3f}")
            else:
                log_lines.append(f"❌ 找不到合适的单元给颜色 {color}")
                return None, "\n".join(log_lines)

    remaining = [c for c in df_ratios if c not in used_cuis]
    if abs(total_ratio - 100) <= 2.0:
        log_lines.append(f"🔄 总比例≈100%，填充最大颜色 {max_color} 到剩余部分")
        for cui in remaining:
            mapping[cui] = max_color
            used_cuis.add(cui)
    else:
        fill_color = next(iter(excluded_colors)) if excluded_colors else "W"
        log_lines.append(f"❗ 总比例≠100%，填充颜色 {fill_color} 到剩余部分")
        for cui in remaining:
            mapping[cui] = fill_color
            used_cuis.add(cui)

    actual_by_color = defaultdict(float)
    for cui, color in mapping.items():
        if color in color_ratios:
            actual_by_color[color] += df_ratios[cui]

    total_error = 0
    priority_error = 0
    for color, expected_val in color_ratios.items():
        actual_val = actual_by_color[color]
        diff = abs(actual_val - expected_val)
        total_error += diff
        log_lines.append(f"📐 误差 {color}: 实际 {actual_val:.2f} vs 期望 {expected_val:.2f} → {diff:.2f}")
        if priority_colors and color in priority_colors:
            priority_error += diff    
    result_parts = []
    cols = list("ABCDEFGH")
    i = 0

    while i < len(cols):
        c1 = cols[i]
        label1 = mapping[c1]
        val1 = df_ratios[c1]

        if i + 1 < len(cols):
            c2 = cols[i + 1]
            label2 = mapping[c2]
            val2 = df_ratios[c2]

            if label1 == label2 and abs(val1 - val2) < 0.2:
                # Giống màu, giống tỉ lệ
                result_parts.append(f"2{label1}")
            elif abs(val1 - val2) < 0.2:
                # Khác màu nhưng giống tỉ lệ
                result_parts.append(f"1{label1}+1{label2}")
            else:
                # Khác màu và khác tỉ lệ
                result_parts.append(f"1{label1}/1{label2}")
            i += 2
        else:
            # Chỉ còn một cúi
            result_parts.append(f"1{label1}")
            i += 1

    # Ghép các phần

    mapping_str = (
        "/".join(result_parts) + " (" +
        f"{format_float_keep_one_decimal(row['牵伸倍数Ⅰ'])}/" +
        f"{format_float_keep_one_decimal(row['牵伸倍数Ⅱ'])}/" +
        f"{format_float_keep_one_decimal(row['牵伸倍数Ⅲ'])})"
    )
    return {
        "Row": row["Row"],
        "Mapping": mapping_str,
        "Sai số": round(total_error, 2),
        "Sai số ưu tiên": round(priority_error, 2),
        "Log": "\n".join(log_lines),
        "Ratios": df_ratios,
        "MappingDict": mapping
    }, None


# ======= 界面函数 =========
def preview_user_ratios(color_input):
    if not color_input.strip():
        return ""
    lines = color_input.strip().split("\n")
    ratios = {}
    log = ["📥 用户输入的颜色比例:"]
    total = 0
    pattern = re.compile(r"^\s*([A-Za-z0-9_\-\(\)\u4e00-\u9fff]+)\s*:\s*([\d\.]+)\s*$")
    for line in lines:
        match = pattern.match(line)
        if not match:
            return f"⚠️ 格式错误，行: '{line}'。正确格式示例: 颜色名: 数字 (如 W: 5.0)"
        k, v = match.groups()
        k = k.strip().upper()
        val = float(v)
        ratios[k] = val
        log.append(f"- {k}: {val:.2f}%")
        total += val
    missing = 100.0 - total
    log.append(f"🎯 总计: {total:.2f}%")
    if missing > 0:
        log.append(f"⚠️ 比例不足: {missing:.2f}%")
    elif missing < 0:
        log.append(f"⚠️ 总比例超过100%，多出 {-missing:.2f}%")
    return "\n".join(log)

def get_structure_line_from_textbox(num_units_str):
    try:
        num_units = int(num_units_str)
        mapping_units = {
            6: "🧱 结构: 2 混合单元 + 6 拆分单元",
            5: "🧱 结构: 3 混合单元 + 5 拆分单元",
            4: "🧱 结构: 4 混合单元 + 4 拆分单元",
            3: "🧱 结构: 5 混合单元 + 3 拆分单元",
            2: "🧱 结构: 6 混合单元 + 2 拆分单元",
            0: "🧱 保持原结构不拆分"
        }
        return mapping_units.get(num_units, "")
    except:
        return ""

def show_product_code_display(code):
    if code.strip():
        return f"### 📌 产品代码: **{code.strip()}**", True
    else:
        return "", False
def render_result_table(results, page, page_size=50):
    start = page * page_size
    end = start + page_size
    page_results = results[start:end]
    if not page_results:
        return "⚠️ Không có kết quả để hiển thị."
    data = []
    for i, r in enumerate(page_results, start=start + 1):
        row_info = {
            "STT": i,
            "Row": r["Row"],
        }
        for col in 'ABCDEFGH':
            val = r["Ratios"].get(col, 0)
            label = r["MappingDict"].get(col, "XX")
            row_info[col] = f"{val:.2f} → {label}"
        row_info["误差"] = r["Sai số"]
        row_info["优先误差"] = r["Sai số ưu tiên"]
        row_info["配纱方案"] = r["Mapping"]
        data.append(row_info)
    columns_order = ["STT", "Row"] + list("ABCDEFGH") + ["误差", "优先误差", "配纱方案"]
    df_result = pd.DataFrame(data)[columns_order]
    return df_result.to_markdown(index=False)

def prev_page(results, current, page_size=50):
    if current <= 0:
        return render_result_table(results, 0, page_size), 0
    return render_result_table(results, current - 1, page_size), current - 1

def next_page(results, current, page_size=50):
    max_page = len(results) // page_size
    if current + 1 > max_page:
        return render_result_table(results, current, page_size), current
    return render_result_table(results, current + 1, page_size), current + 1
# ======= 主处理函数 =========
def run_app(color_input, num_units, elongation_limit, priority_input, split_threshold_input):
    log = []
    try:
        if not color_input.strip():
            return "⚠️ 请填写颜色比例。", "", "", [], 0

        lines = color_input.strip().split("\n")
        ratios = {}
        pattern = re.compile(r"^\s*([A-Za-z0-9_\-\(\)\u4e00-\u9fff]+)\s*:\s*([\d\.]+)\s*$")
        for line in lines:
            match = pattern.match(line)
            if match:
                k, v = match.groups()
                ratios[k.strip().upper()] = float(v.strip())

        total_ratio = sum(ratios.values())
        if not (99.9 <= total_ratio <= 100.1):
            return f"⚠️ 总比例必须为 100%。当前总和为: {total_ratio:.2f}%", "", "", [], 0
        if len(ratios) < 2:
            return "⚠️ 至少需要两种颜色进行匹配。", "", "", [], 0

        log.append(f"🎨 接收到 {len(ratios)} 种颜色: {list(ratios.keys())}")
        log.append(f"📊 输入比例: {ratios}")

        priority_colors = [s.strip().upper() for s in priority_input.split(",") if s.strip()] if priority_input else []
        if priority_colors:
            log.append(f"🔍 优先匹配颜色（误差优先）: {priority_colors}")

        split_threshold = float(split_threshold_input) if split_threshold_input else 21

        df_all = load_data()

        if elongation_limit:
            try:
                elongation_limit = float(elongation_limit)
                df_all = df_all[
                    (df_all["牵伸倍数Ⅰ"] <= elongation_limit) &
                    (df_all["牵伸倍数Ⅱ"] <= elongation_limit) &
                    (df_all["牵伸倍数Ⅲ"] <= elongation_limit)
                ]
                log.append(f"🔍 过滤牵伸倍数 ≤ {elongation_limit}：剩余 {len(df_all)} 条记录")
                if df_all.empty:
                    return "\n".join(log + ["❌ 没有满足牵伸倍数条件的数据。"]), "", "", [], 0
            except ValueError:
                return "\n".join(log + ["⚠️ 牵伸倍数格式错误。"]), "", "", [], 0

        num_units = int(num_units) if num_units else None
        adjusted_ratios, color_percent, adjust_log, max_color_initial, max_white_color, total_after, excluded_colors, no_split_major, max_color = adjust_ratios(ratios, num_units)
        log.extend(adjust_log)
        log.append(f"🔄 调整后比例: {adjusted_ratios}")
        log.append(f"🔄 排除颜色 + 临时颜色: {excluded_colors}")

        results = []
        for idx, row in df_all.iterrows():
            res, _ = match_colors_to_row_debug(
                adjusted_ratios,
                row,
                tolerance=2.0,
                excluded_colors=excluded_colors,
                priority_colors=priority_colors,
                split_threshold=split_threshold
            )
            if res:
                results.append(res)

        results = sorted(results, key=lambda x: (x["Sai số ưu tiên"], x["Sai số"]))

        if not results:
            return "\n".join(log + ["❌ 未找到匹配结果。"]), "", "", [], 0

        first_page_table = render_result_table(results, 0)
        return "\n".join(log), "", first_page_table, results, 0

    except Exception as e:
        return f"⚠️ 错误: {str(e)}", "", "", [], 0


# ======= Gradio界面 =========
def get_three_stretch_app():
    with gr.Blocks() as app:
        gr.Markdown("<h2 style='text-align: center;'>🎨 牵伸倍数配比查询工具</h2>")
        with gr.Row():
            with gr.Column(scale=1):
                color_input = gr.Textbox(lines=6, label="🎨 输入颜色及比例", placeholder="G004: 18.0\nG024: 40.0\nXX: 42.0")
                num_units_input = gr.Textbox(label="🔹 要拆分的粗纱数（2–6，选填）", placeholder="例如: 3")
                elongation_limit_input = gr.Textbox(label="🧪 拉伸倍数上限（例如：2.5）")
                priority_color_input = gr.Textbox(label="🎯 优先匹配颜色（用于误差优化）", placeholder="例如: G004, G024")
            with gr.Column(scale=2):
                realtime_log = gr.Textbox(label="📥 用户输入的比例", lines=8, interactive=False)
                structure_line = gr.Textbox(label="🧱 拆分结构", interactive=False)
                split_threshold_input = gr.Textbox(label="✂️ 拆分阈值（例如：21）", placeholder="例如: 21")
                run_btn = gr.Button("🔍 查询配比")
            with gr.Column(scale=3):
                log_output = gr.Textbox(label="📋 处理信息", lines=19, interactive=False)
        
        gr.Markdown("## 🏽️ 输入产品编号")
        with gr.Row():
            product_code_input = gr.Textbox(label="📦 产品编号", placeholder="请输入产品编号...")
            product_code_display = gr.Markdown(value="", visible=False)
        
        table_output = gr.Markdown(label="📊 查询结果")

        results_state = gr.State([])
        current_page = gr.State(0)
        page_size = 10

        run_btn.click(
            fn=run_app,
            inputs=[color_input, num_units_input, elongation_limit_input, priority_color_input, split_threshold_input],
            outputs=[log_output, structure_line, table_output, results_state, current_page]
        )

        color_input.change(
            fn=preview_user_ratios,
            inputs=color_input,
            outputs=realtime_log
        )

        num_units_input.change(
            fn=get_structure_line_from_textbox,
            inputs=num_units_input,
            outputs=structure_line
        )

        product_code_input.change(
            fn=show_product_code_display,
            inputs=product_code_input,
            outputs=[product_code_display, product_code_display]
        )

        with gr.Row():
            prev_btn = gr.Button("⬅️ 上一页")
            next_btn = gr.Button("➡️ 下一页")

        prev_btn.click(
            fn=prev_page,
            inputs=[results_state, current_page],
            outputs=[table_output, current_page]
        )

        next_btn.click(
            fn=next_page,
            inputs=[results_state, current_page],
            outputs=[table_output, current_page]
        )

    return app

three_stretch_app_zh = get_three_stretch_app()
__all__ = ["three_stretch_app_zh"]
