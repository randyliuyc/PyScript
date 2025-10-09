import gradio as gr
import pandas as pd
import numpy as np
from itertools import combinations
from math import ceil
import os
import re
from collections import defaultdict

# ======= CẤU HÌNH =========
EXCEL_PATH = "merged_ratios.xlsx"
SHEET_NAME = "Sheet1"
PICKLE_PATH = "processed_ratios_all1.pkl"
white_keys = ["W", "SW", "WP", "SWP", "FWP", "WJ", "WPJ", "SWJ", "SWPJ", "FW", "FWJ", "FWPJ", "WAO","WC","WB","WUS","WOC","WGEC","WL","WN","WM","WTE","WT"]

# ======= HÀM CHUẨN HÓA SỐ =========
def normalize_number(value_str):
    """
    Chuẩn hóa chuỗi số, chấp nhận cả dấu . và , làm dấu thập phân
    VD: "2,5" -> 2.5, "2.5" -> 2.5
    """
    if not isinstance(value_str, str):
        return value_str
    # Thay thế dấu , thành . để chuẩn hóa
    return value_str.replace(',', '.')

# ======= LOGIC TẢI DỮ LIỆU TỪ CODE THỨ HAI =========
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

def parse_elongation_filter(elongation_input):
    """
    Parse input kéo dãn theo các format:

    - "2.5" hoặc "2,5" -> tất cả chỉ số ≤ 2.5 (logic cũ)
    - "2.5, 1.3" hoặc "2,5; 1,3" -> tất cả ≤ 2.5 VÀ phải có ít nhất 1 chỉ số = 1.3
    - "1.5, 1.3, 2.5" hoặc "1,5; 1,3; 2,5" -> chính xác: 牵伸倍数Ⅰ=1.5, 牵伸倍数Ⅱ=1.3, 牵伸倍数Ⅲ=2.5
    - "max:2.5, fixed:1.3" -> tất cả ≤ 2.5 VÀ phải có ít nhất 1 chỉ số = 1.3
    - "exact:1.5,1.3,2.5" -> chính xác theo thứ tự I,II,III
    """
    if not elongation_input.strip():
        return None, None, None, ""

    elongation_input = normalize_number(elongation_input.strip())
    max_val = None
    fixed_val = None
    exact_vals = None

    try:
        # Kiểm tra format có exact: không
        if elongation_input.lower().startswith('exact:'):
            exact_part = elongation_input.split(':', 1)[1].strip()
            exact_vals = [float(v.strip()) for v in exact_part.split(',') if v.strip()]
            if len(exact_vals) != 3:
                return None, None, None, "⚠️ Format exact cần đúng 3 số: exact:1.5,1.3,2.5"
            log_msg = f"🔍 Lọc chính xác: I={exact_vals[0]}, II={exact_vals[1]}, III={exact_vals[2]}"
            return None, None, exact_vals, log_msg

        # Kiểm tra format có max:/fixed: không
        elif "max:" in elongation_input.lower() or "fixed:" in elongation_input.lower():
            parts = [p.strip() for p in elongation_input.split(',')]
            for part in parts:
                if part.lower().startswith('max:'):
                    max_val = float(part.split(':', 1)[1].strip())
                elif part.lower().startswith('fixed:'):
                    fixed_val = float(part.split(':', 1)[1].strip())
        else:
            # Format số thuần túy - hỗ trợ cả dấu , và ;
            if ',' in elongation_input or ';' in elongation_input:
                # Tách theo cả , và ;
                parts = re.split(r'[,;]', elongation_input)
                parts = [float(p.strip()) for p in parts if p.strip()]

                if len(parts) == 3:
                    # 3 số -> chính xác theo thứ tự I, II, III
                    exact_vals = parts
                    log_msg = f"🔍 Lọc chính xác: I={exact_vals[0]}, II={exact_vals[1]}, III={exact_vals[2]}"
                    return None, None, exact_vals, log_msg
                elif len(parts) == 2:
                    # 2 số: số đầu là max, số thứ 2 là fixed
                    max_val = parts[0]
                    fixed_val = parts[1]
                elif len(parts) == 1:
                    max_val = parts[0]
            else:
                # Chỉ có 1 số -> giữ logic cũ (chỉ filter max)
                max_val = float(elongation_input)

        # Tạo log message cho trường hợp max + fixed
        if max_val is not None and fixed_val is not None:
            log_msg = f"🔍 Lọc: tất cả ≤ {max_val} VÀ phải có ít nhất 1 chỉ số = {fixed_val}"
        elif max_val is not None:
            log_msg = f"🔍 Lọc theo kéo dãn ≤ {max_val}"
        elif fixed_val is not None:
            log_msg = f"🔍 Lọc: phải có ít nhất 1 chỉ số = {fixed_val}"
        else:
            log_msg = ""

        return max_val, fixed_val, exact_vals, log_msg

    except ValueError:
        return None, None, None, "⚠️ Format kéo dãn không hợp lệ"

def adjust_ratios(ratio_dict, num_units=None):
    log = []
    total_white = sum(ratio_dict.get(k, 0) for k in white_keys)
    white_ratios = {k: ratio_dict.get(k, 0) for k in white_keys if k in ratio_dict}
    max_white_color = max(white_ratios, key=white_ratios.get) if white_ratios else None
    max_white_ratio = white_ratios.get(max_white_color, 0) if max_white_color else 0

    max_color_initial = max(ratio_dict, key=ratio_dict.get)
    max_ratio_initial = ratio_dict[max_color_initial]

    log.append(f"🧊 Tổng màu trắng: {total_white:.2f}")

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
        log.append(f"🖐️ Người dùng chọn số cúi: {num_units} → Dùng tỉ lệ: {color_percent:.3f}")
        if color_percent != 1.0:
            log.append(f"🧮 Cấu trúc: {8 - num_units} CÚI HỖN HỢP + {num_units} CÚI TÁCH ")
        else:
            log.append(f"🖐️ Giữ nguyên tỉ lệ màu")
    else:
        if total_white > max_ratio_initial:
            units, color_percent = get_units_and_percent(total_white)
            log.append(f"📊 Tổng màu trắng là lớn nhất ({total_white:.2f}) → Dùng tỷ lệ: {color_percent:.3f}")
        else:
            units, color_percent = get_units_and_percent(max_ratio_initial)
            log.append(f"📊 Tỉ lệ lớn nhất: {max_ratio_initial:.2f} → Dùng tỷ lệ: {color_percent:.3f}")
        if color_percent == 1.0:
            log.append(f"🖐️ Giữ nguyên tỉ lệ màu")
        else:
            log.append(f"🧮 Cấu trúc: {8 - units} CÚI HỖN HỢP + {units} CÚI TÁCH")

    excluded_colors = set()
    excluded_colors.add(max_color_initial)

    temp_adjusted = {
        k: round(v / color_percent, 2)
        for k, v in ratio_dict.items()
        if k not in excluded_colors
    }
    for k, v in temp_adjusted.items():
        log.append(f"🔎 Đang xử lí màu {k}: {v:.2f}")
    total_after = sum(temp_adjusted.values())
    removed_color = None
    max_color = None

    if total_after > 105:
        excess = total_after - 100
        log.append(f"⚠️ Tổng vượt quá 100: {total_after:.2f}, dư: {excess:.2f}")
        candidates = {k: v for k, v in temp_adjusted.items() if v >= excess}
        if candidates:
            removed_color = min(candidates, key=candidates.get)
            log.append(f"🗑️ Loại bỏ màu {removed_color} (≥ {excess:.2f})")
        else:
            removed_color = max(temp_adjusted, key=temp_adjusted.get)
            log.append(f"🗑️ Không có màu ≥ {excess:.2f}, loại màu lớn nhất: {removed_color}")

        removed_val = temp_adjusted.pop(removed_color)
        total_after -= removed_val
        excluded_colors.add(removed_color)
        log.append(f"📉 Tổng sau loại: {total_after:.2f}")

    # Xác định lại max_color dựa trên temp_adjusted sau khi loại bỏ
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
    log_lines.append(f"=== Debug Row {row['Row']} ===")
    log_lines.append(f"🎨 Tỷ lệ màu cần match: {color_ratios}")
    log_lines.append(f"🔎 Tỷ lệ A–H: {[round(df_ratios[c], 3) for c in 'ABCDEFGH']}")
    log_lines.append(f"🌈 Màu lớn nhất: {max_color} = {max_val:.2f}")
    log_lines.append(f"📊 Tổng tỉ lệ: {total_ratio:.2f}")

    all_colors = sorted(color_ratios.items(), key=lambda x: -x[1])
    for color, val in all_colors:
        if color == max_color and abs(total_ratio - 100) <= 2.0:
            log_lines.append(f"↪️ Bỏ qua match {color} vì tổng tỉ lệ ≈ 100%")
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
                log_lines.append(f"✅ Gán tổ hợp {best_combo} cho màu {color}, sai số {min_error:.3f}")
            else:
                log_lines.append(f"❌ Không tìm tổ hợp phù hợp cho màu {color}")
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
                log_lines.append(f"✅ Gán cúi {best_cui} cho màu {color}, sai số {min_diff:.3f}")
            else:
                log_lines.append(f"❌ Không tìm được cúi phù hợp cho màu {color}")
                return None, "\n".join(log_lines)

    remaining = [c for c in df_ratios if c not in used_cuis]
    if abs(total_ratio - 100) <= 2.0:
        log_lines.append(f"🔄 Tổng tỉ lệ ≈ 100%. Điền màu lớn nhất {max_color} vào phần còn lại")
        for cui in remaining:
            mapping[cui] = max_color
            used_cuis.add(cui)
    else:
        fill_color = next(iter(excluded_colors)) if excluded_colors else "W"
        log_lines.append(f"❗ Tổng tỉ lệ ≠ 100%. Điền {fill_color} vào phần còn lại")
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
        log_lines.append(f"📐 Sai số {color}: thực tế {actual_val:.2f} vs mong muốn {expected_val:.2f} → {diff:.2f}")
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
                result_parts.append(f"2{label1}")
            elif abs(val1 - val2) < 0.2:
                result_parts.append(f"1{label1}+1{label2}")
            else:
                result_parts.append(f"1{label1}/1{label2}")
            i += 2
        else:
            result_parts.append(f"1{label1}")
            i += 1

    mapping_str = "/".join(result_parts) + f" ({format_float_keep_one_decimal(row['牵伸倍数Ⅰ'])}/" \
                                        f"{format_float_keep_one_decimal(row['牵伸倍数Ⅱ'])}/" \
                                        f"{format_float_keep_one_decimal(row['牵伸倍数Ⅲ'])})"
    return {
        "Row": row["Row"],
        "Mapping": mapping_str,
        "Sai số": round(total_error, 2),
        "Sai số ưu tiên": round(priority_error, 2),
        "Log": "\n".join(log_lines),
        "Ratios": df_ratios,
        "MappingDict": mapping
    }, None

def parse_arrangement_to_positions(arrangement_input):
    """
    Parse từ format sắp cúi thành mapping vị trí.

    Hỗ trợ 2 dạng:
    1. Kiểu tuần tự (theo A–H): "1WC/1WC/1B01/1WC/1WC/1WC"
    2. Kiểu chỉ định vị trí: "A:WC, C:B01, F:G02"
    """
    if not arrangement_input.strip():
        return {}

    position_mapping = {}
    positions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # --- Bỏ phần kéo dài ở cuối nếu có (ví dụ "(2.6/1.3/1.1)") ---
    arrangement_input = arrangement_input.strip()
    arrangement_input = re.sub(r"\([^)]*\)$", "", arrangement_input).strip()

    # --- Trường hợp chỉ định vị trí (có dấu :) ---
    if ":" in arrangement_input:
        parts = re.split(r'[;,]', arrangement_input)  # cho phép , hoặc ; phân cách
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if ':' in part:
                pos, color = part.split(':', 1)
                pos, color = pos.strip().upper(), color.strip().upper()
                if pos in positions:
                    position_mapping[pos] = color
        return position_mapping

    # --- Trường hợp tuần tự (có dấu /) ---
    parts = arrangement_input.split('/')
    current_pos = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if '+' in part:  # tách nhiều màu trong một cụm
            sub_parts = part.split('+')
            for sub_part in sub_parts:
                m = re.match(r'(\d+)?([A-Za-z0-9_\-\(\)\u4e00-\u9fff]+)', sub_part.strip())
                if m:
                    count, color = m.groups()
                    count = int(count) if count else 1
                    color = color.upper()
                    for _ in range(count):
                        if current_pos < len(positions):
                            position_mapping[positions[current_pos]] = color
                            current_pos += 1
        else:
            m = re.match(r'(\d+)?([A-Za-z0-9_\-\(\)\u4e00-\u9fff]+)', part)
            if m:
                count, color = m.groups()
                count = int(count) if count else 1
                color = color.upper()
                for _ in range(count):
                    if current_pos < len(positions):
                        position_mapping[positions[current_pos]] = color
                        current_pos += 1

    return position_mapping

def check_arrangement_filter(result, arrangement_filters):
    """
    Kiểm tra xem result có thỏa mãn các điều kiện sắp cúi không
    """
    if not arrangement_filters:
        return True

    mapping_dict = result.get("MappingDict", {})

    for position, expected_colors in arrangement_filters.items():
        actual_color = mapping_dict.get(position, "")

        if '+' in expected_colors:
            # Trường hợp màu trộn - kiểm tra xem actual_color có trong danh sách expected không
            expected_list = [c.strip() for c in expected_colors.split('+')]
            if actual_color not in expected_list:
                return False
        else:
            # Trường hợp màu đơn
            if actual_color != expected_colors:
                return False

    return True

def preview_arrangement_filters(arrangement_input):
    """Preview cách hiểu sắp cúi của người dùng"""
    if not arrangement_input.strip():
        return ""

    try:
        position_mapping = parse_arrangement_to_positions(arrangement_input)
        if not position_mapping:
            return "⚠️ Không thể parse format sắp cúi. VD đúng: 1G02/1G02/1G01+1SW/2SW/1G02/1SW"

        positions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        # Gom tất cả vào 1 dòng

        # Visualization
        visual = "🎨 Visualization: "
        for pos in positions:
            color = position_mapping.get(pos, "?")
            visual += f"{pos}({color}) "

        return  visual.strip()

    except Exception as e:
        return f"⚠️ Lỗi parse: {str(e)}"

def preview_elongation_filter(elongation_input):
    """Preview điều kiện lọc kéo dãn"""
    min_val, max_val, exact_vals, log_msg = parse_elongation_filter(elongation_input)
    return log_msg if log_msg else ""

# ======= CÁC HÀM GIAO DIỆN =========
def combine_color_inputs(color_names, color_ratios):
    """Kết hợp tên màu và tỷ lệ từ 2 input riêng biệt - hỗ trợ cả dấu . và ,"""
    if not color_names.strip() or not color_ratios.strip():
        return ""

    name_lines = [line.strip() for line in color_names.strip().split("\n") if line.strip()]
    ratio_lines = [normalize_number(line.strip()) for line in color_ratios.strip().split("\n") if line.strip()]

    if len(name_lines) != len(ratio_lines):
        return f"⚠️ Số lượng tên màu ({len(name_lines)}) khác với số tỷ lệ ({len(ratio_lines)})"

    combined_lines = []
    for name, ratio in zip(name_lines, ratio_lines):
        combined_lines.append(f"{name}: {ratio}")

    return "\n".join(combined_lines)

def preview_combined_ratios(color_names, color_ratios):
    """Preview tỷ lệ màu sau khi combine"""
    combined_input = combine_color_inputs(color_names, color_ratios)
    if not combined_input or combined_input.startswith("⚠️"):
        return combined_input

    return preview_user_ratios(combined_input)

def preview_user_ratios(color_input):
    if not color_input.strip():
        return ""
    lines = color_input.strip().split("\n")
    ratios = {}
    log = ["📥 Tỷ lệ màu người dùng đã nhập:"]
    total = 0

    # Cho phép: "B014: 12" | "B014\t12" | "B014    12" | "B014, 12" | "B014 12%" ...
    # Đã chuẩn hóa dấu , thành . trong hàm normalize_number
    pattern = re.compile(
        r"^\s*([A-Za-z0-9_\-\(\)\u4e00-\u9fff]+)\s*[:\t,; ]+\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$"
    )

    for line in lines:
        m = pattern.match(line)
        if not m:
            return f"⚠️ Sai định dạng ở dòng: '{line}'. Đúng dạng: Tên: số (ví dụ W: 5.0 hoặc W: 5,0)"
        k, v = m.groups()
        k = k.strip().upper()
        val = float(v)
        ratios[k] = val
        log.append(f"- {k}: {val:.2f}%")
        total += val

    missing = 100.0 - total
    log.append(f"🎯 Tổng cộng: {total:.2f}%")
    if missing > 0:
        log.append(f"⚠️ Tỉ lệ còn thiếu: {missing:.2f}%")
    elif missing < 0:
        log.append(f"⚠️ Tổng tỉ lệ vượt quá 100% thừa {-missing:.2f}%")
    return "\n".join(log)

def get_structure_line_from_textbox(num_units_str):
    try:
        num_units = int(num_units_str)
        mapping_units = {
            6: "🧱 Cấu trúc: 2 CÚI HỖN HỢP + 6 CÚI TÁCH",
            5: "🧱 Cấu trúc: 3 CÚI HỖN HỢP + 5 CÚI TÁCH",
            4: "🧱 Cấu trúc: 4 CÚI HỖN HỢP + 4 CÚI TÁCH",
            3: "🧱 Cấu trúc: 5 CÚI HỖN HỢP + 3 CÚI TÁCH",
            2: "🧱 Cấu trúc: 6 CÚI HỖN HỢP + 2 CÚI TÁCH",
            0: "🧱 giữ nguyên cấu trúc không tách"
        }
        return mapping_units.get(num_units, "")
    except:
        return ""

def show_product_code_display(code):
    if code.strip():
        return f"### 📌 Mã hàng: **{code.strip()}**", True
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
        row_info["Sai số"] = r["Sai số"]
        row_info["Sai số ƯT"] = r["Sai số ưu tiên"]
        row_info["Sắp cúi"] = r["Mapping"]
        data.append(row_info)
    columns_order = ["STT", "Row"] + list("ABCDEFGH") + ["Sai số", "Sai số ƯT", "Sắp cúi"]
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

def run_app(color_names, color_ratios, num_units, elongation_limit, priority_input, split_threshold_input, arrangement_filter_input):
    log = []
    try:
        # Kết hợp tên màu và tỷ lệ
        color_input = combine_color_inputs(color_names, color_ratios)
        if color_input.startswith("⚠️"):
            return color_input, "", "", [], 0

        if not color_input.strip():
            return "⚠️ Vui lòng nhập tên màu và tỷ lệ màu.", "", "", [], 0

        lines = color_input.strip().split("\n")
        ratios = {}
        # Pattern đã được cập nhật để chấp nhận dấu . (sau khi normalize)
        pattern = re.compile(
            r"^\s*([A-Za-z0-9_\-\(\)\u4e00-\u9fff]+)\s*[:\t,; ]+\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$"
        )
        for line in lines:
            match = pattern.match(line)
            if match:
                k, v = match.groups()
                ratios[k.strip().upper()] = float(v.strip())
            else:
                return f"⚠️ Sai định dạng ở dòng: '{line}'. Đúng dạng: Tên: số (ví dụ W: 5.0 hoặc W: 5,0)", "", "", [], 0
        total_ratio = sum(ratios.values())
        if abs(total_ratio - 100.0) > 0.01:
            return f"⚠️ Tổng tỷ lệ phải là 100%. Hiện tại: {total_ratio:.2f}%", "", "", [], 0
        if len(ratios) < 2:
            return "⚠️ Cần ít nhất 2 màu để tra cứu.", "", "", [], 0

        log.append(f"🎨 Nhận được {len(ratios)} màu: {list(ratios.keys())}")
        log.append(f"📊 Tỷ lệ: {ratios}")

        # Parse arrangement filters
        arrangement_filters = parse_arrangement_to_positions(arrangement_filter_input)
        if arrangement_filters:
            log.append(f"🎯 Điều kiện lọc sắp cúi: {arrangement_filters}")

        priority_colors = [s.strip().upper() for s in priority_input.split(",") if s.strip()] if priority_input else []
        if priority_colors:
            log.append(f"🔍 Màu ưu tiên sai số: {priority_colors}")

        # Chuẩn hóa split_threshold_input
        split_threshold_input = normalize_number(split_threshold_input) if split_threshold_input else "21"
        split_threshold = float(split_threshold_input)

        df_all = load_data()

        # Parse và áp dụng filter kéo dãn với các mode khác nhau
        if elongation_limit:
            max_elongation, fixed_elongation, exact_elongations, elongation_log = parse_elongation_filter(elongation_limit)

            if elongation_log.startswith("⚠️"):
                return elongation_log, "", "", [], 0

            if elongation_log:
                log.append(elongation_log)

            # Áp dụng filter
            original_count = len(df_all)
            tolerance = 0.01  # Sai số cho phép khi so sánh float

            if exact_elongations is not None:
                # Mode 3: Lọc chính xác theo từng chỉ số I, II, III
                df_all = df_all[
                    (abs(df_all["牵伸倍数Ⅰ"] - exact_elongations[0]) <= tolerance) &
                    (abs(df_all["牵伸倍数Ⅱ"] - exact_elongations[1]) <= tolerance) &
                    (abs(df_all["牵伸倍数Ⅲ"] - exact_elongations[2]) <= tolerance)
                ]
            else:
                # Mode 1: Filter max (nếu có)
                if max_elongation is not None:
                    df_all = df_all[
                        (df_all["牵伸倍数Ⅰ"] <= max_elongation) &
                        (df_all["牵伸倍数Ⅱ"] <= max_elongation) &
                        (df_all["牵伸倍数Ⅲ"] <= max_elongation)
                    ]

                # Mode 2: Filter fixed (nếu có)
                if fixed_elongation is not None:
                    df_all = df_all[
                        (abs(df_all["牵伸倍数Ⅰ"] - fixed_elongation) <= tolerance) |
                        (abs(df_all["牵伸倍数Ⅱ"] - fixed_elongation) <= tolerance) |
                        (abs(df_all["牵伸倍数Ⅲ"] - fixed_elongation) <= tolerance)
                    ]

            log.append(f"📉 Lọc kéo dãn: {original_count} → {len(df_all)} dòng")

            if df_all.empty:
                return "\n".join(log + ["❌ Không có dữ liệu nào thỏa mãn điều kiện kéo dãn."]), "", "", [], 0

        num_units = int(num_units) if num_units else None
        adjusted_ratios, color_percent, adjust_log, max_color_initial, max_white_color, total_after, excluded_colors, no_split_major, max_color = adjust_ratios(ratios, num_units)
        log.extend(adjust_log)
        log.append(f"🔄 Tỉ lệ sau điều chỉnh: {adjusted_ratios}")
        log.append(f"🔄 Màu dư thừa + sơ bộ: {excluded_colors}")

        results = []
        total_before_filter = 0

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
                total_before_filter += 1
                # Kiểm tra điều kiện lọc sắp cúi
                if check_arrangement_filter(res, arrangement_filters):
                    results.append(res)

        log.append(f"📈 Tìm thấy {total_before_filter} kết quả phù hợp tỷ lệ màu")
        if arrangement_filters:
            log.append(f"🎯 Sau lọc sắp cúi: còn {len(results)} kết quả")

        results = sorted(results, key=lambda x: (x["Sai số ưu tiên"], x["Sai số"]))

        if not results:
            if arrangement_filters:
                return "\n".join(log + ["❌ Không tìm thấy kết quả nào thỏa mãn điều kiện sắp cúi."]), "", "", [], 0
            else:
                return "\n".join(log + ["❌ Không tìm thấy kết quả phù hợp."]), "", "", [], 0

        first_page_table = render_result_table(results, 0)
        return "\n".join(log), "", first_page_table, results, 0

    except Exception as e:
        return f"⚠️ Lỗi: {str(e)}", "", "", [], 0

# ======= GIAO DIỆN GRADIO =========
def get_three_stretch_app():
    with gr.Blocks() as app:
        gr.Markdown("<h2 style='text-align: center;'>🎨 Tra cứu tỷ lệ màu</h2>")
        with gr.Row():
            with gr.Column(scale=1):
                color_names_input = gr.Textbox(
                        lines=4,
                        label="🎨 Tên màu",
                        placeholder="G004\nG024\nXX",
                        scale=1)
                num_units_input = gr.Textbox(label="🔹 Số cúi muốn tách (2–6, tùy chọn)", placeholder="VD: 3")
                elongation_limit_input = gr.Textbox(
                    label="🧪 Lọc chỉ số kéo giãn",
                    placeholder="VD: 2.5 hoặc 2,5 hoặc 2.5;1.3 hoặc 1,5;1,3;2,5"
                )
                priority_color_input = gr.Textbox(label="🎯 Màu ưu tiên sai số", placeholder="VD: G004, G024")
                arrangement_filter_input = gr.Textbox(
                    label="🎯 Lọc theo sắp cúi",
                    placeholder="VD: 1G02/1G02/1G01+1SW/2SW/1G02/1SW hoặc A:SW,H:SW ",
                )
            with gr.Column(scale=2):
                color_ratios_input = gr.Textbox(
                        lines=4,
                        label="📊 Tỷ lệ (%) - Dùng dấu . hoặc ,",
                        placeholder="18.0 hoặc 18,0\n40.0 hoặc 40,0\n42.0 hoặc 42,0",
                        scale=1
                    )
                realtime_log = gr.Textbox(label="📥 Tỷ lệ màu đã nhập", lines=6, interactive=False)
                structure_line = gr.Textbox(label="🧱 Cấu trúc tương ứng", interactive=False)
                arrangement_filter_preview = gr.Textbox(label="🎯 Preview sắp cúi", lines=2, interactive=False)
            with gr.Column(scale=3):
                log_output = gr.Textbox(label="📋 Thông tin xử lý", lines=15, interactive=False)
                split_threshold_input = gr.Textbox(label="✂️ Ngưỡng tách màu (VD: 21 hoặc 21,5)", placeholder="VD: 21")
                run_btn = gr.Button("🔍 Tra cứu")

        table_output = gr.Markdown(label="📊 Kết quả")

        results_state = gr.State([])
        current_page = gr.State(0)
        page_size = 10

        # --- Ẩn 2 nút phân trang lúc đầu ---
        with gr.Row(visible=False) as pagination_row:
            prev_btn = gr.Button("⬅️ Trang trước")
            next_btn = gr.Button("➡️ Trang sau")

        # Khi chạy tra cứu -> show pagination nếu có kết quả
        def run_and_toggle(*args):
            log, structure, table, results, page = run_app(*args)
            show_pagination = gr.update(visible=(len(results) > 0))
            return log, structure, table, results, page, show_pagination

        run_btn.click(
            fn=run_and_toggle,
            inputs=[color_names_input, color_ratios_input, num_units_input, elongation_limit_input, priority_color_input, split_threshold_input, arrangement_filter_input],
            outputs=[log_output, structure_line, table_output, results_state, current_page, pagination_row]
        )

        # Khi thay đổi tên màu hoặc tỷ lệ -> preview combined
        def update_preview(*args):
            return preview_combined_ratios(*args)

        color_names_input.change(update_preview, inputs=[color_names_input, color_ratios_input], outputs=realtime_log)
        color_ratios_input.change(update_preview, inputs=[color_names_input, color_ratios_input], outputs=realtime_log)

        num_units_input.change(get_structure_line_from_textbox, inputs=num_units_input, outputs=structure_line)
        arrangement_filter_input.change(preview_arrangement_filters, inputs=arrangement_filter_input, outputs=arrangement_filter_preview)

        prev_btn.click(prev_page, inputs=[results_state, current_page], outputs=[table_output, current_page])
        next_btn.click(next_page, inputs=[results_state, current_page], outputs=[table_output, current_page])

    return app

three_stretch_app = get_three_stretch_app()
__all__ = ["three_stretch_app"]