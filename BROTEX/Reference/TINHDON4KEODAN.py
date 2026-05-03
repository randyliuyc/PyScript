import gradio as gr
import pandas as pd
import numpy as np
from itertools import product, combinations
import re
import itertools
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

EXCEL_PATH = "merged_ratios_4cols_1.xlsx"
SHEET_NAME = "Sheet1"

white_keys = [
    "W", "SW", "WP", "SWP", "FWP", "WJ", "WPJ", "SWJ", "SWPJ",
    "FW", "FWJ", "FWPJ", "WAO","WC","WB","WUS","WOC","WGEC",
    "WL","WN","WM","WTE","WT"
]

# ===== Cached data loading =====
_cached_df = None

def load_data():
    global _cached_df
    if _cached_df is not None:
        return _cached_df
    
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=0)
        required_cols = ['STT', '牵伸I', '牵伸II', '牵伸III', '牵伸IV', 'A', 'B', 'C', 'D']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"File Excel thiếu các cột: {required_cols}")

        # Vectorized operations
        df = df.dropna(subset=['A','B','C','D','STT'])
        
        # Convert to numeric in batch
        for col in 'ABCD':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['A','B','C','D','STT'])

        # Convert to dict records for faster access
        _cached_df = df.to_dict('records')
        return _cached_df
    except Exception as e:
        print(f"Lỗi load dữ liệu: {e}")
        return []

# ===== Optimized color splitting =====
@lru_cache(maxsize=128)
def split_color_parts_cached(color_tuple, min_split_value=2.6, max_parts_per_color=12, max_part_value=6):
    """Cached version of split_color_parts"""
    color_ratios = dict(color_tuple)
    split_dict = {}
    
    for color, value in color_ratios.items():
        parts = []
        for n in range(1, max_parts_per_color + 1):
            part_val = value / n
            if min_split_value <= part_val < max_part_value:
                parts.append((n, round(part_val, 2)))
            elif part_val < min_split_value:
                break
        if parts:
            split_dict[color] = parts
    return split_dict

# ===== Optimized error calculation =====
@lru_cache(maxsize=1024)
def calculate_error_fast(mapping_str, ratios_str, original_ratios_str):
    """Fast cached error calculation"""
    # Parse strings back to dicts
    mapping_dict = dict(zip('ABCD', mapping_str.split('|')))
    df_ratios = dict(zip('ABCD', map(float, ratios_str.split('|'))))
    original_ratios = dict(item.split(':') for item in original_ratios_str.split('|'))
    original_ratios = {k: float(v) for k, v in original_ratios.items()}
    
    totals = {c: 0.0 for c in original_ratios.keys()}

    for col in "ABCD":
        parts_str = mapping_dict.get(col, "")
        if not parts_str:
            continue
        for part in parts_str.split("+"):
            part = part.strip()
            m = re.match(r"^(\d+)([A-Za-z0-9]+)$", part)
            if not m:
                continue
            num = int(m.group(1))
            color = m.group(2)
            if color in totals:
                totals[color] += df_ratios[col] * num

    err = sum(abs(totals.get(color, 0.0) - float(original_ratios.get(color, 0.0))) 
              for color in original_ratios.keys())
    return round(err, 2)

# ===== Pre-compiled regex patterns =====
PART_PATTERN = re.compile(r"^(\d+)([A-Za-z0-9]+)$")
COLOR_PATTERN = re.compile(r"^\s*([A-Za-z0-9_\-\(\)\u4e00-\u9fff]+)\s*:\s*([\d\.]+)\s*$")

def normalize_mapping_fast(mapping_dict):
    """Fast version with pre-compiled regex"""
    normalized = {}
    for col, parts_str in mapping_dict.items():
        if not parts_str:
            normalized[col] = ""
            continue
            
        counts = {}
        for part in parts_str.split("+"):
            part = part.strip()
            m = PART_PATTERN.match(part)
            if not m:
                continue
            num, color = int(m.group(1)), m.group(2)
            counts[color] = counts.get(color, 0) + num
        
        # Sort by count desc, then by name
        sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        normalized[col] = "+".join(f"{n}{c}" for c, n in sorted_items)
    
    return normalized

# ===== Optimized main matching algorithm =====
def match_colors_optimized_v3(combo, row, tolerance=0.02, priority_colors=None, 
                             largest_color=None, original_ratios=None):
    """Ultra-optimized version with minimal object creation"""
    if priority_colors is None:
        priority_colors = []
    if largest_color is None:
        largest_color = "W"
    if original_ratios is None:
        return None

    # Pre-compute frequently used values
    df_ratios = {c: float(row[c]) for c in "ABCD"}
    colors_data = [(num, val, color) for num, val, color in combo if color != largest_color]
    
    if not colors_data:
        return None
        
    colors_work = [[int(num), float(val), str(color)] for num, val, color in colors_data]
    
    col_assignments = [""] * 4  # Use list instead of dict for speed
    used_units = [0] * 4
    max_units_per_col = 6
    cols = ["A", "B", "C", "D"]
    col_idx = {"A": 0, "B": 1, "C": 2, "D": 3}

    # ===== Step U: Unique assignment (optimized) =====
    k = len(colors_work)
    if k > 0:
        best_perm = None
        best_score = float("inf")
        
        # Only check feasible permutations
        from itertools import permutations
        for cols_choice in permutations(range(4), k):
            if all(used_units[c] < max_units_per_col for c in cols_choice):
                # Calculate score without creating objects
                score = sum(abs(df_ratios[cols[cols_choice[i]]] - colors_work[i][1]) 
                           for i in range(k))
                
                if score < best_score:
                    best_score = score
                    best_perm = cols_choice

        # Apply best assignment
        if best_perm is not None:
            for i in range(k):
                col_i = best_perm[i]
                num, val, color = colors_work[i]
                assign = 1 if num >= 1 else 0
                if assign > 0:
                    if col_assignments[col_i]:
                        col_assignments[col_i] += f"+{assign}{color}"
                    else:
                        col_assignments[col_i] = f"{assign}{color}"
                    used_units[col_i] += assign
                    colors_work[i][0] -= assign

    # ===== Step G: Greedy fill (optimized) =====
    # Sort by remaining units (descending)
    colors_work.sort(key=lambda x: x[0], reverse=True)
    
    for color_entry in colors_work:
        num_parts, val, color = color_entry
        if num_parts <= 0:
            continue
            
        while num_parts > 0:
            # Find available columns
            available_cols = [i for i in range(4) if used_units[i] < max_units_per_col]
            if not available_cols:
                break
                
            # Calculate differences for all available columns at once
            diffs = [(abs(df_ratios[cols[i]] - val), i) for i in available_cols]
            min_diff = min(diffs)[0]
            
            # Find columns within tolerance
            same_val_cols = [i for diff, i in diffs if diff <= min_diff + tolerance]
            
            # Apply concentration heuristic
            if len(same_val_cols) >= 2:
                def concentration_score(col_i):
                    parts = col_assignments[col_i]
                    base_diff = next(diff for diff, i in diffs if i == col_i)
                    
                    if not parts:
                        return (0, base_diff, 0)
                    if color in parts:
                        consolidation_bonus = -0.5 if num_parts >= 2 else -0.1
                        return (0, base_diff + consolidation_bonus, 1)
                    return (2, base_diff, 3)
                
                target_col = min(same_val_cols, key=concentration_score)
            else:
                target_col = min(available_cols, key=lambda i: diffs[available_cols.index(i)][0])
                
                # Check for existing color consolidation
                current_cols_with_color = [i for i in available_cols if color in col_assignments[i]]
                if current_cols_with_color and num_parts >= 2:
                    existing_col = current_cols_with_color[0]
                    diff_existing = next(diff for diff, i in diffs if i == existing_col)
                    diff_best = next(diff for diff, i in diffs if i == target_col)
                    
                    if diff_existing - diff_best <= 0.4:
                        target_col = existing_col

            # Determine assignment size
            remain_capacity = max_units_per_col - used_units[target_col]
            
            if color in col_assignments[target_col] and num_parts >= 2:
                assign_units = min(num_parts, remain_capacity, 3)
            else:
                assign_units = min(num_parts, remain_capacity, 1)
            
            # Apply assignment
            if col_assignments[target_col]:
                col_assignments[target_col] += f"+{assign_units}{color}"
            else:
                col_assignments[target_col] = f"{assign_units}{color}"
            used_units[target_col] += assign_units
            num_parts -= assign_units

    # Convert back to dict format
    col_assignments_dict = {cols[i]: col_assignments[i] for i in range(4)}
    
    # Apply remaining steps (optimized versions)
    final_mapping = assign_remaining_colors_fast(col_assignments_dict, df_ratios, original_ratios, largest_color)
    final_mapping = normalize_mapping_fast(final_mapping)
    
    # Fast error calculation using string caching
    mapping_str = '|'.join(final_mapping.get(c, '') for c in 'ABCD')
    ratios_str = '|'.join(str(df_ratios[c]) for c in 'ABCD')
    original_ratios_str = '|'.join(f"{k}:{v}" for k, v in original_ratios.items())
    
    calc_err = calculate_error_fast(mapping_str, ratios_str, original_ratios_str)

    # Calculate priority error
    priority_error = 0.0
    for num, val, color in colors_data:
        if color in priority_colors:
            for col in "ABCD":
                if color in final_mapping.get(col, ""):
                    priority_error += abs(df_ratios[col] - val)
                    break

    # Format output
    mapping_colors = [final_mapping.get(c, "") for c in "ABCD"]
    stretch_cols = ["牵伸I", "牵伸II", "牵伸III", "牵伸IV"]
    stretch_vals = [row.get(col, "") for col in stretch_cols]
    stretch_str = "(" + "/".join(str(v) for v in stretch_vals if v != "") + ")"
    mapping_str = "/".join(mapping_colors) + " " + stretch_str

    return {
        "Row": row.get("STT", ""),
        "Mapping": mapping_str,
        "Sai số": calc_err,
        "Sai số trắng": 0,
        "Sai số màu": 0,
        "Sai số ưu tiên": round(priority_error, 2),
        "MappingDict": final_mapping,
        "Ratios": df_ratios
    }

def assign_remaining_colors_fast(col_assignments, df_ratios, original_ratios, largest_color):
    """Optimized version of assign_remaining_colors"""
    # Calculate current usage
    used = {}
    for col in ["A", "B", "C", "D"]:
        parts = col_assignments.get(col, "")
        used_units = sum(int(PART_PATTERN.match(p.strip()).group(1)) 
                        for p in parts.split("+") if parts and PART_PATTERN.match(p.strip()))
        used[col] = used_units

    # Sort large colors by value (descending)
    large_list = [(c, v) for c, v in original_ratios.items() if c != largest_color]
    large_list.sort(key=lambda x: x[1], reverse=True)

    # Assign each large color optimally
    for color, target_amount in large_list:
        for _ in range(12):  # Max attempts
            # Calculate current total for this color
            current_total = 0.0
            for col in ["A", "B", "C", "D"]:
                parts = col_assignments.get(col, "")
                for p in parts.split("+") if parts else []:
                    m = PART_PATTERN.match(p.strip())
                    if m and m.group(2) == color:
                        current_total += df_ratios[col] * int(m.group(1))

            # Find best column to add 1 unit
            best_col = None
            best_improve = -1e9
            
            for col in ["A", "B", "C", "D"]:
                if used[col] >= 6:
                    continue
                    
                new_total = current_total + df_ratios[col]
                old_err = abs(current_total - target_amount)
                new_err = abs(new_total - target_amount)
                improvement = old_err - new_err
                
                if improvement > best_improve:
                    best_improve = improvement
                    best_col = col

            if best_col is None or best_improve <= 0:
                break

            # Apply assignment
            prev = col_assignments.get(best_col, "")
            col_assignments[best_col] = (prev + "+" if prev else "") + f"1{color}"
            used[best_col] += 1

    # Fill remaining space with largest_color
    for col in ["A", "B", "C", "D"]:
        remain = 6 - used[col]
        if remain > 0:
            prev = col_assignments.get(col, "")
            col_assignments[col] = (prev + "+" if prev else "") + f"{remain}{largest_color}"

    return col_assignments

# ===== Optimized combo generation =====
def generate_combos_fast(other_colors, max_combos=1000):
    """Fast combo generation with early stopping"""
    # Convert to tuple for caching
    color_tuple = tuple(sorted(other_colors.items()))
    split_dict = split_color_parts_cached(color_tuple)
    
    if not split_dict:
        return []

    # Generate combos with limit
    all_color_parts = [[(num, val, color) for num, val in parts] 
                      for color, parts in split_dict.items()]
    
    combos = []
    for combo in product(*all_color_parts):
        if len(combos) >= max_combos:
            break
        if len(combo) <= 6:
            combos.append(list(combo))
    
    return combos

# ===== UI Helper Functions (optimized) =====
def preview_user_ratios(color_input):
    if not color_input.strip():
        return ""

    lines = color_input.strip().split("\n")
    ratios = {}
    log = ["📥 Tỷ lệ màu người dùng:"]
    total = 0

    for line in lines:
        match = COLOR_PATTERN.match(line)
        if not match:
            return f"⚠️ Sai định dạng: '{line}'. VD đúng: W: 5.0"
        k, v = match.groups()
        k, val = k.strip().upper(), float(v)
        ratios[k] = val
        log.append(f"- {k}: {val:.2f}%")
        total += val

    if ratios:
        max_color = max(ratios, key=ratios.get)
        max_ratio = ratios[max_color]
        log.append(f"🎯 Màu lớn nhất: {max_color} ({max_ratio:.2f}%)")

        other_colors = {k: v for k, v in ratios.items() if k != max_color}
        if other_colors:
            log.append(f"🔹 Màu combo: {list(other_colors.keys())} ({len(other_colors)} màu)")

    log.append(f"📊 Tổng: {total:.2f}%")
    if abs(total - 100) > 0.1:
        log.append(f"⚠️ Chênh lệch: {100-total:+.2f}%")

    return "\n".join(log)

def combine_color_inputs(color_names, color_ratios):
    """Fast input combination"""
    if not color_names.strip() or not color_ratios.strip():
        return ""
    
    name_lines = [line.strip() for line in color_names.strip().split("\n") if line.strip()]
    ratio_lines = [line.strip() for line in color_ratios.strip().split("\n") if line.strip()]
    
    if len(name_lines) != len(ratio_lines):
        return f"⚠️ Số lượng tên màu ({len(name_lines)}) khác với số tỷ lệ ({len(ratio_lines)})"
    
    return "\n".join(f"{name}: {ratio}" for name, ratio in zip(name_lines, ratio_lines))

def preview_combined_ratios(color_names, color_ratios):
    """Preview tỷ lệ màu sau khi combine"""
    combined_input = combine_color_inputs(color_names, color_ratios)
    if not combined_input or combined_input.startswith("⚠️"):
        return combined_input
    
    return preview_user_ratios(combined_input)

# ===== Optimized result rendering =====
def render_result_table(results, page, page_size=10):
    """Fast table rendering with vectorized operations"""
    start = page * page_size
    page_results = results[start:start + page_size]
    if not page_results:
        return "⚠️ Không có kết quả"

    # Pre-allocate data structure
    data = []
    for i, r in enumerate(page_results, start=start + 1):
        ratios = r.get("Ratios", {})
        mapping_dict = r.get("MappingDict", {})
        
        row_info = {
            "STT": i,
            "Row": r.get("Row", ""),
            "Sai số": r.get("Sai số", ""),
            "Sai số ƯT": r.get("Sai số ưu tiên", ""),
            "Sắp cúi": r.get("Mapping", "")
        }
        
        # Batch process ABCD columns
        for c in "ABCD":
            left_val = ratios.get(c, 0.0)
            right_val = mapping_dict.get(c, "")
            row_info[c] = f"{left_val:.2f} → {right_val}"
        
        data.append(row_info)

    df = pd.DataFrame(data)[["STT", "Row"] + list("ABCD") + ["Sai số", "Sai số ƯT", "Sắp cúi"]]
    return df.to_markdown(index=False)

def prev_page(results, current, page_size=10):
    new_page = max(0, current-1)
    return render_result_table(results, new_page, page_size), new_page

def next_page(results, current, page_size=10):
    max_page = len(results) // page_size
    new_page = min(max_page, current+1)
    return render_result_table(results, new_page, page_size), new_page

# ===== Optimized main app logic =====
def run_app(color_names_input, color_ratios_input, elongation_limit, priority_input, page_size=10):
    try:
        # Fast input processing
        combined = combine_color_inputs(color_names_input or "", color_ratios_input or "")
        if combined.startswith("⚠️"):
            return combined, "⚠️ Tỉ lệ sai", [], 0

        # Parse ratios efficiently
        ratios = {}
        for line in combined.strip().split("\n"):
            m = COLOR_PATTERN.match(line)
            if not m:
                return f"⚠️ Sai định dạng: '{line}'", "⚠️ Tỉ lệ sai", [], 0
            k, v = m.groups()
            ratios[k.strip().upper()] = float(v)

        # Validate total
        total_ratio = sum(ratios.values())
        if abs(total_ratio - 100) > 0.001:
            return f"⚠️ Tổng tỉ lệ = {total_ratio}% ≠ 100%", "⚠️ Tỉ lệ sai", [], 0

        # Get largest color and others
        max_color = max(ratios, key=ratios.get)
        other_colors = {k: v for k, v in ratios.items() if k != max_color}

        log = [f"🎨 Màu lớn nhất: {max_color} ({ratios[max_color]:.1f}%)"]

        # Validate minimum ratios
        min_ratio = 2.6
        invalid = [c for c, r in other_colors.items() if r < min_ratio]
        if invalid:
            return f"⚠️ Màu < {min_ratio}%: {invalid}", "⚠️ Tỉ lệ không hợp lệ", [], 0

        # Load data (cached)
        df_all = load_data()
        if not df_all:
            return "⚠️ Không có dữ liệu", "⚠️ Lỗi data", [], 0

        priority_colors = [s.strip().upper() for s in (priority_input or "").split(",") if s.strip()]

        # Generate combos efficiently
        all_combos = generate_combos_fast(other_colors, max_combos=1000)
        if not all_combos:
            log.append("⚠️ Không tạo được combo")
            return "\n".join(log), "⚠️ Không có combo", [], 0

        log.append(f"🔢 Tổ hợp: {len(all_combos)} (màu combo: {len(other_colors)})")

        # Filter by elongation (vectorized)
        if elongation_limit:
            try:
                elong_val = float(elongation_limit)
                df_all = [row for row in df_all 
                         if (row["牵伸I"] <= elong_val and row["牵伸II"] <= elong_val and
                             row["牵伸III"] <= elong_val and row["牵伸IV"] <= elong_val)]
                log.append(f"🔹 Lọc kéo dài ≤ {elong_val}: {len(df_all)} dòng")
            except:
                pass

        # Parallel-style processing (batch matching)
        all_results = []
        matched_combos = 0

        for i, combo in enumerate(all_combos, 1):
            combo_results = []
            
            # Process rows in batches for this combo
            for row in df_all:
                res = match_colors_optimized_v3(
                    combo, row,
                    tolerance=0.5,
                    priority_colors=priority_colors,
                    largest_color=max_color,
                    original_ratios=ratios
                )
                if res:
                    res["Combination"] = i
                    combo_results.append(res)
            
            if combo_results:
                all_results.extend(combo_results)
                matched_combos += 1

        log.append(f"📊 Kết quả: {matched_combos}/{len(all_combos)} combo, {len(all_results)} matches")

        if not all_results:
            return "\n".join(log), "⚠️ Không có kết quả phù hợp", [], 0

        # Fast sorting
        all_results.sort(key=lambda x: (x.get("Sai số ưu tiên", 0), x.get("Sai số", 0)))

        return "\n".join(log), render_result_table(all_results, 0, page_size), all_results, 0

    except Exception as e:
        return f"❌ Lỗi: {str(e)}", "⚠️ Có lỗi xảy ra", [], 0

# ===== Gradio Interface =====
def get_four_stretch_app():
    with gr.Blocks() as app:
        gr.Markdown("<h2 style='text-align: center;'>🎨 Tra cứu 4 chỉ số kéo dài </h2>")
        with gr.Row():
            with gr.Column(scale=1):
                color_names_input = gr.Textbox(
                        lines=6, 
                        label="🎨 Tên màu", 
                        placeholder="G004\nG024\nXX",
                        scale=1)
                elongation_input = gr.Textbox(label="🧪 Lọc chỉ số kéo giãn (VD: 2.5)")
                priority_input = gr.Textbox(label="🎯 Màu ưu tiên sai số", placeholder="B014, B020")
            with gr.Column(scale=2):
                color_ratios_input = gr.Textbox(
                        lines=4, 
                        label="📊 Tỷ lệ (%)", 
                        placeholder="18.0\n40.0\n42.0",
                        scale=1
                    )
                realtime_log = gr.Textbox(label="📥 Xem trước tỷ lệ", lines=10, interactive=False)
            with gr.Column(scale=3):
                log_output = gr.Textbox(label="📋 Kết quả xử lý", lines=15, interactive=False)
                run_btn = gr.Button("🔍 Tra cứu")

        table_output = gr.Markdown(label="📊 Bảng kết quả")
        results_state = gr.State([])
        current_page = gr.State(0)
        page_size = 10

        # Events
        run_btn.click(
            fn=run_app,
            inputs=[color_names_input,color_ratios_input, elongation_input, priority_input, gr.State(page_size)],
            outputs=[log_output, table_output, results_state, current_page]
        )

        def update_preview(*args):
            return preview_combined_ratios(*args)
        
        color_names_input.change(update_preview, inputs=[color_names_input, color_ratios_input], outputs=realtime_log)
        color_ratios_input.change(update_preview, inputs=[color_names_input, color_ratios_input], outputs=realtime_log)

        with gr.Row():
            prev_btn = gr.Button("⬅️ Trang trước")
            next_btn = gr.Button("➡️ Trang sau")
        prev_btn.click(fn=prev_page, inputs=[results_state, current_page], outputs=[table_output, current_page])
        next_btn.click(fn=next_page, inputs=[results_state, current_page], outputs=[table_output, current_page])

    return app

four_stretch_app = get_four_stretch_app()
__all__ = ["four_stretch_app"]