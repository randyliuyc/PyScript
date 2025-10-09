import pandas as pd
from math import isclose
import gradio as gr
import re
from collections import Counter

# ====== Xử lý từ Gradio ======
EXCEL_PATH = "KD2.0.xlsx"

# ====== Đọc bảng kéo dãn ======
def read_stretch_table(file_path: str):
    sheet = pd.read_excel(file_path, header=None)
    top_start = bot_start = None
    for i, val in enumerate(sheet[0]):
        if val == '并条':
            if top_start is None:
                top_start = i
            else:
                bot_start = i
                break
    top_df = sheet.iloc[top_start+2:bot_start-1, 1:].copy()
    top_df.columns = sheet.iloc[top_start+1, 1:]
    top_df.index = sheet.iloc[top_start+2:bot_start-1, 0]
    bot_df = sheet.iloc[bot_start+2:, 1:].copy()
    bot_df.columns = sheet.iloc[bot_start+1, 1:]
    bot_df.index = sheet.iloc[bot_start+2:, 0]
    return top_df.astype(float), bot_df.astype(float)

# ====== Tìm N phương án tổng sai số nhỏ nhất ======
def find_all_good_matches(input_ratios, top_df, bot_df, max_results=10, max_combo=5):
    sorted_items = sorted(input_ratios.items(), key=lambda x: x[1])
    results = []

    for col in top_df.columns:
        for bot_idx in bot_df.index:
            bot_val = bot_df.at[bot_idx, col]
            top_idx = 8 - bot_idx  # Công thức theo file index của bạn
            if top_idx not in top_df.index:
                continue
            top_val = top_df.at[top_idx, col]

            results.append({
                "Chỉ số kéo dãn": col,
                "Trên": top_idx,
                "Dưới": bot_idx,
                "Giá trị trên": round(top_val, 3),
                "Giá trị dưới": round(bot_val, 3),
            })

    df_results = pd.DataFrame(results)
    if df_results.empty:
        return None

    return df_results

# ====== Sinh tổ hợp tổng i+j = 8 theo đúng tỉ lệ top/bot ======
def generate_fixed_combinations(n_colors, total_top, total_bot):
    combos = []
    total_top = int(total_top)
    total_bot = int(total_bot)

    def backtrack(pos, top_remain, bot_remain, current):
        if pos == n_colors:
            if top_remain == 0 and bot_remain == 0:
                combos.append(current[:])
            return
        for i in range(top_remain + 1):
            for j in range(bot_remain + 1):
                current.extend([i, j])
                backtrack(pos + 1, top_remain - i, bot_remain - j, current)
                current.pop()
                current.pop()

    backtrack(0, total_top, total_bot, [])
    return combos

# ====== Tạo chuỗi phân bổ tỉ lệ sắp cúi (đúng tổng top/bot) ======
def distribute_remaining_ratios(input_ratios, top_df, bot_df, top_row, bot_row, stretch_col):
    top_val = top_df.loc[top_row, stretch_col]
    bot_val = bot_df.loc[bot_row, stretch_col]

    color_names = list(input_ratios.keys())
    target_values = [input_ratios[k] for k in color_names]

    best_error = float("inf")
    best_allocation = None
    best_debug = []

    for combo in generate_fixed_combinations(len(color_names), total_top=int(top_row), total_bot=int(bot_row)):
        # ✅ Nếu chỉ còn 1 màu không trắng → yêu cầu bắt buộc chia ≥ 2 phần
        white_keys = {"W", "SW", "WP", "SWP", "FWP", "WJ", "WPJ", "SWJ", "SWPJ", "FW", "FWJ", "FWPJ", "AO"}
        non_white_colors = [name for name in color_names if name.upper() not in white_keys]

        if len(non_white_colors) == 1:
            idx = color_names.index(non_white_colors[0])
            i = combo[idx * 2]
            j = combo[idx * 2 + 1]
            if (i + j) < 2:
                continue  # Bỏ qua nếu không chia ít nhất 2 phần

        total_error = 0
        debug = []
        for idx in range(len(color_names)):
            i = combo[idx*2]
            j = combo[idx*2 + 1]
            approx_val = i * top_val + j * bot_val
            err = abs(approx_val - target_values[idx])
            debug.append((color_names[idx], i, j, approx_val, target_values[idx], err))
            total_error += err

        if total_error < best_error:
            best_error = total_error
            best_allocation = combo[:]
            best_debug = debug[:]

    if best_allocation is None:
        return "❌ Không tìm thấy phương án phù hợp", float("inf"), []

    top_result = []
    bot_result = []
    for idx, name in enumerate(color_names):
        i = best_allocation[idx*2]
        j = best_allocation[idx*2 + 1]
        top_result.extend([name] * i)
        bot_result.extend([name] * j)

    from collections import Counter
    top_counts = Counter(top_result)
    bot_counts = Counter(bot_result)

    top_str = "+".join(f"{v}{k}" if v > 1 else f"1{k}" for k, v in top_counts.items())
    bot_str = "+".join(f"{v}{k}" if v > 1 else f"1{k}" for k, v in bot_counts.items())

    return f"{top_str}/{bot_str} ({stretch_col})", round(best_error, 4), best_debug


# ====== Preview input kiểm tra định dạng ======
def preview_user_ratios(color_input):
    if not color_input.strip():
        return ""
    lines = color_input.strip().split("\n")
    ratios = {}
    log = ["📥 Tỷ lệ màu người dùng đang nhập:"]
    total = 0
    pattern = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*([\d\.]+)\s*$")

    for line in lines:
        match = pattern.match(line)
        if not match:
            return f"⚠️ Sai định dạng ở dòng: '{line}'. Đúng dạng: Tên: số (ví dụ W: 5.0)"
        k, v = match.groups()
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
        log.append(f"⚠️ Tổng tỉ lệ vượt quá 100% thêm {-missing:.2f}%")
    return "\n".join(log)

# ====== Xử lý input, gọi hàm tính toán ======
def process_ratios_textbox(text, elongation_limit=None):
    try:
        lines = text.strip().splitlines()
        input_ratios = {}
        for line in lines:
            if ":" in line:
                name, val = line.split(":")
                input_ratios[name.strip()] = float(val.strip())
        top_df, bot_df = read_stretch_table(EXCEL_PATH)
        df_candidates = find_all_good_matches(input_ratios, top_df, bot_df, max_results=20)
        if df_candidates is None:
            return "Không tìm thấy phương án phù hợp.", None

        all_results = []
        for _, row in df_candidates.iterrows():
            alloc_str, err, debug = distribute_remaining_ratios(
                input_ratios, top_df, bot_df, row["Trên"], row["Dưới"], row["Chỉ số kéo dãn"])
            all_results.append({
                "Chỉ số kéo dãn": row["Chỉ số kéo dãn"],
                "Trên": row["Trên"],
                "Dưới": row["Dưới"],
                "Giá trị trên": row["Giá trị trên"],
                "Giá trị dưới": row["Giá trị dưới"],
                "Tổng sai số": err,
                "Phân bổ sắp cúi": alloc_str
            })

        df_final = pd.DataFrame(all_results)
        
        # --- Thêm lọc elongation_limit ---
        if elongation_limit is not None and elongation_limit != "":
            try:
                limit = float(elongation_limit)
                # Giả sử bạn muốn lọc dựa trên cột "Chỉ số kéo dãn"
                df_final = df_final[df_final["Chỉ số kéo dãn"].astype(float) <= limit]
                if df_final.empty:
                    return f"❌ Không tìm thấy kết quả có chỉ số kéo dãn ≤ {limit}.", None
            except Exception as e:
                return f"Lỗi khi lọc theo giới hạn kéo dãn: {e}", None
        # ------------------------------

        df_final = df_final.sort_values("Tổng sai số").head(20)
        return None, df_final

    except Exception as e:
        return f"Lỗi: {e}", None


# ====== Giao diện Gradio ======
def get_one_stretch_app():
    with gr.Blocks() as app:
        gr.Markdown("## 🧵 Tính Toán Chỉ Số Kéo Dãn Và Hiển Thị Phương Án Sắp Cúi")
        gr.Markdown("Nhập mỗi dòng theo định dạng `Tên: Giá trị`")
        with gr.Row():
            with gr.Column(scale=1):
                input_box = gr.Textbox(lines=8, label="Nhập Tỉ lệ màu")
                elongation_limit_input = gr.Textbox(label="🧪 Lọc chỉ số kéo giãn tối đa (tuỳ chọn)")
            with gr.Column(scale=2):
                realtime_log = gr.Textbox(label="📥 Tỷ lệ người đã dùng nhập", lines=8, interactive=False)
                btn = gr.Button("Tính toán")
        status_output = gr.Textbox(label="Thông báo lỗi hoặc trạng thái", interactive=False)
        result_table = gr.DataFrame(headers=["Chỉ số kéo dãn", "Trên", "Dưới", "Giá trị trên", "Giá trị dưới", "Tổng sai số", "Phân bổ sắp cúi"])

        btn.click(fn=process_ratios_textbox, inputs=[input_box,elongation_limit_input], outputs=[status_output, result_table])
        input_box.change(fn=preview_user_ratios, inputs=input_box, outputs=realtime_log)
    return app

one_stretch_app = get_one_stretch_app()
__all__ = ["one_stretch_app"]
