import gradio as gr
import pandas as pd
import numpy as np
from itertools import combinations, product
from math import ceil
import os
import re
from collections import defaultdict
import duckdb
from functools import lru_cache
from scipy.optimize import differential_evolution, minimize
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import random
import time
import warnings
warnings.filterwarnings('ignore')

# ======= CẤU HÌNH =========
EXCEL_PATH = "Data4kd_ratio_6.xlsx"
SHEET_NAME = "Sheet1"
DUCKDB_PATH = "color_data_6.duckdb"
white_keys = ["W", "SW", "WP", "SWP", "FWP", "WJ", "WPJ", "SWJ", "SWPJ", "FW", "FWJ", "FWPJ", "WAO","WC","WB","WUS","WOC","WGEC","WL","WN","WM","WTE","WT"]

# ===== Hàm làm tròn chuẩn Excel =====
def excel_round(value, digits=2):
    """Làm tròn theo quy tắc ROUND_HALF_UP của Excel"""
    return float(Decimal(str(value)).quantize(Decimal('1.' + '0'*digits), rounding=ROUND_HALF_UP))

# ===== Hàm tính tỷ lệ A–H =====
@lru_cache(maxsize=100000)
def calculate_ratios_from_stretches(e1, e2, e3, e4):
    """
    Tính tỷ lệ A–H theo công thức Excel:
    A = 1/e1 / (1/e1 + 1/e4 + 2 + 2/e2 + 1/e4 + 1/e3)
    và làm tròn từng bước như Excel, hiển thị 2 chữ số thập phân.
    """
    try:
        e1 = Decimal(str(e1))
        e2 = Decimal(str(e2))
        e3 = Decimal(str(e3))
        e4 = Decimal(str(e4))

        inv_e1 = Decimal('1') / e1
        inv_e2 = Decimal('1') / e2
        inv_e3 = Decimal('1') / e3
        inv_e4 = Decimal('1') / e4

        denominator = inv_e1 + inv_e4 + Decimal('2') + (Decimal('2') * inv_e2) + inv_e4 + inv_e3

        A = (inv_e1 / denominator) * 100
        B = (inv_e4 / denominator) * 100
        C = (Decimal('1') / denominator) * 100
        D = (Decimal('1') / denominator) * 100
        E = (inv_e2 / denominator) * 100
        F = (inv_e2 / denominator) * 100
        G = (inv_e4 / denominator) * 100
        H = (inv_e3 / denominator) * 100

        ratios = {
            'A': excel_round(A, 2),
            'B': excel_round(B, 2),
            'C': excel_round(C, 2),
            'D': excel_round(D, 2),
            'E': excel_round(E, 2),
            'F': excel_round(F, 2),
            'G': excel_round(G, 2),
            'H': excel_round(H, 2)
        }

        total = sum(ratios.values())
        ratios['Total'] = excel_round(total, 2)

        return ratios

    except (ZeroDivisionError, ValueError, InvalidOperation):
        return None

def quick_filter_stretches(e1, e2, e3, e4, target_sum=100):
    """Lọc nhanh các tổ hợp không khả thi - TỐI ƯU HƠN"""
    if e1 > 4.0 or e2 > 4.0 or e3 > 4.0 or e4 > 6.0:
        return False

    if e4 <= e1 or e4 <= e3:
        return False
    
    # ← THÊM ĐIỀU KIỆN MỚI
    if e4 / e1 >= 4.0 or e4 / e3 >= 4.0:
        return False

    if any(e < 1.1 for e in [e1, e2, e3, e4]):
        return False

    try:
        inv_sum = 1/e1 + 2/e4 + 2 + 2/e2 + 1/e3
        if inv_sum < 0.5 or inv_sum > 10:
            return False
    except:
        return False

    return True

# ======= KHỞI TẠO DATABASE =========
def init_database():
    """Khởi tạo DuckDB database từ Excel nếu chưa có"""
    if os.path.exists(DUCKDB_PATH):
        try:
            con = duckdb.connect(DUCKDB_PATH, read_only=True)
            row_count = con.execute("SELECT COUNT(*) FROM color_data").fetchone()[0]
            con.close()
            print(f"✅ Database đã tồn tại với {row_count:,} dòng: {DUCKDB_PATH}")
            return
        except:
            print("⚠️ Database bị lỗi, đang xóa và tạo lại...")
            os.remove(DUCKDB_PATH)

    print("🔄 Đang tạo database từ Excel...")
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME).iloc[1:]

    df = df.rename(columns={
        '牵伸倍数1': 'stretch1',
        '牵伸倍数2': 'stretch2',
        '牵伸倍数3': 'stretch3',
        '牵伸倍数4': 'stretch4'
    })

    df = df[(df['stretch1'] >= 1.1) & (df['stretch2'] >= 1.1) &
            (df['stretch3'] >= 1.1) & (df['stretch4'] >= 1.1)]

    for col in 'ABCDEFGH':
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].max() <= 1:
            df[col] *= 100

    df[list('ABCDEFGH')] = df[list('ABCDEFGH')].round(3)
    df['stretch1'] = df['stretch1'].round(2)
    df['stretch2'] = df['stretch2'].round(2)
    df['stretch3'] = df['stretch3'].round(2)
    df['stretch4'] = df['stretch4'].round(2)

    df = df.dropna(subset=list('ABCDEFGH'))
    df = df.reset_index(drop=True)
    df['row_id'] = df.index

    con = duckdb.connect(DUCKDB_PATH)
    try:
        con.execute("CREATE TABLE color_data AS SELECT * FROM df")
        row_count = con.execute("SELECT COUNT(*) FROM color_data").fetchone()[0]
        print(f"✅ Database đã được tạo với {row_count:,} dòng dữ liệu")
    finally:
        con.close()

def query_data(min_val=None, max_val=None, exact_vals=None, stretch_filters=None):
    """Query dữ liệu từ DuckDB với filter kéo dãn"""
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    base_query = "SELECT * FROM color_data"
    conditions = []
    tolerance = 0.01

    conditions.append("stretch4 / stretch1 < 4.0")
    conditions.append("stretch4 / stretch3 < 4.0")

    # ← XỬ LÝ STRETCH_FILTERS
    if stretch_filters:
        stretch_mapping = {
            'E1': 'stretch1',
            'E2': 'stretch2',
            'E3': 'stretch3',
            'E4': 'stretch4'
        }
        for key, value in stretch_filters.items():
            col_name = stretch_mapping.get(key)
            if col_name:
                conditions.append(f"ABS({col_name} - {value}) <= {tolerance}")

    # XỬ LÝ EXACT_VALS, MIN_VAL, MAX_VAL (giữ nguyên)
    if exact_vals is not None:
        conditions.append(f"""
            (ABS(stretch1 - {exact_vals[0]}) <= {tolerance} AND
             ABS(stretch2 - {exact_vals[1]}) <= {tolerance} AND
             ABS(stretch3 - {exact_vals[2]}) <= {tolerance} )
        """)
    elif min_val is not None and max_val is not None:
        conditions.append(f"""
            (stretch1 BETWEEN {min_val} AND {max_val} AND
             stretch2 BETWEEN {min_val} AND {max_val} AND
             stretch3 BETWEEN {min_val} AND {max_val})
        """)
    elif max_val is not None:
        conditions.append(f"""
            (stretch1 <= {max_val} AND
             stretch2 <= {max_val} AND
             stretch3 <= {max_val})
        """)

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    df = con.execute(base_query).df()
    con.close()
    return df

# ======= TÍNH TOÁN ĐỘNG - TÌM KIẾM TỐI ƯU =========
def find_optimal_stretches_scipy(target_ratios, adjusted_full_ratios, excluded_colors,
                                  stretch_bounds=(1.1, 5.0), method='differential_evolution',
                                  priority_colors=None):
    """
    Tìm kiếm tối ưu bằng scipy optimization
    Điều kiện bắt buộc:
    - E4 > E1 và E4 > E3
    - E1, E2, E3 ≤ 4.0
    - E4 ≤ 6.0
    - Lấy TẤT CẢ kết quả có sai số < 1.5 - TỐI ƯU TỐC ĐỘ
    """
    ERROR_THRESHOLD = 1.5

    calc_cache = {}

    def objective_function(stretches):
        e1, e2, e3, e4 = stretches
        if e1 > 4.0 or e2 > 4.0 or e3 > 4.0:
            return 10000.0
        if e4 > 6.0:
            return 10000.0
        if e4 <= e1 or e4 <= e3:
            return 10000.0
        
        # ← THÊM ĐIỀU KIỆN MỚI
        if e4 / e1 >= 4.0 and e4 / e3 >= 4.0:
            return 10000.0

        e1, e2, e3, e4 = round(e1, 3), round(e2, 3), round(e3, 3), round(e4, 3)
        stretch_key = (e1, e2, e3, e4)

        if stretch_key in calc_cache:
            cached_result = calc_cache[stretch_key]
            if cached_result is None:
                return 1000.0
            return cached_result

        if not quick_filter_stretches(e1, e2, e3, e4):
            calc_cache[stretch_key] = None
            return 1000.0

        calc_ratios = calculate_ratios_from_stretches(e1, e2, e3, e4)
        if calc_ratios is None:
            calc_cache[stretch_key] = None
            return 1000.0

        match_result = match_colors_to_calculated_ratios(
            target_ratios, calc_ratios, tolerance=2.0,
            excluded_colors=excluded_colors,
            adjusted_full_ratios=adjusted_full_ratios,
            priority_colors=priority_colors
        )

        if match_result is None:
            calc_cache[stretch_key] = None
            return 1000.0

        error = match_result['total_error']
        calc_cache[stretch_key] = error
        return error

    min_bound = stretch_bounds[0]
    bounds = [
        (min_bound, min(4.0, stretch_bounds[1])),
        (min_bound, min(4.0, stretch_bounds[1])),
        (min_bound, min(4.0, stretch_bounds[1])),
        (1.1, 6.0)     # E4 cũng chỉ chạy trong khoảng này
    ]

    all_results = []
    found_count = 0

    # Tăng số lần chạy và random vị trí xuất phát
    for seed_val in range(5000):  # Tăng lên 300 lần
        if  seed_val > 0:
            print(f"   Optimization: {seed_val}/5000 lần | Tìm được {found_count} kết quả | Cache: {len(calc_cache)}")

        # Random vị trí xuất phát để đa dạng kết quả
        x0 = [
            round(random.uniform(min_bound, min(4.0, stretch_bounds[1])), 3),
            round(random.uniform(min_bound, min(4.0, stretch_bounds[1])), 3),
            round(random.uniform(min_bound, min(4.0, stretch_bounds[1])), 3),
            round(random.uniform(min_bound, min(6.0, stretch_bounds[1])), 3)
        ]

        result = minimize(
            objective_function,
            x0,
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': 500}
        )

        if result.fun < ERROR_THRESHOLD:
            e1, e2, e3, e4 = [round(x, 3) for x in result.x]
            if (e4 > e1 and e4 > e3 and
                e1 <= 4.0 and e2 <= 4.0 and e3 <= 4.0 and e4 <= 6.0 and
                e4 / e1 < 4.0 and e4 / e3 < 4.0):  # ← THÊM ĐIỀU KIỆN
                all_results.append((e1, e2, e3, e4, result.fun))
                found_count += 1

    # Cho phép các tổ hợp gần nhau cùng xuất hiện (không loại trùng quá gắt)
    unique_results = []
    seen = set()
    for r in sorted(all_results, key=lambda x: x[4]):
        key = (r[0], r[1], r[2], r[3])  # Không làm tròn, không loại trùng sát nhau
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    print(f"✅ Optimization: {len(unique_results)} unique results (error < {ERROR_THRESHOLD}) | Cache hits: {len(calc_cache)}")
    return unique_results

def find_stretches_grid_search(target_ratios, adjusted_full_ratios, excluded_colors,
                                stretch_range=None, max_combinations=100000, priority_colors=None):
    """
    Tìm kiếm dạng lưới (grid search) với sampling
    E1, E2, E3 ≤ 4.0, E4 ≤ 6.0
    Lấy TẤT CẢ kết quả có sai số < 1.5 - TỐI ƯU TỐC ĐỘ
    """
    ERROR_THRESHOLD = 1.5

    if stretch_range is None:
        stretch_range_123 = np.arange(1.1, 4.1, 0.01).round(2)
        stretch_range_4 = np.arange(1.1, 6.1, 0.01).round(2)
    else:
        stretch_range_123 = [x for x in stretch_range if x <= 4.0]
        stretch_range_4 = [x for x in stretch_range if x <= 6.0]

    print(f"🔍 Grid search: E1,E2,E3={len(stretch_range_123)} giá trị (≤4.0), E4={len(stretch_range_4)} giá trị (≤6.0)")

    total_combinations = len(stretch_range_123) ** 3 * len(stretch_range_4)

    if total_combinations > max_combinations:
        print(f"⚠️ Lấy mẫu {max_combinations:,} tổ hợp ngẫu nhiên")
        np.random.seed(42)
        samples = []
        for _ in range(max_combinations):
            e1 = np.random.choice(stretch_range_123)
            e2 = np.random.choice(stretch_range_123)
            e3 = np.random.choice(stretch_range_123)
            e4 = np.random.choice(stretch_range_4)
            if e4 > e1 and e4 > e3:
                samples.append((e1, e2, e3, e4))
        print(f"📊 Sau lọc: {len(samples):,} tổ hợp hợp lệ")
    else:
        print(f"📊 Kiểm tra {total_combinations:,} tổ hợp")
        samples = [(e1, e2, e3, e4)
                   for e1 in stretch_range_123
                   for e2 in stretch_range_123
                   for e3 in stretch_range_123
                   for e4 in stretch_range_4
                   if e4 > e1 and e4 > e3]
        print(f"📊 Sau lọc điều kiện E4: {len(samples):,} tổ hợp")

    results = []
    checked = 0
    skipped_quick = 0

    calc_cache = {}

    for i, (e1, e2, e3, e4) in enumerate(samples):
        if i % 10000 == 0 and i > 0:
            hit_rate = (len(results) / checked * 100) if checked > 0 else 0
            print(f"   Đã xử lý {i:,}/{len(samples):,} | Tìm được {len(results)} | Hit rate: {hit_rate:.1f}% | Skipped: {skipped_quick:,}")

        if not quick_filter_stretches(e1, e2, e3, e4):
            skipped_quick += 1
            continue

        checked += 1

        stretch_key = (e1, e2, e3, e4)
        if stretch_key not in calc_cache:
            calc_ratios = calculate_ratios_from_stretches(e1, e2, e3, e4)
            if calc_ratios is None:
                calc_cache[stretch_key] = None
                continue
            calc_cache[stretch_key] = calc_ratios
        else:
            calc_ratios = calc_cache[stretch_key]
            if calc_ratios is None:
                continue

        match_result = match_colors_to_calculated_ratios(
            target_ratios, calc_ratios, tolerance=2.0,
            excluded_colors=excluded_colors,
            adjusted_full_ratios=adjusted_full_ratios,
            priority_colors=priority_colors
        )

        if match_result and match_result['total_error'] < ERROR_THRESHOLD:
            results.append((e1, e2, e3, e4, match_result['total_error']))

    print(f"✅ Grid search: Checked {checked:,} | Found {len(results)} (error < {ERROR_THRESHOLD}) | Skipped {skipped_quick:,}")

    unique_results = []
    seen = set()
    for r in sorted(results, key=lambda x: x[4]):
        key = (round(r[0], 2), round(r[1], 2), round(r[2], 2), round(r[3], 2))
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    print(f"✅ Unique results: {len(unique_results)}")
    return sorted(unique_results, key=lambda x: x[4])[:500]

def parse_elongation_filter(elongation_input):
    """
    Parse input kéo dãn - HỖ TRỢ NHIỀU FORMAT:
    - "2.5" → E1,E2,E3 ≤ 2.5
    - "1.5,3.0" → 1.5 ≤ E1,E2,E3 ≤ 3.0
    - "exact:1.5,1.3,2.5" → E1=1.5, E2=1.3, E3=2.5
    - "E1=1.5, E3=3.0" → E1=1.5 và E3=3.0 (hỗ trợ cả ':' và '=')
    
    Returns: (min_val, max_val, exact_vals, stretch_filters, log_msg)
    """
    if not elongation_input.strip():
        return None, None, None, {}, ""

    elongation_input = elongation_input.strip()
    min_val = None
    max_val = None
    exact_vals = None
    stretch_filters = {}

    try:
        # ← FORMAT MỚI: E1=1.5, E3=3.0 HOẶC E1:1.5, E3:3.0
        if 'E' in elongation_input.upper() and ('=' in elongation_input or ':' in elongation_input):
            parts = [p.strip() for p in elongation_input.split(',') if p.strip()]
            parsed_count = 0
            for part in parts:
                # Hỗ trợ cả ':' và '='
                if ':' in part:
                    key, value = part.split(':', 1)
                elif '=' in part:
                    key, value = part.split('=', 1)
                else:
                    continue
                
                key = key.strip().upper()
                
                if key in ['E1', 'E2', 'E3', 'E4']:
                    try:
                        stretch_filters[key] = float(value.strip())
                        parsed_count += 1
                    except ValueError:
                        return None, None, None, {}, f"⚠️ Không thể parse giá trị '{value}' cho {key}"
            
            if stretch_filters:
                log_msg = f"🎯 Lọc kéo dãn cụ thể: {stretch_filters} ({parsed_count} điều kiện)"
                return None, None, None, stretch_filters, log_msg
            else:
                return None, None, None, {}, "⚠️ Không parse được filter kéo dãn nào. VD đúng: E1:1.3, E2:2.5"
        
        # FORMAT CŨ: exact:1.5,1.3,2.5
        elif elongation_input.lower().startswith('exact:'):
            exact_part = elongation_input.split(':', 1)[1].strip()
            exact_vals = [float(v.strip()) for v in exact_part.split(',') if v.strip()]
            if len(exact_vals) != 3:
                return None, None, None, {}, "⚠️ Format exact cần đúng 3 số: exact:1.5,1.3,2.5"
            log_msg = f"🔍 Lọc chính xác (E1,E2,E3): {exact_vals}"
            return None, None, exact_vals, {}, log_msg

        # FORMAT CŨ: 1.5,2.5 hoặc 2.5
        elif ',' in elongation_input:
            parts = [float(p.strip()) for p in elongation_input.split(',') if p.strip()]
            if len(parts) == 3:
                exact_vals = parts
                log_msg = f"🔍 Lọc chính xác (E1,E2,E3): {exact_vals}"
                return None, None, exact_vals, {}, log_msg
            elif len(parts) == 2:
                min_val = min(parts)
                max_val = max(parts)
                log_msg = f"🔍 Lọc (E1,E2,E3): {min_val} ≤ kéo dãn ≤ {max_val}"
                return min_val, max_val, None, {}, log_msg
            elif len(parts) == 1:
                max_val = parts[0]
                log_msg = f"🔍 Lọc (E1,E2,E3) ≤ {max_val}"
                return None, max_val, None, {}, log_msg
        else:
            max_val = float(elongation_input)
            log_msg = f"🔍 Lọc (E1,E2,E3) ≤ {max_val}"
            return None, max_val, None, {}, log_msg

        return None, None, None, {}, ""

    except ValueError:
        return None, None, None, {}, "⚠️ Format kéo dãn không hợp lệ. VD: E1:1.3,E2:2.5 hoặc 2.5 hoặc 1.5,3.0"

def adjust_ratios(ratio_dict, num_units=None):
    """Logic điều chỉnh tỷ lệ màu"""
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

    if abs(total_after - 100) < 2 and temp_adjusted:
        max_color = max(temp_adjusted, key=temp_adjusted.get)
    else:
        max_color = None

    no_split_major = max_color is not None

    adjusted_full_ratios = temp_adjusted.copy()
    remaining_percent = round(100 - total_after, 2)

    if remaining_percent > 0:
        if removed_color and removed_color not in adjusted_full_ratios:
            adjusted_full_ratios[removed_color] = remaining_percent
            log.append(f"🎨 Màu {removed_color} (đã bị loại) được rải để đủ 100%: {remaining_percent:.2f}%")
        elif max_color_initial not in adjusted_full_ratios:
            adjusted_full_ratios[max_color_initial] = remaining_percent
            log.append(f"🎨 Màu {max_color_initial} (màu lớn) được rải để đủ 100%: {remaining_percent:.2f}%")
        else:
            adjusted_full_ratios[max_color_initial] += remaining_percent
            log.append(f"🎨 Màu {max_color_initial} được rải thêm: {remaining_percent:.2f}% → tổng: {adjusted_full_ratios[max_color_initial]:.2f}%")

    for orig_color in ratio_dict.keys():
        if orig_color not in adjusted_full_ratios:
            adjusted_full_ratios[orig_color] = 0.0
            log.append(f"⚪ Màu {orig_color} không được sử dụng: 0.00%")

    return temp_adjusted, color_percent, log, max_color_initial, max_white_color, total_after, excluded_colors, no_split_major, max_color, adjusted_full_ratios

def format_float_keep_one_decimal(x):
    s = f"{x:.2f}"
    s = s.rstrip('0').rstrip('.')
    return s

def match_colors_to_calculated_ratios(color_ratios, calc_ratios, tolerance=2.0,
                                       excluded_colors=None, adjusted_full_ratios=None,
                                       priority_colors=None):
    """
    Match màu với tỷ lệ A-H đã tính toán
    - TỐI ƯU + FIX PRIORITY
    - ƯU TIÊN MÀU TRẮNG Ở VỊ TRÍ A VÀ H (LINH HOẠT)
    """
    if excluded_colors is None:
        excluded_colors = set()
    if priority_colors is None:
        priority_colors = []

    mapping = {}
    used_cuis = set()
    used_colors = set()  # Track màu đã dùng

    # Xác định màu trắng trong input
    white_colors_in_input = [c for c in color_ratios.keys() if c in white_keys]
    
    # ƯU TIÊN ĐẶT MÀU TRẮNG VÀO VỊ TRÍ A VÀ H TRƯỚC
    priority_positions = ['A', 'H']
    
    # Thử tất cả các kết hợp màu trắng cho A và H
    best_white_mapping = None
    best_white_error = float('inf')
    
    if len(white_colors_in_input) >= 2:
        # Nếu có ít nhất 2 màu trắng, thử tất cả các kết hợp
        from itertools import permutations, combinations_with_replacement
        
        # Thử các cặp màu trắng khác nhau: (W, SW), (SW, W), (W, W), (SW, SW)...
        for white_combo in combinations_with_replacement(white_colors_in_input, 2):
            for white_perm in set(permutations(white_combo)):
                temp_error = 0
                valid = True
                
                # Kiểm tra A với màu trắng đầu tiên
                if abs(calc_ratios['A'] - color_ratios[white_perm[0]]) <= tolerance:
                    temp_error += abs(calc_ratios['A'] - color_ratios[white_perm[0]])
                else:
                    valid = False
                
                # Kiểm tra H với màu trắng thứ hai
                if valid and abs(calc_ratios['H'] - color_ratios[white_perm[1]]) <= tolerance:
                    temp_error += abs(calc_ratios['H'] - color_ratios[white_perm[1]])
                else:
                    valid = False
                
                # Lưu kết hợp tốt nhất
                if valid and temp_error < best_white_error:
                    best_white_error = temp_error
                    best_white_mapping = white_perm
    
    elif len(white_colors_in_input) == 1:
        # Nếu chỉ có 1 màu trắng, thử đặt vào cả A và H
        white_color = white_colors_in_input[0]
        error_A = abs(calc_ratios['A'] - color_ratios[white_color])
        error_H = abs(calc_ratios['H'] - color_ratios[white_color])
        
        # Ưu tiên vị trí có sai số nhỏ hơn
        if error_A <= tolerance or error_H <= tolerance:
            if error_A <= error_H:
                best_white_mapping = (white_color, white_color)
            else:
                best_white_mapping = (white_color, white_color)
    
    # Áp dụng kết hợp tốt nhất
    if best_white_mapping:
        mapping['A'] = best_white_mapping[0]
        used_cuis.add('A')
        used_colors.add(best_white_mapping[0])
        
        mapping['H'] = best_white_mapping[1]
        used_cuis.add('H')
        used_colors.add(best_white_mapping[1])

    # Tiếp tục match các màu còn lại
    all_colors = sorted(color_ratios.items(), key=lambda x: -x[1])

    available_cuis = list('ABCDEFGH')
    combo_2 = list(combinations(available_cuis, 2))
    combo_3 = list(combinations(available_cuis, 3))

    for color, val in all_colors:
        # Bỏ qua màu đã được match đủ số lần (kiểm tra số lần xuất hiện)
        color_count_needed = 1
        color_count_used = sum(1 for c in mapping.values() if c == color)
        
        # Nếu màu đã dùng đủ, bỏ qua
        if color_count_used >= color_count_needed and color in white_colors_in_input:
            # Màu trắng có thể dùng nhiều lần
            pass
        elif color in used_colors and color not in white_colors_in_input:
            continue
            
        best_cui = None
        min_diff = tolerance + 1

        for cui in available_cuis:
            if cui in used_cuis:
                continue
            diff = abs(calc_ratios[cui] - val)
            if diff < min_diff:
                best_cui = cui
                min_diff = diff

        if best_cui and min_diff <= tolerance:
            mapping[best_cui] = color
            used_cuis.add(best_cui)
            used_colors.add(color)
        else:
            found = False

            # Thử kết hợp 2 cúi
            for combo in combo_2:
                if any(c in used_cuis for c in combo):
                    continue
                combo_sum = calc_ratios[combo[0]] + calc_ratios[combo[1]]
                if abs(combo_sum - val) <= tolerance:
                    for c in combo:
                        mapping[c] = color
                        used_cuis.add(c)
                    used_colors.add(color)
                    found = True
                    break

            # Thử kết hợp 3 cúi nếu chưa tìm được
            if not found:
                for combo in combo_3:
                    if any(c in used_cuis for c in combo):
                        continue
                    combo_sum = sum(calc_ratios[c] for c in combo)
                    if abs(combo_sum - val) <= tolerance:
                        for c in combo:
                            mapping[c] = color
                            used_cuis.add(c)
                        used_colors.add(color)
                        found = True
                        break

            if not found:
                return None

    # Điền màu fill cho các vị trí còn trống
    fill_color = next(iter(excluded_colors)) if excluded_colors else "W"
    for cui in available_cuis:
        if cui not in used_cuis:
            mapping[cui] = fill_color

    # Tính toán sai số
    actual_by_color = defaultdict(float)
    for cui, color in mapping.items():
        actual_by_color[color] += calc_ratios[cui]

    color_errors = {}
    total_error = 0
    priority_error = 0

    ratios_to_check = adjusted_full_ratios if adjusted_full_ratios else color_ratios

    for color in ratios_to_check:
        expected_val = ratios_to_check[color]
        actual_val = actual_by_color.get(color, 0)
        diff = abs(actual_val - expected_val)
        color_errors[color] = {
            'expected': expected_val,
            'actual': actual_val,
            'error': diff
        }
        total_error += diff
        
        if priority_colors and color in priority_colors:
            priority_error += diff

    # Tạo chuỗi kết quả
    result_parts = []
    cols = list("ABCDEFGH")
    i = 0
    while i < len(cols):
        c1 = cols[i]
        label1 = mapping[c1]
        val1 = calc_ratios[c1]
        if i + 1 < len(cols):
            c2 = cols[i + 1]
            label2 = mapping[c2]
            val2 = calc_ratios[c2]
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

    return {
        "mapping": "/".join(result_parts),
        "total_error": round(total_error, 2),
        "priority_error": round(priority_error, 2),
        "color_errors": color_errors,
        "calc_ratios": calc_ratios,
        "mapping_dict": mapping
    }
def match_colors_to_row_debug(color_ratios, row, tolerance=1.5, excluded_colors=None,
                               priority_colors=None, split_threshold=21,
                               max_color_initial=None, adjusted_full_ratios=None):
    """Logic match màu với row từ database"""
    if excluded_colors is None:
        excluded_colors = {"W", "SW", "FW"}

    df_ratios = {c: row[c] for c in 'ABCDEFGH'}
    mapping = {}
    used_cuis = set()

    max_color, max_val = max(color_ratios.items(), key=lambda x: x[1])
    total_ratio = sum(color_ratios.values())

    all_colors = sorted(color_ratios.items(), key=lambda x: -x[1])
    for color, val in all_colors:
        if color == max_color and abs(total_ratio - 100) <= 2.0:
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
            else:
                return None, None
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
            else:
                return None, None

    remaining = [c for c in df_ratios if c not in used_cuis]
    if abs(total_ratio - 100) <= 2.0:
        for cui in remaining:
            mapping[cui] = max_color
            used_cuis.add(cui)
    else:
        fill_color = next(iter(excluded_colors)) if excluded_colors else "W"
        for cui in remaining:
            mapping[cui] = fill_color
            used_cuis.add(cui)

    actual_by_color = defaultdict(float)
    for cui, color in mapping.items():
        actual_by_color[color] += df_ratios[cui]

    color_errors = {}
    total_error = 0
    priority_error = 0

    ratios_to_check = adjusted_full_ratios if adjusted_full_ratios else color_ratios

    for color in ratios_to_check:
        expected_val = ratios_to_check[color]
        actual_val = actual_by_color.get(color, 0)
        diff = abs(actual_val - expected_val)
        color_errors[color] = {
            'expected': expected_val,
            'actual': actual_val,
            'error': diff
        }
        total_error += diff
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

    mapping_str = "/".join(result_parts) + f" ({format_float_keep_one_decimal(row['stretch1'])}/" \
                                        f"{format_float_keep_one_decimal(row['stretch2'])}/" \
                                        f"{format_float_keep_one_decimal(row['stretch3'])}/" \
                                        f"{format_float_keep_one_decimal(row['stretch4'])})"

    return {
        "Row": int(row["row_id"]),
        "Mapping": mapping_str,
        "Sai số": round(total_error, 2),
        "Sai số ưu tiên": round(priority_error, 2),
        "Ratios": df_ratios,
        "MappingDict": mapping,
        "ColorErrors": color_errors
    }, None

def parse_arrangement_to_positions(arrangement_input):
    """Parse arrangement filter"""
    if not arrangement_input.strip():
        return {}

    position_mapping = {}
    positions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    arrangement_input = arrangement_input.strip()
    arrangement_input = re.sub(r"\([^)]*\)$", "", arrangement_input).strip()

    if ":" in arrangement_input:
        parts = re.split(r'[;,]', arrangement_input)
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

    parts = arrangement_input.split('/')
    current_pos = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if '+' in part:
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
    """Kiểm tra arrangement filter"""
    if not arrangement_filters:
        return True

    mapping_dict = result.get("MappingDict") or result.get("mapping_dict", {})

    for position, expected_colors in arrangement_filters.items():
        actual_color = mapping_dict.get(position, "")

        if '+' in expected_colors:
            expected_list = [c.strip() for c in expected_colors.split('+')]
            if actual_color not in expected_list:
                return False
        else:
            if actual_color != expected_colors:
                return False

    return True

def preview_arrangement_filters(arrangement_input):
    """Preview arrangement"""
    if not arrangement_input.strip():
        return ""

    try:
        position_mapping = parse_arrangement_to_positions(arrangement_input)
        if not position_mapping:
            return "⚠️ Không thể parse format sắp cúi. VD đúng: 1G02/1G02/1G01+1SW/2SW/1G02/1SW"

        positions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        visual = "🎨 Visualization: "
        for pos in positions:
            color = position_mapping.get(pos, "?")
            visual += f"{pos}({color}) "

        return visual.strip()

    except Exception as e:
        return f"⚠️ Lỗi parse: {str(e)}"

def combine_color_inputs(color_names, color_ratios):
    """Kết hợp tên màu và tỷ lệ"""
    if not color_ratios.strip():
        return ""

    ratio_lines = [line.strip() for line in color_ratios.strip().split("\n") if line.strip()]
    
    # Nếu không có tên màu, tự động tạo tên A, B, C, D...
    if not color_names.strip():
        auto_names = [chr(65 + i) for i in range(len(ratio_lines))]  # A=65 trong ASCII
        combined_lines = []
        for name, ratio in zip(auto_names, ratio_lines):
            combined_lines.append(f"{name}: {ratio}")
        return "\n".join(combined_lines)
    
    # Nếu đã có tên màu, sử dụng như cũ
    name_lines = [line.strip() for line in color_names.strip().split("\n") if line.strip()]

    if len(name_lines) != len(ratio_lines):
        return f"⚠️ Số lượng tên màu ({len(name_lines)}) khác với số tỷ lệ ({len(ratio_lines)})"

    combined_lines = []
    for name, ratio in zip(name_lines, ratio_lines):
        combined_lines.append(f"{name}: {ratio}")

    return "\n".join(combined_lines)
def preview_combined_ratios(color_names, color_ratios):
    """Preview tỷ lệ màu"""
    combined_input = combine_color_inputs(color_names, color_ratios)
    if not combined_input or combined_input.startswith("⚠️"):
        return combined_input

    if not combined_input.strip():
        return ""
    lines = combined_input.strip().split("\n")
    ratios = {}
    log = ["📥 Tỷ lệ màu người dùng đã nhập:"]
    total = 0

    pattern = re.compile(
        r"^\s*([A-Za-z0-9_\-\(\)\u4e00-\u9fff]+)\s*[:\t,; ]+\s*([0-9]+(?:[.,][0-9]+)?)\s*%?\s*$"
    )

    for line in lines:
        m = pattern.match(line)
        if not m:
            return f"⚠️ Sai định dạng ở dòng: '{line}'. Đúng dạng: Tên: số (ví dụ W: 5.0)"
        k, v = m.groups()
        k = k.strip().upper()
        val = float(v.replace(",", "."))
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
def check_stretch_filter(mapping_str, stretch_filters):
    """
    Kiểm tra kéo dãn từ mapping string
    VD: "1G02/1G02/1SW+1G01/2SW/1G02/1SW (1.3/2.3/1.5/3.2)"
    stretch_filters = {'E1': 1.3, 'E2': 2.5} → Chỉ lấy kết quả có E1≈1.3 VÀ E2≈2.5
    """
    if not stretch_filters:
        return True
    
    try:
        # Trích xuất phần kéo dãn từ dấu ngoặc
        if '(' not in mapping_str or ')' not in mapping_str:
            return True
        
        stretch_part = mapping_str[mapping_str.rindex('(') + 1:mapping_str.rindex(')')]
        stretch_values = [float(x.strip()) for x in stretch_part.split('/')]
        
        if len(stretch_values) != 4:
            return True
        
        e1, e2, e3, e4 = stretch_values
        
        tolerance = 0.01  # Sai số cho phép ±0.01
        
        # Kiểm tra TỪNG điều kiện
        for key, expected_val in stretch_filters.items():
            if key == 'E1' and abs(e1 - expected_val) > tolerance:
                return False
            elif key == 'E2' and abs(e2 - expected_val) > tolerance:
                return False
            elif key == 'E3' and abs(e3 - expected_val) > tolerance:
                return False
            elif key == 'E4' and abs(e4 - expected_val) > tolerance:
                return False
        
        return True
    except Exception as e:
        # Debug: In lỗi ra console
        print(f"⚠️ check_stretch_filter error: {e} | mapping_str={mapping_str}")
        return True

def check_arrangement_filter(result, arrangement_filters):
    """Kiểm tra arrangement filter"""
    if not arrangement_filters:
        return True

    mapping_dict = result.get("MappingDict") or result.get("mapping_dict", {})

    for position, expected_colors in arrangement_filters.items():
        actual_color = mapping_dict.get(position, "")

        if '+' in expected_colors:
            expected_list = [c.strip() for c in expected_colors.split('+')]
            if actual_color not in expected_list:
                return False
        else:
            if actual_color != expected_colors:
                return False

    return True
def render_result_table(results, page, page_size=10):
    """Render bảng kết quả"""
    start = page * page_size
    end = start + page_size
    page_results = results[start:end]
    if not page_results:
        return "⚠️ Không có kết quả để hiển thị."

    data = []
    for i, r in enumerate(page_results, start=start + 1):
        row_info = {
            "STT": i,
            "Row": r.get("Row", "-"),
            "Tên SP": r.get("ProductName", ""),
        }

        color_errors = r.get("ColorErrors") or r.get("color_errors", {})

        for color, error_info in sorted(color_errors.items()):
            expected = error_info['expected']
            actual = error_info['actual']
            error = error_info['error']
            row_info[color] = f"{color}: {expected:.2f} → {actual:.2f} = {error:.2f}"

        row_info["Sai số"] = r.get("Sai số") or r.get("total_error", 0)
        row_info["Sai số ƯT"] = r.get("Sai số ưu tiên", 0)
        row_info["Sắp cúi"] = r.get("Mapping") or r.get("mapping", "")
        data.append(row_info)

    # ✅ Sửa chỗ này
    all_colors = set()
    for row in data:
        all_colors.update([
            k for k in row.keys()
            if k not in ["STT", "Row", "Tên SP", "Sai số", "Sai số ƯT", "Sắp cúi"]
        ])

    columns_order = ["STT", "Row", "Tên SP"] + sorted(all_colors) + ["Sai số", "Sai số ƯT", "Sắp cúi"]

    df_result = pd.DataFrame(data)
    for col in all_colors:
        if col not in df_result.columns:
            df_result[col] = ""
        else:
            df_result[col] = df_result[col].fillna("")

    df_result = df_result[columns_order]
    return df_result.to_markdown(index=False)

def prev_page(results, current, page_size=10):
    if current <= 0:
        return render_result_table(results, 0, page_size), 0
    return render_result_table(results, current - 1, page_size), current - 1

def next_page(results, current, page_size=10):
    max_page = len(results) // page_size
    if current + 1 > max_page:
        return render_result_table(results, current, page_size), current
    return render_result_table(results, current + 1, page_size), current + 1

def run_app(product_name, color_names, color_ratios, num_units, elongation_limit,
            priority_input, split_threshold_input, arrangement_filter_input, 
            search_mode): 
    log = []
    start_time = time.time()

    try:
        color_input = combine_color_inputs(color_names, color_ratios)
        if color_input.startswith("⚠️"):
            return color_input, "", "", [], 0
        if not color_input.strip():
            return "⚠️ Vui lòng nhập tên màu và tỷ lệ màu.", "", "", [], 0

        ratios = {}
        pattern = re.compile(
            r"^\s*([A-Za-z0-9_\-\(\)\u4e00-\u9fff]+)\s*[:\t,; ]+\s*([0-9]+(?:[.,][0-9]+)?)\s*%?\s*$"
        )
        for line in color_input.strip().split("\n"):
            match = pattern.match(line)
            if match:
                k, v = match.groups()
                ratios[k.strip().upper()] = float(v.replace(",", ".").strip())
            else:
                return f"⚠️ Sai định dạng ở dòng: '{line}'", "", "", [], 0

        total_ratio = sum(ratios.values())
        if total_ratio != 100.0:
            return f"⚠️ Tổng tỷ lệ phải chính xác phải bằng 100.00%. Hiện tại: {total_ratio:.2f}%", "", "", [], 0
        if len(ratios) < 2:
            return "⚠️ Cần ít nhất 2 màu để tra cứu.", "", "", [], 0

        log.append(f"🎨 Nhận được {len(ratios)} màu: {list(ratios.keys())}")
        log.append(f"📊 Tỷ lệ ban đầu: {ratios}")

        arrangement_filters = parse_arrangement_to_positions(arrangement_filter_input)
        if arrangement_filters:
            log.append(f"🎯 Điều kiện lọc sắp xếp: {arrangement_filters}")

        priority_colors = [s.strip().upper() for s in priority_input.split(",") if s.strip()] if priority_input else []
        if priority_colors:
            log.append(f"🔍 Màu ưu tiên sai số: {priority_colors}")

        split_threshold = float(split_threshold_input) if split_threshold_input else 21

        num_units = int(num_units) if num_units else None
        adjusted_ratios, color_percent, adjust_log, max_color_initial, \
            max_white_color, total_after, excluded_colors, \
            no_split_major, max_color, adjusted_full_ratios = adjust_ratios(ratios, num_units)

        log.extend(adjust_log)
        log.append(f"🔄 Tỉ lệ sau điều chỉnh: {adjusted_ratios}")
        log.append(f"🎨 Tỉ lệ đầy đủ (tính sai số): {adjusted_full_ratios}")

        if search_mode == "🔥 Kết hợp toàn diện (All Methods)":
            log.append("🚀🔥 CHẾ ĐỘ KẾT HỢP TOÀN DIỆN - TÌM KIẾM TỐI ĐA")
            
            all_results = []
            
            # === 1. TRA CỨU DATABASE ===
            log.append("\n📊 [1/3] Tra cứu Database...")
            min_val, max_val, exact_vals, stretch_filters, elongation_log = parse_elongation_filter(elongation_limit)
            if elongation_log and not elongation_log.startswith("⚠️"):
                log.append(elongation_log)
            
            try:
                df_all = query_data(min_val, max_val, exact_vals, stretch_filters)
                log.append(f"   Database: {len(df_all):,} dòng")
                
                db_count = 0
                for _, row in df_all.iterrows():
                    match_result = match_colors_to_row_debug(
                        adjusted_ratios, row, tolerance=2.0,
                        excluded_colors=excluded_colors,
                        priority_colors=priority_colors,
                        split_threshold=split_threshold,
                        max_color_initial=max_color_initial,
                        adjusted_full_ratios=adjusted_full_ratios
                    )
                    if match_result is not None:
                        res, _ = match_result
                        if res and check_arrangement_filter(res, arrangement_filters):
                            all_results.append(res)
                            db_count += 1
                log.append(f"   ✅ Database: Tìm thấy {db_count} kết quả")
            except Exception as e:
                log.append(f"   ⚠️ Database lỗi: {str(e)}")
            
            # === 2. OPTIMIZATION ===
            log.append("\n🎯 [2/3] Optimization...")
            stretch_bounds = (1.1, 6.0)
            if min_val and max_val:
                stretch_bounds = (min_val, max_val)
            elif max_val:
                stretch_bounds = (1.1, max_val)
            
            stretch_candidates_opt = find_optimal_stretches_scipy(
                adjusted_ratios,
                adjusted_full_ratios,
                excluded_colors,
                stretch_bounds=stretch_bounds,
                priority_colors=priority_colors
            )
            
            opt_count = 0
            for e1, e2, e3, e4, error in stretch_candidates_opt:
                calc_ratios = calculate_ratios_from_stretches(e1, e2, e3, e4)
                if calc_ratios is None:
                    continue

                match_result = match_colors_to_calculated_ratios(
                    adjusted_ratios, calc_ratios, tolerance=2.0,
                    excluded_colors=excluded_colors,
                    adjusted_full_ratios=adjusted_full_ratios,
                    priority_colors=priority_colors
                )

                if match_result:
                    mapping_str = match_result['mapping'] + \
                        f" ({format_float_keep_one_decimal(e1)}/" \
                        f"{format_float_keep_one_decimal(e2)}/" \
                        f"{format_float_keep_one_decimal(e3)}/" \
                        f"{format_float_keep_one_decimal(e4)})"

                    result_entry = {
                        "Row": "-",
                        "Mapping": mapping_str,
                        "total_error": match_result['total_error'],
                        "Sai số ưu tiên": match_result['priority_error'],
                        "ColorErrors": match_result['color_errors'],
                        "mapping_dict": match_result['mapping_dict']
                    }
                    
                    if check_arrangement_filter(result_entry, arrangement_filters):
                        all_results.append(result_entry)
                        opt_count += 1
            
            log.append(f"   ✅ Optimization: Tìm thêm {opt_count} kết quả")
            
            # === 3. GRID SEARCH ===
            log.append("\n🔍 [3/3] Grid Search...")
            if exact_vals:
                stretch_range = []
                for val in exact_vals:
                    stretch_range.extend(np.arange(val - 0.3, val + 0.3, 0.01).round(2))
                stretch_range = sorted(set(stretch_range))
            elif min_val and max_val:
                stretch_range = np.arange(min_val, min(max_val, 6.0) + 0.01, 0.01).round(2)
            elif max_val:
                stretch_range = np.arange(1.1, min(max_val, 6.0) + 0.01, 0.01).round(2)
            else:
                stretch_range = np.arange(1.5, 6.0, 0.01).round(2)
            
            stretch_candidates_grid = find_stretches_grid_search(
                adjusted_ratios,
                adjusted_full_ratios,
                excluded_colors,
                stretch_range=stretch_range,
                max_combinations=50000,
                priority_colors=priority_colors
            )
            
            grid_count = 0
            for e1, e2, e3, e4, error in stretch_candidates_grid:
                calc_ratios = calculate_ratios_from_stretches(e1, e2, e3, e4)
                if calc_ratios is None:
                    continue

                match_result = match_colors_to_calculated_ratios(
                    adjusted_ratios, calc_ratios, tolerance=2.0,
                    excluded_colors=excluded_colors,
                    adjusted_full_ratios=adjusted_full_ratios,
                    priority_colors=priority_colors
                )

                if match_result:
                    mapping_str = match_result['mapping'] + \
                        f" ({format_float_keep_one_decimal(e1)}/" \
                        f"{format_float_keep_one_decimal(e2)}/" \
                        f"{format_float_keep_one_decimal(e3)}/" \
                        f"{format_float_keep_one_decimal(e4)})"

                    result_entry = {
                        "Row": "-",
                        "Mapping": mapping_str,
                        "total_error": match_result['total_error'],
                        "Sai số ưu tiên": match_result['priority_error'],
                        "ColorErrors": match_result['color_errors'],
                        "mapping_dict": match_result['mapping_dict']
                    }
                    
                    if check_arrangement_filter(result_entry, arrangement_filters):
                        all_results.append(result_entry)
                        grid_count += 1
            
            log.append(f"   ✅ Grid Search: Tìm thêm {grid_count} kết quả")
            
            # === DEDUPLICATE & FILTER STRETCH ===
            log.append(f"\n🔄 Gộp kết quả: {len(all_results):,} → ")
            
            # ← LỌC THEO STRETCH_FILTERS
            if stretch_filters:
                before_stretch = len(all_results)
                all_results = [r for r in all_results 
                              if check_stretch_filter(r.get("Mapping", "") or r.get("mapping", ""), stretch_filters)]
                log.append(f"   📉 Lọc kéo dãn: {before_stretch:,} → {len(all_results):,}")
            
            unique_results = []
            seen = set()
            for r in all_results:
                key = r.get("Mapping", "")
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)
            
            results = unique_results
            log.append(f"{len(results):,} unique")
            log.append(f"📊 Database: {db_count} | Opt: {opt_count} | Grid: {grid_count}")
            
        elif search_mode == "Tính toán động (với bước nhảy 0.01)":
            log.append("🚀 Chế độ: TÍNH TOÁN ĐỘNG ")

            min_val, max_val, exact_vals, stretch_filters, elongation_log = parse_elongation_filter(elongation_limit)
            if elongation_log and not elongation_log.startswith("⚠️"):
                log.append(elongation_log)

            stretch_bounds = (1.1, 6.0)
            if min_val and max_val:
                stretch_bounds = (min_val, max_val)
            elif max_val:
                stretch_bounds = (1.1, max_val)

            log.append(f"📏 Khoảng kéo dãn: {stretch_bounds} (E1,E2,E3 max=4.0, E4 max=6.0)")

            stretch_candidates = find_optimal_stretches_scipy(
                adjusted_ratios,
                adjusted_full_ratios,
                excluded_colors,
                stretch_bounds=stretch_bounds,
                priority_colors=priority_colors
            )

            if not stretch_candidates:
                elapsed_time = time.time() - start_time
                return "\n".join(log + [f"❌ Không tìm thấy kết quả phù hợp. ⏱️ Thời gian: {elapsed_time:.2f}s"]), "", "", [], 0

            log.append(f"✅ Tìm thấy {len(stretch_candidates)} ứng viên từ optimization")

            results = []
            for e1, e2, e3, e4, error in stretch_candidates:
                calc_ratios = calculate_ratios_from_stretches(e1, e2, e3, e4)
                if calc_ratios is None:
                    continue

                match_result = match_colors_to_calculated_ratios(
                    adjusted_ratios, calc_ratios, tolerance=2.0,
                    excluded_colors=excluded_colors,
                    adjusted_full_ratios=adjusted_full_ratios,
                    priority_colors=priority_colors
                )

                if match_result:
                    mapping_str = match_result['mapping'] + \
                        f" ({format_float_keep_one_decimal(e1)}/" \
                        f"{format_float_keep_one_decimal(e2)}/" \
                        f"{format_float_keep_one_decimal(e3)}/" \
                        f"{format_float_keep_one_decimal(e4)})"

                    result_entry = {
                        "Row": "-",
                        "Mapping": mapping_str,
                        "total_error": match_result['total_error'],
                        "Sai số ưu tiên": match_result['priority_error'],
                        "ColorErrors": match_result['color_errors'],
                        "mapping_dict": match_result['mapping_dict']
                    }

                    if check_arrangement_filter(result_entry, arrangement_filters):
                        results.append(result_entry)

            # ← LỌC THEO STRETCH_FILTERS
            if stretch_filters:
                before_count = len(results)
                results = [r for r in results 
                          if check_stretch_filter(r.get("Mapping", ""), stretch_filters)]
                log.append(f"📉 Sau lọc kéo dãn: {before_count:,} → {len(results):,}")

        elif search_mode == "Tính toán động (Grid Search)":
            log.append("🔍 Chế độ: TÍNH TOÁN ĐỘNG - Grid Search")

            min_val, max_val, exact_vals, stretch_filters, elongation_log = parse_elongation_filter(elongation_limit)
            if elongation_log and not elongation_log.startswith("⚠️"):
                log.append(elongation_log)

            if exact_vals:
                stretch_range = []
                for val in exact_vals:
                    stretch_range.extend(np.arange(val - 0.3, val + 0.3, 0.01).round(2))
                stretch_range = sorted(set(stretch_range))
            elif min_val and max_val:
                stretch_range = np.arange(min_val, min(max_val, 6.0) + 0.01, 0.01).round(2)
            elif max_val:
                stretch_range = np.arange(1.1, min(max_val, 6.0) + 0.01, 0.01).round(2)
            else:
                stretch_range = np.arange(1.5, 6.0, 0.01).round(2)

            log.append(f"📏 Khoảng tìm kiếm: {len(stretch_range)} giá trị (max E1,E2,E3=4.0, max E4=6.0)")

            stretch_candidates = find_stretches_grid_search(
                adjusted_ratios,
                adjusted_full_ratios,
                excluded_colors,
                stretch_range=stretch_range,
                priority_colors=priority_colors
            )

            if not stretch_candidates:
                elapsed_time = time.time() - start_time
                return "\n".join(log + [f"❌ Không tìm thấy kết quả phù hợp. ⏱️ Thời gian: {elapsed_time:.2f}s"]), "", "", [], 0

            log.append(f"✅ Tìm thấy {len(stretch_candidates)} kết quả")

            results = []
            for e1, e2, e3, e4, error in stretch_candidates:
                calc_ratios = calculate_ratios_from_stretches(e1, e2, e3, e4)
                if calc_ratios is None:
                    continue

                match_result = match_colors_to_calculated_ratios(
                    adjusted_ratios, calc_ratios, tolerance=2.0,
                    excluded_colors=excluded_colors,
                    adjusted_full_ratios=adjusted_full_ratios,
                    priority_colors=priority_colors
                )

                if match_result:
                    mapping_str = match_result['mapping'] + \
                        f" ({format_float_keep_one_decimal(e1)}/" \
                        f"{format_float_keep_one_decimal(e2)}/" \
                        f"{format_float_keep_one_decimal(e3)}/" \
                        f"{format_float_keep_one_decimal(e4)})"

                    result_entry = {
                        "Row": "-",
                        "Mapping": mapping_str,
                        "total_error": match_result['total_error'],
                        "Sai số ưu tiên": match_result['priority_error'],
                        "ColorErrors": match_result['color_errors'],
                        "mapping_dict": match_result['mapping_dict']
                    }

                    if check_arrangement_filter(result_entry, arrangement_filters):
                        results.append(result_entry)

            # ← LỌC THEO STRETCH_FILTERS
            if stretch_filters:
                before_count = len(results)
                results = [r for r in results 
                          if check_stretch_filter(r.get("Mapping", ""), stretch_filters)]
                log.append(f"📉 Sau lọc kéo dãn: {before_count:,} → {len(results):,}")

        else:  # Tra cứu Database
            log.append("💾 Chế độ: TRA CỨU DATABASE")

            min_val, max_val, exact_vals, stretch_filters, elongation_log = parse_elongation_filter(elongation_limit)
            if elongation_log.startswith("⚠️"):
                return elongation_log, "", "", [], 0
            if elongation_log:
                log.append(elongation_log)

            try:
                df_all = query_data(min_val, max_val, exact_vals, stretch_filters)
                con = duckdb.connect(DUCKDB_PATH, read_only=True)
                total_count = con.execute("SELECT COUNT(*) FROM color_data").fetchone()[0]
                con.close()

                if elongation_limit:
                    log.append(f"📉 Lọc kéo dãn: {total_count:,} → {len(df_all):,} dòng")
                else:
                    log.append(f"📊 Tổng số dòng: {len(df_all):,}")

            except Exception as e:
                elapsed_time = time.time() - start_time
                return f"⚠️ Lỗi query database: {str(e)} ⏱️ {elapsed_time:.2f}s", "", "", [], 0

            if df_all.empty:
                elapsed_time = time.time() - start_time
                return "\n".join(log + [f"❌ Không có dữ liệu thỏa mãn điều kiện. ⏱️ {elapsed_time:.2f}s"]), "", "", [], 0

            results = []
            total_before_filter = 0
            for _, row in df_all.iterrows():
                match_result = match_colors_to_row_debug(
                    adjusted_ratios, row, tolerance=2.0,
                    excluded_colors=excluded_colors,
                    priority_colors=priority_colors,
                    split_threshold=split_threshold,
                    max_color_initial=max_color_initial,
                    adjusted_full_ratios=adjusted_full_ratios
                )
                if match_result is not None:
                    res, _ = match_result
                    if res:
                        total_before_filter += 1
                        if check_arrangement_filter(res, arrangement_filters):
                            results.append(res)

            log.append(f"📈 Tìm thấy {total_before_filter:,} kết quả phù hợp")
            if arrangement_filters:
                log.append(f"🎯 Sau lọc sắp xếp: {len(results):,} kết quả")
            
            # ← LỌC THEO STRETCH_FILTERS
            if stretch_filters:
                before_count = len(results)
                results = [r for r in results 
                          if check_stretch_filter(r.get("Mapping", ""), stretch_filters)]
                log.append(f"📉 Sau lọc kéo dãn: {before_count:,} → {len(results):,}")

        # Sắp xếp results
        results = sorted(results, key=lambda x: (x.get("Sai số ưu tiên", 0), x.get("total_error") or x.get("Sai số", 0)))

        # Gán ProductName
        for r in results:
            r["ProductName"] = product_name

        if not results:
            elapsed_time = time.time() - start_time
            return "\n".join(log + [f"❌ Không tìm thấy kết quả phù hợp. ⏱️ Thời gian: {elapsed_time:.2f}s"]), "", "", [], 0

        elapsed_time = time.time() - start_time
        log.append(f"⏱️ Thời gian xử lý: {elapsed_time:.2f} giây")
        log.append(f"📊 Tốc độ: {len(results)/elapsed_time:.1f} kết quả/giây")

        first_page_table = render_result_table(results, 0)
        return "\n".join(log), "", first_page_table, results, 0

    except Exception as e:
        import traceback
        elapsed_time = time.time() - start_time
        return f"⚠️ Lỗi: {str(e)}\n⏱️ Thời gian: {elapsed_time:.2f}s\n\n{traceback.format_exc()}", "", "", [], 0

# ======= GIAO DIỆN GRADIO =========
def get_four_mg_stretch_app():
    if os.path.exists(EXCEL_PATH):
        init_database()
    else:
        print("⚠️ Không tìm thấy file Excel. Chỉ chạy chế độ tính toán động.")

    with gr.Blocks() as app:
        gr.Markdown("<h2 style='text-align: center;'>🎨 Tra cứu tỷ lệ màu máy ghép 5.0 </h2>")
        with gr.Row():
            with gr.Column(scale=1):
                color_names_input = gr.Textbox(
                    lines=4,
                    label="🎨 Tên màu",
                    placeholder="G004\nG024\nXX"
                )
                num_units_input = gr.Textbox(
                    label="🔹 Số cúi tách (2–6, tùy chọn)",
                    placeholder="VD: 3",
                    value="0" 
                )
                elongation_limit_input = gr.Textbox(
                    label="🧪 Lọc 4 chỉ số kéo giãn",
                    placeholder="VD: 2.5 hoặc 1.5,3.0 (khoảng) hoặc exact:1.5,1.3,2.5"
                )
                priority_color_input = gr.Textbox(
                    label="🎯 Màu ưu tiên sai số",
                    placeholder="VD: G004, G024"
                )
                product_name_input = gr.Textbox(
                    label="📦 Tên sản phẩm",
                    placeholder="VD: ABC-123",
                    value=""
                )
            with gr.Column(scale=2):
                color_ratios_input = gr.Textbox(
                    lines=4,
                    label="📊 Tỷ lệ (%)",
                    placeholder="18.0\n40.0\n42.0"
                )
                realtime_log = gr.Textbox(
                    label="📥 Tỷ lệ màu đã nhập",
                    lines=6,
                    interactive=False
                )
                structure_line = gr.Textbox(
                    label="🧱 Cấu trúc tương ứng",
                    interactive=False
                )

            with gr.Column(scale=3):
                search_mode_dropdown = gr.Dropdown(
                    choices=[
                        "Tính toán động (với bước nhảy 0.01)",
                        # "Tính toán động (Grid Search)",
                        "Tra cứu Database",
                        "🔥 Kết hợp toàn diện (All Methods)"
                    ],
                    value="Tính toán động (với bước nhảy 0.01)",
                    label="🔧 Chế độ tìm kiếm",
                    info="Optimization: Nhanh | Grid: Đầy đủ | Database: Có sẵn | All: Tối đa kết quả"
                )
                arrangement_filter_input = gr.Textbox(
                    label="🎯 Lọc theo sắp cúi",
                    placeholder="VD: 1G02/1G02/1G01+1SW/2SW/1G02/1SW"
                )
                arrangement_filter_preview = gr.Textbox(
                    label="🎯 Preview sắp cúi",
                    lines=2,
                    interactive=False
                )
                split_threshold_input = gr.Textbox(
                    label="✂️ Ngưỡng tách màu",
                    placeholder="VD: 21"
                )
                run_btn = gr.Button("🔍 Tra cứu", variant="primary")

            with gr.Column(scale=4):
                log_output = gr.Textbox(
                    label="📋 Thông tin xử lý",
                    lines=15,
                    interactive=False
                )

        table_output = gr.Markdown(label="📊 Kết quả")

        results_state = gr.State([])
        current_page = gr.State(0)

        with gr.Row(visible=False) as pagination_row:
            prev_btn = gr.Button("⬅️ Trang trước")
            next_btn = gr.Button("➡️ Trang sau")

        def run_and_toggle(*args):
            log, structure, table, results, page = run_app(*args)
            show_pagination = gr.update(visible=(len(results) > 0))
            return log, structure, table, results, page, show_pagination

        run_btn.click(
            fn=run_and_toggle,
            inputs=[
                product_name_input,
                color_names_input,
                color_ratios_input,
                num_units_input,
                elongation_limit_input,    
                priority_color_input,
                split_threshold_input,
                arrangement_filter_input,
                search_mode_dropdown
            ],
            outputs=[log_output, structure_line, table_output, results_state, current_page, pagination_row]
        )

        color_names_input.change(
            preview_combined_ratios,
            inputs=[color_names_input, color_ratios_input],
            outputs=realtime_log
        )
        color_ratios_input.change(
            preview_combined_ratios,
            inputs=[color_names_input, color_ratios_input],
            outputs=realtime_log
        )

        num_units_input.change(
            get_structure_line_from_textbox,
            inputs=num_units_input,
            outputs=structure_line
        )
        arrangement_filter_input.change(
            preview_arrangement_filters,
            inputs=arrangement_filter_input,
            outputs=arrangement_filter_preview
        )

        prev_btn.click(
            prev_page,
            inputs=[results_state, current_page],
            outputs=[table_output, current_page]
        )
        next_btn.click(
            next_page,
            inputs=[results_state, current_page],
            outputs=[table_output, current_page]
        )

    return app

four_stretch_app_mg = get_four_mg_stretch_app()
__all__ = ["four_stretch_app_mg"]