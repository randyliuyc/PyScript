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

# Responsive CSS
from responsive_css import RESPONSIVE_CSS

# ======= CẤU HÌNH =========
EXCEL_PATH = "Data4kd_ratio_6.xlsx"
SHEET_NAME = "Sheet1"
DUCKDB_PATH = "color_data_6.duckdb"
white_keys = ["W", "SW", "WP", "SWP", "FWP", "WJ", "WPJ", "SWJ", "SWPJ", "FW", "FWJ", "FWPJ", "WAO","WC","WB","WUS","WOC","WGEC","WL","WN","WM","WTE","WT"]

# ===== Hàm làm tròn chuẩn Excel =====
def excel_round(value, digits=2):
    """Làm tròn theo quy tắc ROUND_HALF_UP của Excel"""
    return float(Decimal(str(value)).quantize(Decimal('1.' + '0'*digits), rounding=ROUND_HALF_UP))
def calculate_ratios_from_stretches(e1, e2, e3, e4):
    """
    Tính tỷ lệ A–H (KHÔNG LÀM TRÒN)
    Chỉ trả về float (high precision)
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
        
        if denominator == 0:
            return None

        A = (inv_e1 / denominator) * 100
        B = (inv_e4 / denominator) * 100
        C = (Decimal('1') / denominator) * 100
        D = (Decimal('1') / denominator) * 100
        E = (inv_e2 / denominator) * 100
        F = (inv_e2 / denominator) * 100
        G = (inv_e4 / denominator) * 100
        H = (inv_e3 / denominator) * 100

        # ‼️ THAY ĐỔI: TRẢ VỀ GIÁ TRỊ CHÍNH XÁC (FLOAT), KHÔNG LÀM TRÒN TỪNG PHẦN
        ratios = {
            'A': float(A),
            'B': float(B),
            'C': float(C),
            'D': float(D),
            'E': float(E),
            'F': float(F),
            'G': float(G),
            'H': float(H)
        }
        
        # Total vẫn làm tròn để hiển thị, không ảnh hưởng tính toán
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
    
    # ✨ THÊM: E4 phải lớn hơn E1 và E3 ít nhất 1.1 lần
    if e4 / e1 < 1.1 or e4 / e3 < 1.1:
        return False
    
    # Giữ nguyên điều kiện cũ: không quá 4.0 lần
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

def calculate_stretch_variance(mapping_str):
    """
    Tính độ phân tán của E1, E2, E3 (variance)
    Kéo dãn gần nhau → variance nhỏ → ưu tiên hơn
    VD: [1.5, 1.6, 1.7] có variance nhỏ hơn [1.5, 2.5, 3.5]
    """
    try:
        if '(' not in mapping_str or ')' not in mapping_str:
            return 999.0
        
        stretch_part = mapping_str[mapping_str.rindex('(') + 1:mapping_str.rindex(')')]
        stretch_values = [float(x.strip()) for x in stretch_part.split('/')]
        
        if len(stretch_values) == 4:
            e1, e2, e3, e4 = stretch_values
            # Chỉ tính variance của E1, E2, E3
            mean_123 = (e1 + e2 + e3) / 3
            variance = sum((x - mean_123) ** 2 for x in [e1, e2, e3]) / 3
            return variance
        return 999.0
    except:
        return 999.0
def calculate_total_stretch(mapping_str):
    """
    Tính tổng E1+E2+E3+E4
    Kéo dãn tổng nhỏ → máy dễ chạy hơn
    VD: (1.47/1.31/1.68/3.92) → tổng = 8.38
    """
    try:
        if '(' not in mapping_str or ')' not in mapping_str:
            return 999.0
        
        stretch_part = mapping_str[mapping_str.rindex('(') + 1:mapping_str.rindex(')')]
        stretch_values = [float(x.strip()) for x in stretch_part.split('/')]
        
        if len(stretch_values) == 4:
            return sum(stretch_values)
        return 999.0
    except:
        return 999.0
# ===== THAY THẾ HÀM find_optimal_stretches_scipy CŨ =====
def find_optimal_stretches_scipy(target_ratios, adjusted_full_ratios, excluded_colors,
                                 stretch_bounds=(1.1, 5.0), method='differential_evolution',
                                 priority_colors=None, iteration=1, use_cache=True, cache_dict=None,
                                 excluded_colors_ratios=None, fill_color_for_matching=None):
    
    if cache_dict is None:
        cache_dict = {'results': [], 'explored_regions': set(), 'best_seeds': []}
    ERROR_THRESHOLD = 1.5

    if iteration == 1:
        max_iterations = 500   
        exploration_rate = 1.0 
        print(f"⚡ Lần {iteration}: TÌM NHANH (500 vòng, 100% random)")
    elif iteration == 2:
        max_iterations = 1000
        exploration_rate = 0.7  
        print(f"🎯 Lần {iteration}: TỐI ƯU (1000 vòng, 70% random + 30% cache)")
    elif iteration == 3:
        max_iterations = 2000   
        exploration_rate = 0.5  
        print(f"🔥 Lần {iteration}: TÌM SÂU (2000 vòng, 50% random + 50% cache)")
    else:
        max_iterations = 2000 + (iteration - 3) * 1000
        exploration_rate = 0.3
        print(f"🚀 Lần {iteration}: TÌM CHUYÊN SÂU ({max_iterations} vòng, 30% random + 70% cache)")

    calc_cache = {}
    all_results = [] 

    if use_cache and iteration > 1 and cache_dict['results']:
        print(f"♻️ Dùng {len(cache_dict['results'])} kết quả cache làm 'seeds'")
        best_results = sorted(cache_dict['results'], key=lambda x: x[4])[:20]
        cache_dict['best_seeds'] = [r[:4] for r in best_results]
        print(f"📌 Có {len(cache_dict['best_seeds'])} điểm xuất phát tốt")
    else:
        cache_dict['best_seeds'] = []

    def objective_function(stretches):
        e1_raw, e2_raw, e3_raw, e4_raw = stretches
        if e1_raw > 4.0 or e2_raw > 4.0 or e3_raw > 4.0: return 10000.0
        if e4_raw > 6.0: return 10000.0
        if e4_raw <= e1_raw or e4_raw <= e3_raw: return 10000.0
        if e4_raw / e1_raw < 1.1 or e4_raw / e3_raw < 1.1: return 10000.0
        if e4_raw / e1_raw >= 4.0 and e4_raw / e3_raw >= 4.0: return 10000.0

        e1, e2, e3, e4 = round(e1_raw, 2), round(e2_raw, 2), round(e3_raw, 2), round(e4_raw, 2)
        stretch_key = (e1, e2, e3, e4)

        if stretch_key in calc_cache:
            cached_result = calc_cache[stretch_key]
            if cached_result is None: return 1000.0
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
            priority_colors=priority_colors,
            excluded_colors_ratios=excluded_colors_ratios,
            fill_color_for_matching=fill_color_for_matching
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
        (1.1, 6.0)
    ]
    found_count = 0
    progress_interval = max(100, max_iterations // 100)

    for seed_val in range(max_iterations):
        if seed_val > 0 and seed_val % progress_interval == 0:
            progress_pct = (seed_val / max_iterations) * 100
            print(f"    Progress: {seed_val}/{max_iterations} ({progress_pct:.1f}%) | Found {found_count} | Cache: {len(calc_cache)}")

        if random.random() < exploration_rate or not cache_dict['best_seeds'] or iteration == 1:
            x0 = [
                round(random.uniform(min_bound, min(4.0, stretch_bounds[1])), 2),
                round(random.uniform(min_bound, min(4.0, stretch_bounds[1])), 2),
                round(random.uniform(min_bound, min(4.0, stretch_bounds[1])), 2),
                round(random.uniform(min_bound, min(6.0, stretch_bounds[1])), 2)
            ]
        else:
            best_seed = random.choice(cache_dict['best_seeds'])
            x0 = [
                round(max(min_bound, min(4.0, best_seed[0] + random.uniform(-0.5, 0.5))), 2),
                round(max(min_bound, min(4.0, best_seed[1] + random.uniform(-0.5, 0.5))), 2),
                round(max(min_bound, min(4.0, best_seed[2] + random.uniform(-0.5, 0.5))), 2),
                round(max(1.1, min(6.0, best_seed[3] + random.uniform(-0.5, 0.5))), 2)
            ]

        result = minimize(
            objective_function, x0, method='Nelder-Mead',
            bounds=bounds, options={'maxiter': 500}
        )

        if result.fun < ERROR_THRESHOLD:
            e1, e2, e3, e4 = [round(x, 2) for x in result.x]
            
            if (e4 > e1 and e4 > e3 and
                e1 <= 4.0 and e2 <= 4.0 and e3 <= 4.0 and e4 <= 6.0 and
                e4 / e1 >= 1.1 and e4 / e3 >= 1.1 and 
                e4 / e1 < 4.0 and e4 / e3 < 4.0):
                
                stretch_key = (e1, e2, e3, e4)
                
                if stretch_key not in calc_cache:
                    all_results.append((e1, e2, e3, e4, result.fun))
                    found_count += 1
                elif calc_cache[stretch_key] is None:
                    all_results.append((e1, e2, e3, e4, result.fun))
                    found_count += 1
                    calc_cache[stretch_key] = result.fun
                elif result.fun < calc_cache[stretch_key]:
                    calc_cache[stretch_key] = result.fun
                    all_results.append((e1, e2, e3, e4, result.fun))
                    found_count += 1
                elif stretch_key not in (r[:4] for r in all_results):
                    all_results.append((e1, e2, e3, e4, result.fun))
                    found_count += 1

    # ✅ SỬA: Sắp xếp kết quả mới
    sorted_new_results = sorted(all_results, key=lambda x: x[4])
    
    # ✅ SỬA: Ghép cache cũ + kết quả mới
    all_cached_data = cache_dict['results'] + sorted_new_results
    
    # ✅ SỬA: Loại trùng (ưu tiên error thấp hơn)
    dedup_cached = []
    seen_cached = set()
    
    # Sắp xếp tất cả theo error trước khi loại trùng
    for r in sorted(all_cached_data, key=lambda x: x[4]):
        key = (r[0], r[1], r[2], r[3])
        if key not in seen_cached:
            seen_cached.add(key)
            dedup_cached.append(r)
    
    # ✅ SỬA: Lưu cache (giữ top 500 tốt nhất)
    cache_dict['results'] = sorted(dedup_cached, key=lambda x: x[4])[:500]
    
    cache_dict['explored_regions'].update(tuple(r[:4]) for r in all_results)
    
    # ✅ SỬA CHÍNH: TRẢ VỀ TẤT CẢ KẾT QUẢ (cache + mới), không chỉ sorted_new_results
    return dedup_cached

def optimize_sliver_arrangement(mapping, df_ratios, positions="ABCDEFGH"):
    """
    Tối ưu hóa việc sắp xếp cúi bằng cách đổi chỗ các vị trí có cùng tỷ lệ kéo dãn
    để tránh các cúi cùng màu nằm cạnh nhau.
    """
    pos_list = list(positions)
    n = len(pos_list)
    
    # 1. Xác định các cặp có cùng tỷ lệ kéo dãn (để có thể đổi chỗ cho nhau)
    swappable_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            # Nếu tỷ lệ kéo dãn giống nhau (sai lệch cực nhỏ)
            if abs(df_ratios[pos_list[i]] - df_ratios[pos_list[j]]) < 0.001:
                swappable_pairs.append((pos_list[i], pos_list[j]))
    
    def count_adj_identical(m):
        count = 0
        for i in range(n - 1):
            if m[pos_list[i]] == m[pos_list[i+1]]:
                count += 1
        return count

    best_mapping = mapping.copy()
    min_adj = count_adj_identical(best_mapping)
    
    if min_adj == 0:
        return best_mapping

    # Thử đổi chỗ các cặp có cùng tỷ lệ để giảm số lượng cúi cùng màu cạnh nhau
    improved = True
    while improved:
        improved = False
        for p1, p2 in swappable_pairs:
            # Thử đổi
            test_mapping = best_mapping.copy()
            test_mapping[p1], test_mapping[p2] = test_mapping[p2], test_mapping[p1]
            
            test_adj = count_adj_identical(test_mapping)
            if test_adj < min_adj:
                min_adj = test_adj
                best_mapping = test_mapping
                improved = True
    
    return best_mapping

def parse_elongation_filter(elongation_input):
    if not elongation_input.strip():
        return None, None, None, {}, ""

    elongation_input = elongation_input.strip()
    min_val = None
    max_val = None
    exact_vals = None
    stretch_filters = {}

    try:
        # ← FORMAT MỚI: E1=1.5, E3=3.0, E4=5.0 HOẶC E1:1.5, E3:3.0, E4:5.0
        if 'E' in elongation_input.upper() and ('=' in elongation_input or ':' in elongation_input):
            parts = [p.strip() for p in elongation_input.split(',') if p.strip()]
            parsed_count = 0
            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                elif '=' in part:
                    key, value = part.split('=', 1)
                else:
                    continue
                
                key = key.strip().upper()
                
                if key in ['E1', 'E2', 'E3', 'E4']:
                    try:
                        val = float(value.strip())
                        # ✅ Kiểm tra giới hạn hợp lệ
                        if key in ['E1', 'E2', 'E3'] and (val < 1.1 or val > 4.0):
                            return None, None, None, {}, f"⚠️ {key} phải trong khoảng [1.1, 4.0]. Giá trị hiện tại: {val}"
                        if key == 'E4' and (val < 1.1 or val > 6.0):
                            return None, None, None, {}, f"⚠️ E4 phải trong khoảng [1.1, 6.0]. Giá trị hiện tại: {val}"
                        
                        stretch_filters[key] = val
                        parsed_count += 1
                    except ValueError:
                        return None, None, None, {}, f"⚠️ Không thể parse giá trị '{value}' cho {key}"
            
            if stretch_filters:
                log_msg = f"🎯 Lọc kéo dãn cụ thể: {stretch_filters} ({parsed_count} điều kiện)"
                return None, None, None, stretch_filters, log_msg
            else:
                return None, None, None, {}, "⚠️ Không parse được filter kéo dãn nào. VD đúng: E1:1.3, E2:2.5, E4:5.5"
        
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
                # ← THÊM: Tạo stretch_filters để lọc kết quả
                stretch_filters = {'MAX_E123': max_val, 'MIN_E123': min_val}
                log_msg = f"🔍 Lọc (E1,E2,E3): {min_val} ≤ kéo dãn ≤ {max_val}"
                return min_val, max_val, None, stretch_filters, log_msg
            elif len(parts) == 1:
                max_val = parts[0]
                # ← SỬA: Tạo filter cho TẤT CẢ E1, E2, E3, E4
                stretch_filters = {
                    'MAX_E1': max_val,
                    'MAX_E2': max_val, 
                    'MAX_E3': max_val,
                    'MAX_E4': max_val
                }
                log_msg = f"🔍 Lọc TẤT CẢ kéo dãn (E1, E2, E3, E4) ≤ {max_val}"
                return None, max_val, None, stretch_filters, log_msg
        else:
            max_val = float(elongation_input)
            # ← SỬA: Tạo filter cho TẤT CẢ E1, E2, E3, E4
            stretch_filters = {
                'MAX_E1': max_val,
                'MAX_E2': max_val, 
                'MAX_E3': max_val,
                'MAX_E4': max_val
            }
            log_msg = f"🔍 Lọc TẤT CẢ kéo dãn (E1, E2, E3, E4) ≤ {max_val}"
            return None, max_val, None, stretch_filters, log_msg

        return None, None, None, {}, ""

    except ValueError:
        return None, None, None, {}, "⚠️ Format kéo dãn không hợp lệ. VD: E1:1.3,E2:2.5 hoặc 2.5 hoặc 1.5,3.0"

def adjust_ratios(ratio_dict, num_units=None, blend_input=""):
    """
    Logic điều chỉnh tỷ lệ màu - TỰ ĐỘNG XÁC ĐỊNH SỐ CÚI TÁCH, có Ghép Sơ Bộ.
    ✨ ƯU TIÊN: Giá trị thủ công → Tự động (khi rỗng)
    ✨ TỔNG TRẮNG: Nếu màu lớn nhất là white_key → tính tổng tất cả màu trắng
    ✨ num_units = 0 → KHÔNG TÁCH CÚI
    """
    log = []
    
    mapping_units = {6: 0.25, 5: 0.375, 4: 0.5, 3: 0.625, 2: 0.75, 0: 1.0}
    blend_type_mapping = {
        6: "2/6", 5: "3/5", 4: "4/4", 3: "5/3", 2: "6/2", 0: "8/0"
    }
    
    # 1. Tự động xác định số cúi/tỷ lệ
    def get_units_and_percent(value):
        if value >= 90: return 6, 0.25
        elif value >= 85: return 5, 0.375
        elif value >= 75: return 4, 0.5
        else: return 0, 1.0

    max_color_initial = max(ratio_dict, key=ratio_dict.get)
    max_value = ratio_dict[max_color_initial]
    
    # ✅ ƯU TIÊN: Kiểm tra GIÁ TRỊ THỦ CÔNG trước
    if num_units is not None and str(num_units).strip() not in ["", "None"]:
        # THỦ CÔNG: người dùng chỉ định số cúi (bao gồm cả 0)
        num_units = int(num_units)
        color_percent = mapping_units.get(num_units, 1.0)
        
        if num_units == 0:
            log.append(f"🧮 Cấu trúc thủ công: 8 CÚI HỖN HỢP (KHÔNG TÁCH CÚI)")
        else:
            log.append(f"🧮 Cấu trúc thủ công: {8 - num_units} CÚI HỖN HỢP + {num_units} CÚI TÁCH")
    else:
        # TỰ ĐỘNG: Khi num_units rỗng hoặc None
        if max_color_initial in white_keys:
            # ✨ Nếu màu lớn nhất là WHITE_KEY → tính TỔNG TRẮNG
            total_white = sum(v for k, v in ratio_dict.items() if k in white_keys)
            num_units, color_percent = get_units_and_percent(total_white)
            white_list = [k for k in ratio_dict.keys() if k in white_keys]
            log.append(f"🤖 Tự động xác định (TỔNG TRẮNG): {8 - num_units} CÚI HỖN HỢP + {num_units} CÚI TÁCH")
            log.append(f"   📊 Tổng trắng: {total_white:.2f}% (gồm: {', '.join(white_list)})")
        else:
            # Màu thường → dùng giá trị của màu lớn nhất
            num_units, color_percent = get_units_and_percent(max_value)
            log.append(f"🤖 Tự động xác định: {8 - num_units} CÚI HỖN HỢP + {num_units} CÚI TÁCH (dựa trên {max_color_initial}={max_value}%)")

    blend_type = blend_type_mapping.get(num_units, "8/0") 
    
    excluded_colors = set()
    
    # 2. TÁCH CÚI 12.5% VÀ TÍNH Adjusted Ratio (temp_adjusted)
    multiples_12_5 = {} 
    temp_adjusted = {} 
    total_units_allocated = 0

    # ✅ CHỈ TÁCH CÚI KHI num_units > 0
    if num_units > 0:
# ✨ LOGIC MỚI: Ưu tiên max_color_initial trước, sau đó bội 12.5
        
        # Bước 1: Phân loại màu theo 4 nhóm
        max_color_group = []     # Màu lớn nhất — ưu tiên tuyệt đối
        white_multiples = []     # Trắng bội 12.5 (trừ max_color_initial)
        color_multiples = []     # Thường bội 12.5
        large_colors = []        # Có available > 0 nhưng không bội 12.5

        for color, ratio in ratio_dict.items():
            available_units = int(ratio / 12.5)
            if available_units == 0:
                continue
            is_multiple = abs(ratio - (available_units * 12.5)) < 0.1

            if color == max_color_initial:
                max_color_group.append((color, ratio, available_units))
            elif is_multiple:
                if color in white_keys:
                    white_multiples.append((color, ratio, available_units))
                else:
                    color_multiples.append((color, ratio, available_units))
            else:
                large_colors.append((color, ratio, available_units))

        # Sắp xếp mỗi nhóm theo tỷ lệ giảm dần
        white_multiples.sort(key=lambda x: -x[1])
        color_multiples.sort(key=lambda x: -x[1])
        large_colors.sort(key=lambda x: -x[1])

        # Bước 2: Phân bổ cúi theo thứ tự ưu tiên
        priority_list = max_color_group + white_multiples + color_multiples + large_colors
        
        for color, ratio, available_units in priority_list:
            if total_units_allocated >= num_units:
                break
            
            units_to_take = min(num_units - total_units_allocated, available_units)
            
            if units_to_take > 0:
                multiples_12_5[color] = units_to_take
                total_units_allocated += units_to_take

        # 2.2 Tính Adjusted Ratio (với tách cúi)
        for color, ratio in ratio_dict.items():
            units = multiples_12_5.get(color, 0)
            cui_ratio = units * 12.5
            remaining_ratio = ratio - cui_ratio
            
            if remaining_ratio > 0.01:
                adjusted_ratio = round(remaining_ratio / color_percent, 2)
                temp_adjusted[color] = adjusted_ratio
        
        # Loại bỏ các màu đã bị tách HẾT
        final_temp_adjusted = {k: v for k, v in temp_adjusted.items() if v >= 0.01}
    else:
        # ✅ KHÔNG TÁCH CÚI: Giữ nguyên tỷ lệ ban đầu
        final_temp_adjusted = ratio_dict.copy()
        # excluded_colors.add(max_color_initial)
    
    total_after_adj = sum(final_temp_adjusted.values())
    
    # 3. XỬ LÝ GHÉP SƠ BỘ (BLEND)
    blended_colors = [] 

    if blend_input.strip():
        # ✅ CHỈ XỬ LÝ KHI CÓ INPUT
        final_temp_adjusted, blend_log, genhuyu_key, blended_colors = process_blend_after_split(
            final_temp_adjusted, blend_input, None
        )
        
        if blend_log:
            log.append(blend_log)
        
        total_after_adj = sum(final_temp_adjusted.values()) 
        
        log.append(f"📊 Tỷ lệ sau ghép sơ bộ:")
        for color, val in sorted(final_temp_adjusted.items(), key=lambda x: -x[1]):
            log.append(f"    {color}: {val:.2f}%")
        log.append(f"    ➡️ Tổng: {total_after_adj:.2f}%")
    else:
        # ✅ KHÔNG CÓ INPUT → BỎ QUA GHÉP SƠ BỘ
        total_after_adj = sum(final_temp_adjusted.values())

    # 4. CHUẨN HÓA VÀ TÍNH adjusted_full_ratios
    removed_color = None
    total_after_check = total_after_adj
    
    # 4.1 Chuẩn hóa nếu Tổng > 100%
    if total_after_check > 100.01:
        excess = round(total_after_check - 100.0, 2)
        log.append(f"⚠️ Tổng vượt quá 100: {total_after_check:.2f}, dư: {excess:.2f}")
        
        candidates = {k: v for k, v in final_temp_adjusted.items() if v >= excess}
        
        if candidates:
            removed_color_candidate = min(candidates, key=candidates.get) 
        else:
            removed_color_candidate = max(final_temp_adjusted, key=final_temp_adjusted.get)
            
        removed_val = final_temp_adjusted.pop(removed_color_candidate)
        total_after_check -= removed_val
        
        excluded_colors.add(removed_color_candidate) 
        removed_color = removed_color_candidate

    # 4.2 CƠ SỞ TÍNH LỖI (adjusted_full_ratios)
    adjusted_full_ratios = final_temp_adjusted.copy()
    remaining_percent = round(100.00 - sum(adjusted_full_ratios.values()), 2)
    
    # 4.3 Bù phần dư vào màu FILL
    if abs(remaining_percent) > 0.01:
        fill_color_target = None
        
        if removed_color:
            fill_color_target = removed_color
        else:
            fill_color_target = max_color_initial
            
        if fill_color_target:
            current_fill_ratio = adjusted_full_ratios.get(fill_color_target, 0.0)
            adjusted_full_ratios[fill_color_target] = round(current_fill_ratio + remaining_percent, 2)

    # 4.4 Đảm bảo tất cả màu ban đầu có mặt
    for color in ratio_dict.keys():
        if color not in adjusted_full_ratios:
            adjusted_full_ratios[color] = 0.0
        
        units = multiples_12_5.get(color, 0)
        if units > 0 and ratio_dict[color] - units * 12.5 <= 0.01:
            excluded_colors.add(color)
    
    # Loại màu đã ghép sơ bộ
    excluded_colors_ratios = {}
    for color in excluded_colors:
        if color in adjusted_full_ratios:
            excluded_colors_ratios[color] = adjusted_full_ratios[color]
    
    for color in blended_colors:
        if color not in final_temp_adjusted:
            if color in adjusted_full_ratios:
                adjusted_full_ratios.pop(color)
    
    # Chỉ loại màu excluded nếu tỷ lệ = 0 (đã tách cúi hết)
    for color in list(excluded_colors):
        if color in adjusted_full_ratios and adjusted_full_ratios[color] <= 0.01:
            adjusted_full_ratios.pop(color)
    
    # LOG Hỗn I
    if num_units > 0:
        mix_parts = [f"{8 - num_units}根精梳"]
        for color, units in sorted(multiples_12_5.items()):
            mix_parts.append(f"{units}{color}")
        log.append(f"混一: {' + '.join(mix_parts)}")
    else:
        log.append(f"混一: 8根精梳")
    
    log.append(f"📊 Tỷ lệ sau tách cúi:")
    for color, val in sorted(final_temp_adjusted.items(), key=lambda x: -x[1]):
        log.append(f"    • {color}: {val:.2f}%")
    log.append(f"    ➡️ Tổng: {total_after_adj:.2f}%")
    
    # ✅ THÊM: Xác định màu fill (màu có tỷ lệ lớn nhất trong adjusted_full_ratios)
    fill_color_for_matching = None
    if adjusted_full_ratios:
        fill_color_for_matching = max(adjusted_full_ratios, key=adjusted_full_ratios.get)

    # Trả về kết quả
    return (
        final_temp_adjusted,      # Tỷ lệ sau tách cúi
        color_percent,            # Hệ số điều chỉnh
        log,                      # Log messages
        max_color_initial,        # Màu lớn nhất ban đầu
        None,                     # max_white_color (deprecated)
        total_after_check,        # Tổng sau chuẩn hóa
        excluded_colors,          # Set màu bị loại
        False,                    # no_split_major (deprecated)
        max_color_initial,        # max_color (duplicate)
        adjusted_full_ratios,     # Tỷ lệ đầy đủ (cho so sánh)
        excluded_colors_ratios,   # Tỷ lệ của màu bị loại
        fill_color_for_matching   # Màu fill (cho matching)
    )
    """Xử lý ghép sơ bộ KHÔNG tách cúi"""
def process_blend_no_split(ratio_dict, blend_input):
    try:
        if not blend_input.strip():
            return ratio_dict, "", "", []  # ✅ THÊM []
        
        parts = re.split(r'[+,]', blend_input.strip())
        blend_colors = []
        blend_units_input = {}  # ← LƯU SỐ LƯỢNG CÚI TỪ INPUT
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            match = re.match(r'(\d+)?([A-Za-z0-9_\-\(\)\u4e00-\u9fff]+)', part)
            if match:
                count_str, color = match.groups()
                color = color.upper()
                if color in ratio_dict:
                    blend_colors.append(color)
                    # ✅ LƯU SỐ LƯỢNG CÚI NẾU CÓ
                    if count_str:
                        blend_units_input[color] = int(count_str)
        
        if len(blend_colors) < 2:
            return ratio_dict, "", "", []  # ✅ THÊM []
        
        # Tính tổng
        total = sum(ratio_dict.get(c, 0) for c in blend_colors)
        
        # Tạo 根混预
        new_dict = {}
        for color, ratio in ratio_dict.items():
            if color not in blend_colors:
                new_dict[color] = round(ratio, 2)
        
        new_dict["根混预"] = round(total, 2)
        
        # ✅ TẠO LOG VỚI SỐ LƯỢNG CÚI
        log_parts = []
        if blend_units_input:
            # Nếu người dùng nhập số lượng → dùng input
            for color in blend_colors:
                if color in blend_units_input:
                    log_parts.append(f"{blend_units_input[color]}{color}")
                else:
                    # Nếu không có số → tính từ tỷ lệ
                    units = round(ratio_dict.get(color, 0) / 12.5)
                    log_parts.append(f"{units}{color}")
        else:
            # Nếu không có số nào → chỉ hiển thị tên màu
            log_parts = blend_colors
        
        blend_log = f"混预: {' + '.join(log_parts)}"
        return new_dict, blend_log, "根混预", blend_colors  # ✅ THÊM blend_colors
        
    except Exception as e:
        print(f"⚠️ Lỗi xử lý ghép sơ bộ không tách cúi: {e}")
        return ratio_dict, "", "", []  # ✅ THÊM []
def format_float_keep_one_decimal(x):
    s = f"{x:.2f}"
    return s
def process_blend_after_split(adjusted_dict, blend_input, blend_type=None):
    """Xử lý ghép sơ bộ SAU KHI đã tách cúi"""
    try:
        if not blend_input.strip():
            return adjusted_dict, "", "", []
        
        blend_ratios = {
            "1/7": 0.125,
            "2/6": 0.25, 
            "3/5": 0.375, 
            "4/4": 0.5, 
            "5/3": 0.625, 
            "6/2": 0.75,
            "7/1": 0.875
        }
        
        parts = re.split(r'[+,]', blend_input.strip())
        blend_colors = []
        blend_units_input = {}
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            match = re.match(r'(\d+)?([A-Za-z0-9_\-\(\)\u4e00-\u9fff]+)', part)
            if match:
                count_str, color = match.groups()
                color = color.upper()
                
                if color in adjusted_dict:
                    blend_colors.append(color)
                    if count_str:
                        blend_units_input[color] = int(count_str)
        
        if not blend_colors:
            return adjusted_dict, "", "", []
        
        # ✅ LOGIC: Xác định blend_type từ INPUT
        detected_blend_type = None
        
        if blend_units_input and len(blend_colors) >= 1:
            first_color = blend_colors[0]
            first_units = blend_units_input.get(first_color, None)
            
            if first_units is not None:
                total_units = sum(blend_units_input.get(c, 0) for c in blend_colors)
                
                if total_units == 8:
                    blend_type_map = {
                        1: "1/7",
                        2: "2/6",
                        3: "3/5",
                        4: "4/4",
                        5: "5/3",
                        6: "6/2",
                        7: "7/1",
                    }
                    detected_blend_type = blend_type_map.get(first_units, None)
        
        # Ưu tiên: detected > parameter > default
        if detected_blend_type:
            blend_type = detected_blend_type
        elif blend_type is None:
            blend_type = "2/6"

        divisor = blend_ratios.get(blend_type, 0.25)
        
        first_blend_color = blend_colors[0]
        first_color_ratio = adjusted_dict.get(first_blend_color, 0)
        
        genhuyu_ratio = round(first_color_ratio / divisor, 2)
        total_blend_base = sum(adjusted_dict.get(color, 0) for color in blend_colors)
        remaining_ratio = round(total_blend_base - genhuyu_ratio, 2)
        
        # Tạo log
        log_parts = []
        if blend_units_input:
            for color in blend_colors:
                if color in blend_units_input:
                    log_parts.append(f"{blend_units_input[color]}{color}")
                else:
                    units = round(adjusted_dict.get(color, 0) / 12.5)
                    log_parts.append(f"{units}{color}")
        else:
            for color in blend_colors:
                units = round(adjusted_dict.get(color, 0) / 12.5)
                log_parts.append(f"{units}{color}")
        
        blend_log = f"混预: {' + '.join(log_parts)}"
        
        # Cập nhật dict
        new_dict = {}
        for color, ratio in adjusted_dict.items():
            if color not in blend_colors:
                new_dict[color] = round(ratio, 2)
        
        new_dict["根混预"] = genhuyu_ratio
        
        if remaining_ratio > 0:
            last_blend_color = blend_colors[-1]
            new_dict[last_blend_color] = remaining_ratio
        
        return new_dict, blend_log, "根混预", blend_colors
        
    except Exception as e:
        print(f"⚠️ Lỗi xử lý ghép sơ bộ: {e}")
        import traceback
        traceback.print_exc()
        return adjusted_dict, "", "", []
def match_colors_to_calculated_ratios(color_ratios, calc_ratios, tolerance=2.0,
                                      excluded_colors=None, adjusted_full_ratios=None,
                                      priority_colors=None, excluded_colors_ratios=None,
                                      fill_color_for_matching=None):
    """
    Match màu với tỷ lệ A-H (ĐÃ SỬA LOGIC LÀM TRÒN).
    Nhận calc_ratios (A-H) ở độ chính xác cao (chưa làm tròn).
    Chỉ làm tròn TỔNG CUỐI CÙNG (actual_val) trước khi so sánh.
    """
    
    if excluded_colors is None:
        excluded_colors = set()
    if priority_colors is None:
        priority_colors = []

    mapping = {}
    used_cuis = set()
    used_colors = set()

    # ✅ THÊM: Loại màu excluded ra khỏi color_ratios trước khi matching
    color_ratios_for_matching = {k: v for k, v in color_ratios.items() 
                                 if k not in excluded_colors}

    # ✅ THÊM: Nếu có fill_color_for_matching, loại nó ra khỏi matching
    if fill_color_for_matching and fill_color_for_matching in color_ratios_for_matching:
        color_ratios_for_matching.pop(fill_color_for_matching)

    # ✅ ĐỔI: Dùng color_ratios_for_matching thay vì color_ratios
    all_colors = sorted(color_ratios_for_matching.items(), key=lambda x: -x[1])

    available_cuis = list('ABCDEFGH')
    combo_2 = list(combinations(available_cuis, 2))
    combo_3 = list(combinations(available_cuis, 3))

    for color, val in all_colors:
        # Kiểm tra nếu màu đã dùng đủ lần
        color_count_needed = 1
        color_count_used = sum(1 for c in mapping.values() if c == color)
        
        # Màu trắng có thể dùng nhiều lần
        if color_count_used >= color_count_needed and color in white_keys:
            pass
        elif color in used_colors and color not in white_keys:
            continue
            
        best_cui = None
        min_diff = tolerance + 1

        # Thử match với 1 cúi
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
                # Không tìm thấy match cho màu này
                return None

    # ✅ SỬA: Điền màu fill cho các vị trí còn trống
    # Ưu tiên dùng fill_color_for_matching nếu có
    if fill_color_for_matching:
        fill_color = fill_color_for_matching
    else:
        fill_color = next(iter(excluded_colors)) if excluded_colors else "W"
    
    for cui in available_cuis:
        if cui not in used_cuis:
            mapping[cui] = fill_color

    # TÍNH TOÁN SAI SỐ
    actual_by_color = defaultdict(float)
    for cui, color in mapping.items():
        actual_by_color[color] += calc_ratios[cui]

    color_errors = {}
    total_error = 0
    priority_error = 0

    ratios_to_check = adjusted_full_ratios if adjusted_full_ratios else color_ratios

    for color in ratios_to_check:
        expected_val = ratios_to_check[color]
        
        actual_val_high_precision = actual_by_color.get(color, 0.0) 
        
        # Làm tròn tổng (giống Excel) trước khi so sánh
        actual_val_rounded = excel_round(actual_val_high_precision, 2) 
        
        diff = abs(actual_val_rounded - expected_val)
        
        color_errors[color] = {
            'expected': expected_val,
            'actual': actual_val_rounded,
            'error': diff
        }
        total_error += diff
        
        if priority_colors and color in priority_colors:
            priority_error += diff

    # Tính sai số cho màu excluded (màu fill)
    if excluded_colors and excluded_colors_ratios:
        for fill_color_check in excluded_colors:
            expected_fill = excluded_colors_ratios.get(fill_color_check, 0)
            if expected_fill <= 0.01:
                continue
            
            actual_fill_high_precision = actual_by_color.get(fill_color_check, 0.0)
            actual_fill_rounded = excel_round(actual_fill_high_precision, 2)
            
            diff_fill = abs(actual_fill_rounded - expected_fill)
            
            color_errors[fill_color_check] = {
                'expected': expected_fill,
                'actual': actual_fill_rounded,
                'error': diff_fill
            }
            total_error += diff_fill
            
            if priority_colors and fill_color_check in priority_colors:
                priority_error += diff_fill

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

    # --- Tối ưu hóa sắp xếp cúi để tránh trùng màu cạnh nhau ---
    mapping = optimize_sliver_arrangement(mapping, df_ratios)

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
    if not color_ratios.strip():
        return ""

    ratio_lines = [line.strip() for line in color_ratios.strip().split("\n") if line.strip()]
    if not color_names.strip():
        auto_names = [chr(65 + i) for i in range(len(ratio_lines))]  
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
    """Preview tỷ lệ màu - hiện warning + thông tin nếu số lượng không khớp"""
    if not color_ratios.strip():
        return ""
    
    ratio_lines = [line.strip() for line in color_ratios.strip().split("\n") if line.strip()]
    name_lines = [line.strip() for line in color_names.strip().split("\n") if line.strip()] if color_names.strip() else []
    
    log = ["📥 Tỷ lệ màu đã nhập:"]
    warning = ""
    
    # Kiểm tra số lượng
    if name_lines and len(name_lines) != len(ratio_lines):
        warning = f"⚠️ Số lượng tên màu ({len(name_lines)}) khác với số tỷ lệ ({len(ratio_lines)})"
    
    # Parse và hiển thị tất cả ratio có thể
    total = 0
    pattern = re.compile(r"^([0-9]+(?:[.,][0-9]+)?)\s*%?\s*$")
    
    for i, ratio_str in enumerate(ratio_lines):
        try:
            m = pattern.match(ratio_str)
            if m:
                val = float(m.group(1).replace(",", "."))
            else:
                val = float(ratio_str.replace(",", "."))
            
            # Lấy tên màu nếu có
            if i < len(name_lines):
                color_name = name_lines[i].upper()
            else:
                color_name = chr(65 + i)  # A, B, C...
            
            log.append(f"- {color_name}: {val:.2f}%")
            total += val
        except:
            log.append(f"- ???: {ratio_str} (lỗi)")
    
    log.append(f"🎯 Tổng cộng: {total:.2f}%")
    
    missing = 100.0 - total
    if missing > 0.1:
        log.append(f"⚠️ Tỉ lệ còn thiếu: {missing:.2f}%")
    elif missing < -0.1:
        log.append(f"⚠️ Vượt quá: {-missing:.2f}%")
    
    # Thêm warning vào đầu nếu có
    if warning:
        log.insert(1, warning)
    
    return "\n".join(log)
def check_stretch_filter(mapping_str, stretch_filters):
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
        
        # ← THÊM: Kiểm tra MAX_E123 (từ input "2.5")
        if 'MAX_E123' in stretch_filters:
            max_val = stretch_filters['MAX_E123']
            if e1 > max_val or e2 > max_val or e3 > max_val:
                return False
        
        # ← THÊM: Kiểm tra MIN_E123 (từ input "1.5,3.0")
        if 'MIN_E123' in stretch_filters:
            min_val = stretch_filters['MIN_E123']
            if e1 < min_val or e2 < min_val or e3 < min_val:
                return False
        
        for key, expected_val in stretch_filters.items():
            if key in ['MAX_E123', 'MIN_E123']:
                continue  
            
            # ✅ THÊM: Kiểm tra MAX cho từng kéo dãn riêng lẻ
            if key == 'MAX_E1' and e1 > expected_val:
                return False
            elif key == 'MAX_E2' and e2 > expected_val:
                return False
            elif key == 'MAX_E3' and e3 > expected_val:
                return False
            elif key == 'MAX_E4' and e4 > expected_val:
                return False
            # Logic cũ cho E1/E2/E3/E4 exact
            elif key == 'E1' and abs(e1 - expected_val) > tolerance:
                return False
            elif key == 'E2' and abs(e2 - expected_val) > tolerance:
                return False
            elif key == 'E3' and abs(e3 - expected_val) > tolerance:
                return False
            elif key == 'E4' and abs(e4 - expected_val) > tolerance:
                return False
        
        return True
    except Exception as e:
        print(f"⚠️ check_stretch_filter error: {e} | mapping_str={mapping_str}")
        return True

def check_arrangement_filter(result, arrangement_filters):
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
def extract_stretches_from_mapping(mapping_str):
    try:
        if '(' not in mapping_str or ')' not in mapping_str:
            return None
        
        stretch_part = mapping_str[mapping_str.rindex('(') + 1:mapping_str.rindex(')')]
        stretch_values = [float(x.strip()) for x in stretch_part.split('/')]
        
        if len(stretch_values) == 4:
            return stretch_values
        return None
    except:
        return None
def render_result_table(results, page, page_size=10,show_debug=False):
    if results and len(results) > 0:
        r = results[0]  # Check result đầu tiên
        color_errors = r.get('ColorErrors') or r.get('color_errors', {})
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
            # "Row": r.get("Row", "-"),  # ← XÓA DÒNG NÀY
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

    all_colors = set()
    for row in data:
        all_colors.update([
            k for k in row.keys()
            if k not in ["STT", "Tên SP", "Sai số", "Sai số ƯT", "Sắp cúi"]  
        ])

    columns_order = ["STT", "Tên SP"] + sorted(all_colors) + ["Sai số", "Sai số ƯT", "Sắp cúi"]  

    df_result = pd.DataFrame(data)
    for col in all_colors:
        if col not in df_result.columns:
            df_result[col] = ""
        else:
            df_result[col] = df_result[col].fillna("")

    df_result = df_result[columns_order]
    return df_result.to_markdown(index=False)
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
def prev_page(results, current, page_size=10):
    if current <= 0:
        return render_result_table(results, 0, page_size), 0
    return render_result_table(results, current - 1, page_size), current - 1

def next_page(results, current, page_size=10):
    max_page = len(results) // page_size
    if current + 1 > max_page:
        return render_result_table(results, current, page_size), current
    return render_result_table(results, current + 1, page_size), current + 1
def first_page(results, page_size=10):
    return render_result_table(results, 0, page_size), 0

def last_page(results, page_size=10):
    max_page = max(0, (len(results) - 1) // page_size)
    return render_result_table(results, max_page, page_size), max_page

def goto_page(results, page_num, page_size=10):
    if not results:
        return render_result_table(results, 0, page_size), 0, "1"
    
    max_page = max(0, (len(results) - 1) // page_size)
    
    # Validate page number
    try:
        page_num = int(page_num)
        if page_num < 1:
            page_num = 1
        elif page_num > max_page + 1:
            page_num = max_page + 1
    except (ValueError, TypeError):
        page_num = 1
    
    # Convert to 0-indexed
    page_index = page_num - 1
    return render_result_table(results, page_index, page_size), page_index, str(page_num)
# ===== THAY THẾ HÀM run_app CŨ =====
def run_app(product_name, color_names, color_ratios, num_units, elongation_limit,
            priority_input, arrangement_filter_input, blend_input="", 
            current_iteration=0, cache_dict=None):
    """
    Hàm chính - CHỈ DÙNG OPTIMIZATION SCIPY
    Trả về 7 giá trị: (log, structure_line, table, results, page, iteration, cache)
    """
    
    # XÓA CACHE CẤP THẤP (FAILSAFE)
    try:
        calculate_ratios_from_stretches.cache_clear()
        print("INFO: @lru_cache (nếu có) đã được xóa.")
    except Exception as e:
        pass 

    if cache_dict is None:
        cache_dict = {'results': [], 'explored_regions': set(), 'best_seeds': []}
    
    log = []
    start_time = time.time()

    current_iteration += 1
    
    if current_iteration == 1:
        log.append("⚡ Chế độ: TÌM NHANH (500 vòng)")
    elif current_iteration == 2:
        log.append(f"🎯 Chế độ: TỐI ƯU THÊM (Cache: {len(cache_dict.get('results', []))} kết quả)")
    else:
        log.append(f"🔥 Chế độ: TÌM SÂU (Cache: {len(cache_dict.get('results', []))} kết quả)")

    try:
        color_input = combine_color_inputs(color_names, color_ratios)
        if color_input.startswith("⚠️"):
            return color_input, "", "", [], 0, current_iteration, cache_dict
        if not color_input.strip():
            return "⚠️ Vui lòng nhập tên màu và tỷ lệ màu.", "", "", [], 0, current_iteration, cache_dict

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
                return f"⚠️ Sai định dạng ở dòng: '{line}'", "", "", [], 0, current_iteration, cache_dict

        total_ratio = sum(ratios.values())
        if abs(total_ratio - 100.0) > 0.0001:  # Cho phép sai số ±0.01%
            return f"⚠️ Tổng tỷ lệ phải gần bằng 100.00%. Hiện tại: {total_ratio:.2f}%", "", "", [], 0, current_iteration, cache_dict
        if len(ratios) < 2:
            return "⚠️ Cần ít nhất 2 màu để tra cứu.", "", "", [], 0, current_iteration, cache_dict
        
        arrangement_filters = parse_arrangement_to_positions(arrangement_filter_input)
        if arrangement_filters:
            log.append(f"🎯 Điều kiện lọc sắp xếp: {arrangement_filters}")

        priority_colors = [s.strip().upper() for s in priority_input.split(",") if s.strip()] if priority_input else []
        if priority_colors:
            log.append(f"🔍 Màu ưu tiên sai số: {priority_colors}")

        # Xử lý num_units
        num_units = int(num_units) if num_units else None
        
        # ✅ SỬA: Nhận thêm fill_color_for_matching từ adjust_ratios
        adjusted_ratios, color_percent, adjust_log, max_color_initial, \
            max_white_color, total_after, excluded_colors, \
            no_split_major, max_color, adjusted_full_ratios, excluded_colors_ratios, \
            fill_color_for_matching = adjust_ratios(ratios, num_units, blend_input)

        log.extend(adjust_log)

        # Tạo structure_line
        structure_line = get_structure_line_from_textbox(str(num_units)) if num_units else ""

        min_val, max_val, exact_vals, stretch_filters, elongation_log = parse_elongation_filter(elongation_limit)
        if elongation_log and not elongation_log.startswith("⚠️"):
            log.append(elongation_log)

        stretch_bounds = (1.1, 6.0)
        if min_val and max_val:
            stretch_bounds = (min_val, max_val)
        elif max_val:
            stretch_bounds = (1.1, max_val)

        # ✅ SỬA: Truyền fill_color_for_matching vào optimization
        stretch_candidates = find_optimal_stretches_scipy(
            adjusted_ratios,
            adjusted_full_ratios,
            excluded_colors,
            stretch_bounds=stretch_bounds,
            priority_colors=priority_colors,
            iteration=current_iteration,
            use_cache=(current_iteration > 1),
            cache_dict=cache_dict,
            excluded_colors_ratios=excluded_colors_ratios,
            fill_color_for_matching=fill_color_for_matching
        )

        if not stretch_candidates:
            elapsed_time = time.time() - start_time
            return "\n".join(log + [f"❌ Không tìm thấy kết quả phù hợp. ⏱️ Thời gian: {elapsed_time:.2f}s"]), structure_line, "", [], 0, current_iteration, cache_dict

        log.append(f"✅ Tìm thấy {len(stretch_candidates)} ứng viên từ optimization")

        results = []
        for e1, e2, e3, e4, error in stretch_candidates:
            calc_ratios = calculate_ratios_from_stretches(e1, e2, e3, e4)
            if calc_ratios is None:
                continue

            # ✅ SỬA: Truyền fill_color_for_matching vào matching
            match_result = match_colors_to_calculated_ratios(
                adjusted_ratios, calc_ratios, tolerance=2.0,
                excluded_colors=excluded_colors,
                adjusted_full_ratios=adjusted_full_ratios,
                priority_colors=priority_colors,
                excluded_colors_ratios=excluded_colors_ratios,
                fill_color_for_matching=fill_color_for_matching
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

        if stretch_filters:
            before_count = len(results)
            results = [r for r in results 
                       if check_stretch_filter(r.get("Mapping", ""), stretch_filters)]
            log.append(f"📉 Sau lọc kéo dãn: {before_count:,} → {len(results):,}")

        # Sắp xếp kết quả
        results = sorted(results, key=lambda x: (
            x.get("Sai số ưu tiên", 0),                 
            x.get("total_error") or x.get("Sai số", 0),    
            calculate_total_stretch(x.get("Mapping", "")),
            calculate_stretch_variance(x.get("Mapping", ""))       
        ))
        
        # Loại trùng
        unique_results = []
        seen_keys = set()
        
        for r in results:
            mapping_str = r.get("Mapping", "")
            e_values = extract_stretches_from_mapping(mapping_str)
            
            if e_values:
                sorted_e_key = tuple(round(v, 3) for v in e_values)
                unique_key = (
                    sorted_e_key, 
                    round(r.get("total_error", 0), 2),
                    round(r.get("Sai số ưu tiên", 0), 2)
                )
                
                if unique_key not in seen_keys:
                    seen_keys.add(unique_key)
                    unique_results.append(r)
            else:
                mapping_key = r.get("Mapping", "")
                if mapping_key not in seen_keys:
                    seen_keys.add(mapping_key)
                    unique_results.append(r)
                    
        results = unique_results 
        
        for r in results:
            r["ProductName"] = product_name

        if not results:
            elapsed_time = time.time() - start_time
            return "\n".join(log + [f"❌ Không tìm thấy kết quả phù hợp sau lọc. ⏱️ Thời gian: {elapsed_time:.2f}s"]), structure_line, "", [], 0, current_iteration, cache_dict
        elapsed_time = time.time() - start_time
        log.append(f"📝 Cache Status: Lần {current_iteration} | Cache có {len(cache_dict.get('results', []))} kết quả | Explored: {len(cache_dict.get('explored_regions', set()))} vùng")

        first_page_table = render_result_table(results, 0)
        return "\n".join(log), structure_line, first_page_table, results, 0, current_iteration, cache_dict

    except Exception as e:
        import traceback
        elapsed_time = time.time() - start_time
        return f"⚠️ Lỗi: {str(e)}\n⏱️ Thời gian: {elapsed_time:.2f}s\n\n{traceback.format_exc()}", "", "", [], 0, current_iteration, cache_dict
def clear_all_except_colors(color_names, color_ratios):
    """Xóa hết trừ màu và tỷ lệ"""
    return (
        color_names,           # Giữ nguyên tên màu
        color_ratios,          # Giữ nguyên tỷ lệ
        "",                    # num_units
        "",                    # elongation_limit
        "",                    # priority_color
        "",                    # arrangement_filter
        "",                    # blend_input
        "",                    # product_name
        "",                    # realtime_log
        "",                    # structure_line
        "",                    # arrangement_filter_preview
        "",                    # log_output
        "",                    # table_output
        [],                    # results_state
        [],                    # original_results_state
        0,                     # current_page
        0,                     # iteration_counter
        "",                    # last_color_input
        "",                    # last_filters_state
        {'results': [], 'explored_regions': set(), 'best_seeds': []},  # cache_state
        "",                    # page_info
        "1",                   # page_input
        gr.update(visible=False),  # pagination_row
        gr.update(visible=False)   # page_info visibility
    )
# ===== THAY THẾ HÀM get_four_mg_stretch_app() =====
def get_four_mg_stretch_app():
    with gr.Blocks(css=RESPONSIVE_CSS) as app:
        gr.Markdown("<h2 style='text-align: center;'>🎨 PHỐI MÀU CÔNG NGHỆ GHÉP 5.0 </h2>")
        search_mode_state = gr.State("optimization")
        
        product_name_input = gr.Textbox(
            label="📦 Tên sản phẩm",
            placeholder="VD: ABC-123",
            value=""
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                color_names_input = gr.Textbox(
                    lines=4,
                    label="🎨 Mã màu",
                    placeholder="G004\nG024\nXX"
                )
                blend_input = gr.Textbox(
                    label="🔗 Ghép sơ bộ (根混预)",
                    placeholder="VD: 2FW + 6W hoặc FW, W",
                    value=""
                )
                num_units_input = gr.Textbox(
                    label="🔹 Số cúi ghép cần tách (2-6, tùy chọn)",
                    placeholder="VD: 3" 
                )
                clear_btn = gr.Button("🗑️ Xóa", variant="secondary")
                
            with gr.Column(scale=2):
                color_ratios_input = gr.Textbox(
                    lines=2,
                    label="📊 Tỷ lệ (%)",
                    placeholder="18.0\n40.0\n42.0"
                )
                realtime_log = gr.Textbox(
                    label="📥 Hiển thị tỷ lệ màu đã nhập",
                    lines=2,
                    interactive=False
                )
                structure_line = gr.Textbox(
                    label="🧱 Cấu trúc tương ứng",
                    interactive=False
                )
                run_btn = gr.Button("🔍 Tra cứu", variant="primary")
            with gr.Column(scale=3):
                elongation_limit_input = gr.Textbox(
                    label="🧪 Lọc 4 chỉ số kéo giãn",
                    placeholder="VD: 2.5 hoặc 1.5,3.0 (khoảng) hoặc exact:1.5,1.3,2.5"
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
                priority_color_input = gr.Textbox(
                    label="🎯 Màu ưu tiên sai số",
                    placeholder="VD: G004, G024"
                )
                
            with gr.Column(scale=4):
                log_output = gr.Textbox(
                    label="📋 Thông tin xử lý",
                    lines=15,
                    interactive=False
                )

        table_output = gr.Markdown(label="📊 Kết quả")
        results_state = gr.State([])
        original_results_state = gr.State([])  
        current_page = gr.State(0)
        iteration_counter = gr.State(0)
        last_color_input = gr.State("")
        last_filters_state = gr.State("")  
        cache_state = gr.State({
            'results': [],
            'explored_regions': set(),
            'best_seeds': []
        })
        
        # ← ĐỊNH NGHĨA pagination_row Ở ĐÂY
        with gr.Row(visible=False) as pagination_row:
            first_btn = gr.Button("⏮️ Trang đầu")
            prev_btn = gr.Button("⬅️ Trang trước")
            page_input = gr.Textbox(
                value="1",
                container=False,
                min_width=60,
                max_lines=1,
                scale=0,
                elem_classes="page-input-sm"
            )
            next_btn = gr.Button("➡️ Trang sau")
            last_btn = gr.Button("⏭️ Trang cuối")
        
        page_info = gr.Markdown("", visible=False)
        def auto_reset_on_color_change(color_names, color_ratios, num_units, blend_input, 
                                        last_input, current_iteration, cache_dict):
            """
            Tự động reset cache khi thay đổi màu, tỷ lệ, tách cúi, hoặc ghép sơ bộ
            """
            # ✅ Tạo key bao gồm TẤT CẢ thông tin ảnh hưởng đến adjusted_ratios
            current_input = f"{color_names.strip()}|{color_ratios.strip()}|{num_units}|{blend_input.strip()}"
            
            # Nếu input khác với lần trước → reset
            if current_input != last_input and last_input != "":
                # Reset cache của session này
                cache_dict = {'results': [], 'explored_regions': set(), 'best_seeds': []}
                iteration_reset = 0
            else:
                iteration_reset = current_iteration  
            
            # Preview tỷ lệ màu
            preview = preview_combined_ratios(color_names, color_ratios)
            
            return preview, iteration_reset, current_input, cache_dict
        # ===== HÀM FILTER NHANH =====
        def quick_filter_results(original_results, arrangement_filter_input, elongation_limit, last_filters):
            if not original_results:
                return None, "", last_filters  # ← Trả về 3 giá trị
            
            current_filters = f"{arrangement_filter_input.strip()}|{elongation_limit.strip()}"
            
            if current_filters == last_filters:
                return None, "", current_filters  # ← Trả về 3 giá trị
            
            log = []
            log.append(f"🔄 Đang lọc lại {len(original_results):,} kết quả có sẵn...")
            
            arrangement_filters = parse_arrangement_to_positions(arrangement_filter_input)
            _, _, _, stretch_filters, _ = parse_elongation_filter(elongation_limit)
            
            filtered_results = []
            
            for r in original_results:
                if arrangement_filters and not check_arrangement_filter(r, arrangement_filters):
                    continue
                
                if stretch_filters:
                    mapping_str = r.get("Mapping", "") or r.get("mapping", "")
                    if not check_stretch_filter(mapping_str, stretch_filters):
                        continue
                
                filtered_results.append(r)
            
            log.append(f"✅ Sau lọc: {len(filtered_results):,} kết quả")
            
            if arrangement_filters:
                log.append(f"   🎯 Filter sắp cúi: {arrangement_filters}")
            if stretch_filters:
                log.append(f"   📏 Filter kéo dãn: {stretch_filters}")
            
            return filtered_results, "\n".join(log), current_filters  # ← Đã đúng 3 giá trị

        # ===== SỰ KIỆN TRA CỨU =====
        def update_with_page_info(log, table, results, page_num, new_iteration, current_input, cache_dict, _original_results, current_filters):
            if not results or len(results) == 0:
                return log, table, results, page_num, new_iteration, current_input, cache_dict, _original_results, current_filters, "", 1, gr.update(visible=False), gr.update(visible=False)
            
            max_page = max(0, (len(results) - 1) // 10)
            page_info_text = f"Trang: {page_num + 1}/{max_page + 1} | Tổng: {len(results)} kết quả"
            
            # Hiển thị pagination nếu có kết quả
            show_pagination = len(results) > 0
            
            return log, table, results, page_num, new_iteration, current_input, cache_dict, _original_results, current_filters, page_info_text, str(page_num + 1), gr.update(visible=show_pagination), gr.update(visible=show_pagination)
        # ===== HÀM WRAPPER CHO FILTER =====
        def smart_filter_or_search(product_name, color_names, color_ratios, num_units, 
                                    elongation_limit, priority_input, 
                                    arrangement_filter_input, blend_input, search_mode, current_iteration, 
                                    last_input, cache_dict, original_results, last_filters):
            # ✅ Tạo key bao gồm màu + tỷ lệ + tách cúi + ghép sơ bộ
            current_input = f"{color_names.strip()}|{color_ratios.strip()}|{num_units}|{blend_input.strip()}"
            current_filters = f"{arrangement_filter_input.strip()}|{elongation_limit.strip()}"
            
            _original_results = original_results if original_results is not None else []
            
            # Trường hợp 1: MÀU/TỶ LỆ/TÁCH CÚI/GHÉP SƠ BỘ THAY ĐỔI
            if current_input != last_input and last_input != "":
                cache_dict = {'results': [], 'explored_regions': set(), 'best_seeds': []}
                current_iteration = 0
                
                log, structure, table, results, page_num, new_iteration, cache_dict = run_app(
                    product_name, color_names, color_ratios, num_units, 
                    elongation_limit, priority_input, 
                    arrangement_filter_input, blend_input, current_iteration, cache_dict
                )
                
                _original_results = results
                
                return log, table, results, page_num, new_iteration, current_input, cache_dict, _original_results, current_filters             
            # Trường hợp 2: CHỈ FILTER THAY ĐỔI
            if current_filters != last_filters and _original_results:
                filtered_results, filter_log, new_filters = quick_filter_results(
                    _original_results, arrangement_filter_input, elongation_limit, last_filters
                )
                
                if filtered_results is not None:
                    first_page_table = render_result_table(filtered_results, 0)
                    
                    return filter_log, first_page_table, filtered_results, 0, current_iteration, current_input, cache_dict, _original_results, new_filters
                        
            # Trường hợp 3: CHẠY MỚI
            # ← SỬA: Unpack 7 giá trị (thêm structure)
            log, structure, table, results, page_num, new_iteration, cache_dict = run_app(
                product_name, color_names, color_ratios, num_units, 
                elongation_limit, priority_input, 
                arrangement_filter_input, blend_input, current_iteration, cache_dict
            )
            
            if results:
                _original_results = results
            return log, table, results, page_num, new_iteration, current_input, cache_dict, _original_results, current_filters
        # ===== SỰ KIỆN TRA CỨU =====
        run_btn.click(
            fn=smart_filter_or_search,
            inputs=[
                product_name_input,
                color_names_input,
                color_ratios_input,
                num_units_input,
                elongation_limit_input,    
                priority_color_input,
                arrangement_filter_input,
                blend_input,
                search_mode_state,
                iteration_counter,
                last_color_input,
                cache_state,
                original_results_state,  
                last_filters_state       
            ],
            outputs=[
                log_output,  
                table_output, 
                results_state, 
                current_page, 
                iteration_counter,
                last_color_input,
                cache_state,
                original_results_state,  
                last_filters_state       
            ]
        ).then(
            fn=update_with_page_info,
            inputs=[
                log_output,  
                table_output, 
                results_state, 
                current_page, 
                iteration_counter,
                last_color_input,
                cache_state,
                original_results_state,  
                last_filters_state
            ],
            outputs=[
                log_output,  
                table_output,
                results_state, 
                current_page, 
                iteration_counter,
                last_color_input,
                cache_state,
                original_results_state,  
                last_filters_state,
                page_info,
                page_input,
                pagination_row,  
                page_info        
            ]
        )

        # ===== SỰ KIỆN THAY ĐỔI MÀU/TỶ LỆ =====
        color_names_input.change(
            fn=auto_reset_on_color_change,
            inputs=[color_names_input, color_ratios_input, num_units_input, blend_input,
                    last_color_input, iteration_counter, cache_state],
            outputs=[realtime_log, iteration_counter, last_color_input, cache_state]
        )

        color_ratios_input.change(
            fn=auto_reset_on_color_change,
            inputs=[color_names_input, color_ratios_input, num_units_input, blend_input,
                    last_color_input, iteration_counter, cache_state],
            outputs=[realtime_log, iteration_counter, last_color_input, cache_state]
        )

        # ✅ SỰ KIỆN THAY ĐỔI TÁCH CÚI (2 sự kiện độc lập)
        num_units_input.change(
            fn=auto_reset_on_color_change,
            inputs=[color_names_input, color_ratios_input, num_units_input, blend_input,
                    last_color_input, iteration_counter, cache_state],
            outputs=[realtime_log, iteration_counter, last_color_input, cache_state]
        )
        # ✅ SỰ KIỆN THAY ĐỔI GHÉP SƠ BỘ
        blend_input.change(
            fn=auto_reset_on_color_change,
            inputs=[color_names_input, color_ratios_input, num_units_input, blend_input,
                    last_color_input, iteration_counter, cache_state],
            outputs=[realtime_log, iteration_counter, last_color_input, cache_state]
        )
        def auto_apply_filter(arrangement_input, elongation_input, original_results, last_filters, current_page_num):
            if not original_results:
                return gr.update(), gr.update(), original_results, 0, last_filters, gr.update(visible=False), gr.update(visible=False), "1"
            
            filtered_results, filter_log, new_filters = quick_filter_results(
                original_results, arrangement_input, elongation_input, last_filters
            )
            
            if filtered_results is None:
                return gr.update(), gr.update(), original_results, current_page_num, last_filters, gr.update(), gr.update(), str(current_page_num + 1)
            
            first_page_table = render_result_table(filtered_results, 0)
            max_page = max(0, (len(filtered_results) - 1) // 10)
            page_info_text = f"Trang: 1/{max_page + 1} | Tổng: {len(filtered_results)} kết quả"
            show_pagination = len(filtered_results) > 0
            
            # ← TRẢ VỀ filtered_results ĐỂ CẬP NHẬT results_state
            return gr.update(), first_page_table, filtered_results, 0, new_filters, gr.update(visible=show_pagination, value=page_info_text), gr.update(visible=show_pagination), "1"

        # Cập nhật outputs:
        arrangement_filter_input.change(
            fn=auto_apply_filter,
            inputs=[arrangement_filter_input, elongation_limit_input, original_results_state, last_filters_state, current_page],
            outputs=[log_output, table_output, results_state, current_page, last_filters_state, page_info, pagination_row, page_input]
        )

        elongation_limit_input.change(
            fn=auto_apply_filter,
            inputs=[arrangement_filter_input, elongation_limit_input, original_results_state, last_filters_state, current_page],
            outputs=[log_output, table_output, results_state, current_page, last_filters_state, page_info, pagination_row, page_input]
        )

        prev_btn.click(
            prev_page,
            inputs=[results_state, current_page],
            outputs=[table_output, current_page]
        ).then(
            fn=lambda results, page: (f"Trang: {page + 1}/{max(1, (len(results) - 1) // 10 + 1)} | Tổng: {len(results)} kết quả", str(page + 1)),
            inputs=[results_state, current_page],
            outputs=[page_info, page_input]
        )
        
        next_btn.click(
            next_page,
            inputs=[results_state, current_page],
            outputs=[table_output, current_page]
        ).then(
            fn=lambda results, page: (f"Trang: {page + 1}/{max(1, (len(results) - 1) // 10 + 1)} | Tổng: {len(results)} kết quả", str(page + 1)),
            inputs=[results_state, current_page],
            outputs=[page_info, page_input]
        )
        
        first_btn.click(
            first_page,
            inputs=[results_state],
            outputs=[table_output, current_page]
        ).then(
            fn=lambda results: (f"Trang: 1/{max(1, (len(results) - 1) // 10 + 1)} | Tổng: {len(results)} kết quả", "1"),
            inputs=[results_state],
            outputs=[page_info, page_input]
        )
        
        last_btn.click(
            last_page,
            inputs=[results_state],
            outputs=[table_output, current_page]
        ).then(
            fn=lambda results, page: (f"Trang: {page + 1}/{max(1, (len(results) - 1) // 10 + 1)} | Tổng: {len(results)} kết quả", str(page + 1)),
            inputs=[results_state, current_page],
            outputs=[page_info, page_input]
        )
        clear_btn.click(
            fn=clear_all_except_colors,
            inputs=[color_names_input, color_ratios_input],
            outputs=[
                color_names_input,
                color_ratios_input,
                num_units_input,
                elongation_limit_input,
                priority_color_input,
                arrangement_filter_input,
                blend_input,
                product_name_input,
                realtime_log,
                structure_line,
                arrangement_filter_preview,
                log_output,
                table_output,
                results_state,
                original_results_state,
                current_page,
                iteration_counter,
                last_color_input,
                last_filters_state,
                cache_state,
                page_info,
                page_input,
                pagination_row,
                page_info
            ]
        )
        page_input.change(
            goto_page,
            inputs=[results_state, page_input],
            outputs=[table_output, current_page]
        ).then(
            fn=lambda results, page: f"Trang: {page + 1}/{max(1, (len(results) - 1) // 10 + 1)} | Tổng: {len(results)} kết quả",
            inputs=[results_state, current_page],
            outputs=[page_info]
        )
        num_units_input.change(
            get_structure_line_from_textbox,
            inputs=num_units_input,
            outputs=structure_line
        )
    return app

four_stretch_app_mg = get_four_mg_stretch_app()
__all__ = ["four_stretch_app_mg"]
