#!/usr/bin/env python3
# mixing_optimizer.py
"""
混配牵伸倍数优化原型（网格+细化）
- 支持从 Markdown/文本文件提取“数据表”或使用内置示例
- 输出 top-N 解（CSV）与全部候选（JSON）
- 约束：X1,X2,X3 ∈ [1.1,2.0], X4 ∈ [1.1,6.0], 且 X4 > X1, X4 > X3, X4/X1 < 4, X4/X3 < 4
- 等效桶对：C<->D, E<->F, B<->G（用于去重）
"""

import re, os, json, argparse
from collections import defaultdict
import itertools
import math
import pandas as pd
import numpy as np

# -----------------------
# Utilities: parsing input
# -----------------------
def read_file_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_table_blocks_from_text(text):
    """
    Try to extract blocks for labels '数据1' and '数据2' (simple heuristics).
    Returns dict label->block_text (may be absent).
    """
    blocks = {}
    for label in ["数据1", "数据2"]:
        # try heading style '数据1' newline then block
        m = re.search(r"(?:^|\n)(?:#+\s*)?%s[:：]?\s*\n(.+?)(?=\n(?:#+\s*\w|$|\n{2,}))" % re.escape(label), text, flags=re.S)
        if m:
            blocks[label] = m.group(1).strip()
            continue
        # try code fence containing the label
        m2 = re.search(r"```(?:[\w+]*)\s*([\s\S]*?%s[\s\S]*?)```" % re.escape(label), text)
        if m2:
            blocks[label] = m2.group(1).strip()
    return blocks

def parse_table_like_block(block):
    """
    Try to parse CSV-like, pipe markdown table or whitespace separated table.
    Returns a pandas.DataFrame or None.
    """
    if not block or not block.strip():
        return None
    lines = [ln.rstrip() for ln in block.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    # CSV-like (commas)
    if all("," in ln for ln in lines[:min(6, len(lines))]):
        try:
            return pd.read_csv(pd.io.common.StringIO("\n".join(lines)), engine="python")
        except Exception:
            pass
    # Markdown pipe table
    if all("|" in ln for ln in lines[:min(6, len(lines))]):
        # remove separator lines like |---|
        good = [ln for ln in lines if not re.match(r"^\s*\|?\s*-{2,}", ln)]
        rows = []
        for ln in good:
            parts = [p.strip() for p in ln.strip().strip("|").split("|")]
            rows.append(parts)
        try:
            df = pd.DataFrame(rows[1:], columns=rows[0])
            return df
        except Exception:
            pass
    # whitespace-separated aligned columns
    parts = [re.split(r"\s+", ln) for ln in lines]
    if len(parts) >= 2 and all(len(r) == len(parts[0]) for r in parts[1:]):
        try:
            df = pd.DataFrame(parts[1:], columns=parts[0])
            return df
        except Exception:
            pass
    return None

# -----------------------
# Build target percents
# -----------------------
def build_targets_from_df(df, mode_hint="auto"):
    """
    Return dict: color_name -> target_percent (sum to 100)
    modes: if df has 'Percent' column use it, else if columns P1..Pn exist sum them per row.
    Otherwise equal split.
    """
    if df is None or df.shape[0] == 0:
        return {}
    cols = [c for c in df.columns]
    # use explicit Percent column
    if "Percent" in df.columns:
        names = df.iloc[:,0].astype(str).tolist()
        perc = df["Percent"].astype(float).tolist()
        s = sum(perc) or 1.0
        return dict(zip(names, [100.0*p/s for p in perc]))
    # sum P* columns if present
    pcols = [c for c in df.columns if re.match(r"^P\d+$", str(c), flags=re.I)]
    if pcols:
        names = df.iloc[:,0].astype(str).tolist()
        vals = df[pcols].sum(axis=1).astype(float).tolist()
        s = sum(vals) or 1.0
        return dict(zip(names, [100.0*v/s for v in vals]))
    # fallback: use second column numeric if looks like percentages
    # else equal split
    try:
        if df.shape[1] >= 2:
            second = df.iloc[:,1].astype(float).tolist()
            s = sum(second) or 1.0
            if all(v >= 0 for v in second):
                names = df.iloc[:,0].astype(str).tolist()
                return dict(zip(names, [100.0*v/s for v in second]))
    except Exception:
        pass
    # equal split
    names = df.iloc[:,0].astype(str).tolist()
    n = max(1, len(names))
    return dict(zip(names, [100.0/n]*n))

# -----------------------
# Problem-specific functions
# -----------------------
BUCKETS = ["A","B","C","D","E","F","G","H"]
EQUIV_GROUPS = [["C","D"], ["E","F"], ["B","G"]]

def bucket_speeds(X1, X2, X3, X4):
    """Return dict bucket -> speed (1/X or 1)"""
    return {
        "A": 1.0/float(X1),
        "B": 1.0/float(X4),
        "C": 1.0,
        "D": 1.0,
        "E": 1.0/float(X2),
        "F": 1.0/float(X2),
        "G": 1.0/float(X4),
        "H": 1.0/float(X3),
    }

def achieved_percent_from_assignment(assign, X1, X2, X3, X4):
    """
    assign: dict bucket->color_name
    return: achieved dict color->percent, per_bucket_percent dict, D float
    """
    speeds = bucket_speeds(X1,X2,X3,X4)
    D = sum(speeds[b] for b in BUCKETS)
    per_bucket = {b: 100.0*speeds[b]/D for b in BUCKETS}
    achieved = defaultdict(float)
    for b,c in assign.items():
        achieved[c] += per_bucket[b]
    return dict(achieved), per_bucket, D

# -----------------------
# Template generation
# -----------------------
def generate_templates_from_targets(targets, max_templates=300):
    """
    Create a small set of heuristic bucket assignments (templates).
    We try:
      - baseline: give each color at least one bucket, extra buckets to largest colors
      - variations: split top colors into more buckets (round-robin)
      - some symmetric patterns
    Return list of dicts (bucket->color).
    """
    if not targets:
        return []
    colors_sorted = sorted(list(targets.keys()), key=lambda c: -targets[c])
    templates = []

    # Template 1: one-per-color then fill remaining with largest colors
    assign = {}
    idx = 0
    for c in colors_sorted:
        if idx >= len(BUCKETS): break
        assign[BUCKETS[idx]] = c; idx += 1
    while idx < len(BUCKETS):
        assign[BUCKETS[idx]] = colors_sorted[(idx - len(colors_sorted)) % len(colors_sorted)]
        idx += 1
    templates.append(assign.copy())

    # Template 2: split top k colors more
    for top_split in [1,2]:
        for variant in range(2):
            assign = {}
            idx = 0
            # give each color one bucket first
            for c in colors_sorted:
                if idx >= len(BUCKETS): break
                assign[BUCKETS[idx]] = c; idx += 1
            # distribute remaining buckets to top_split largest
            while idx < len(BUCKETS):
                for i in range(top_split):
                    assign[BUCKETS[idx]] = colors_sorted[i % len(colors_sorted)]
                    idx += 1
                    if idx >= len(BUCKETS): break
            templates.append(assign.copy())
            if len(templates) >= max_templates: break
        if len(templates) >= max_templates: break

    # Template 3: keep symmetric pairs same color for some patterns
    for m in range(min(4, len(colors_sorted))):
        assign = {}
        for i,b in enumerate(BUCKETS):
            assign[b] = colors_sorted[(i + m) % len(colors_sorted)]
        templates.append(assign.copy())
        if len(templates) >= max_templates: break

    # canonicalize templates to remove duplicates due to symmetry
    def canonical_key(assign):
        arr = assign.copy()
        for g in EQUIV_GROUPS:
            vals = sorted([arr[g[0]], arr[g[1]]])
            arr[g[0]], arr[g[1]] = vals[0], vals[1]
        return tuple(arr[b] for b in BUCKETS)
    seen = set()
    unique = []
    for t in templates:
        k = canonical_key(t)
        if k in seen: continue
        seen.add(k); unique.append(t)
    return unique

# -----------------------
# Search (coarse grid + local refine)
# -----------------------
def search_targets(targets,
                   templates,
                   coarse_steps=6,
                   x1_range=(1.1,2.0),
                   x2_range=(1.1,2.0),
                   x3_range=(1.1,2.0),
                   x4_range=(1.1,6.0),
                   top_k_refine=20,
                   refine_points=9):
    """
    Returns list of candidate solutions (dicts). Each solution contains:
      assignment, X1..X4, D, E (sum abs error), achieved, by_bucket
    """
    candidates = []

    # build coarse grid arrays
    x1_vals = list(np.linspace(x1_range[0], x1_range[1], coarse_steps))
    x2_vals = list(np.linspace(x2_range[0], x2_range[1], coarse_steps))
    x3_vals = list(np.linspace(x3_range[0], x3_range[1], coarse_steps))
    x4_vals = list(np.linspace(x4_range[0], x4_range[1], coarse_steps*4))  # more resolution for X4

    # enumerate
    for tidx, templ in enumerate(templates):
        for X1 in x1_vals:
            for X2 in x2_vals:
                for X3 in x3_vals:
                    for X4 in x4_vals:
                        # constraints and quick prunes
                        if not (X4 > X1 and X4 > X3):
                            continue
                        if X4/X1 >= 4.0 or X4/X3 >= 4.0:
                            continue
                        achieved, by_bucket, D = achieved_percent_from_assignment(templ, X1,X2,X3,X4)
                        # compute E = sum absolute error across all target colors
                        E = 0.0
                        for c in targets:
                            E += abs(achieved.get(c, 0.0) - targets[c])
                        candidates.append({
                            "template_idx": tidx,
                            "assignment": templ,
                            "X1": float(X1),
                            "X2": float(X2),
                            "X3": float(X3),
                            "X4": float(X4),
                            "D": float(D),
                            "E": float(E),
                            "achieved": achieved,
                            "by_bucket": by_bucket
                        })
    # sort and take top_k_refine for local refine
    candidates.sort(key=lambda r: r["E"])
    top_coarse = candidates[:top_k_refine]

    refined = []
    for base in top_coarse:
        bx1,bx2,bx3,bx4 = base["X1"], base["X2"], base["X3"], base["X4"]
        # local grids: +/- 0.12 around base, refine_points evenly spaced
        def local_range(x, low, high):
            lo = max(low, x - 0.12)
            hi = min(high, x + 0.12)
            return list(np.linspace(lo, hi, refine_points))
        for X1 in local_range(bx1, x1_range[0], x1_range[1]):
            for X2 in local_range(bx2, x2_range[0], x2_range[1]):
                for X3 in local_range(bx3, x3_range[0], x3_range[1]):
                    for X4 in local_range(bx4, x4_range[0], x4_range[1]):
                        if not (X4 > X1 and X4 > X3):
                            continue
                        if X4/X1 >= 4.0 or X4/X3 >= 4.0:
                            continue
                        achieved, by_bucket, D = achieved_percent_from_assignment(base["assignment"], X1,X2,X3,X4)
                        E = sum(abs(achieved.get(c,0.0) - targets[c]) for c in targets)
                        refined.append({
                            "template_idx": base["template_idx"],
                            "assignment": base["assignment"],
                            "X1": float(X1),
                            "X2": float(X2),
                            "X3": float(X3),
                            "X4": float(X4),
                            "D": float(D),
                            "E": float(E),
                            "achieved": achieved,
                            "by_bucket": by_bucket
                        })

    combined = candidates + refined
    # deduplicate canonical solutions (assignment canonicalized + rounded Xs)
    def canonical_key(sol):
        assign = sol["assignment"].copy()
        for g in EQUIV_GROUPS:
            vals = sorted([assign[g[0]], assign[g[1]]])
            assign[g[0]], assign[g[1]] = vals[0], vals[1]
        xs = (round(sol["X1"],3), round(sol["X2"],3), round(sol["X3"],3), round(sol["X4"],3))
        key = tuple(assign[b] for b in BUCKETS) + xs
        return key
    seen = set()
    unique = []
    for sol in sorted(combined, key=lambda r: r["E"]):
        k = canonical_key(sol)
        if k in seen:
            continue
        seen.add(k)
        unique.append(sol)
    return unique

# -----------------------
# Result formatting & output
# -----------------------
def summarize_top_results(results, targets, topN=10):
    rows = []
    for i, r in enumerate(results[:topN]):
        achieved = r["achieved"]
        # per-color detail
        per_color = "; ".join(f"{c}: t{targets[c]:.2f}->a{achieved.get(c,0.0):.2f} (err {abs(achieved.get(c,0.0)-targets[c]):.2f})"
                              for c in targets)
        rows.append({
            "rank": i+1,
            "E_total": round(r["E"],6),
            "D": round(r["D"],6),
            "X1": round(r["X1"],6),
            "X2": round(r["X2"],6),
            "X3": round(r["X3"],6),
            "X4": round(r["X4"],6),
            "per_color_summary": per_color,
            "assignment": json.dumps(r["assignment"], ensure_ascii=False)
        })
    return pd.DataFrame(rows)

# -----------------------
# Main runnable entry
# -----------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Mixing/stretch multiplier optimizer (grid+refine prototype).")
    parser.add_argument("-i","--input", help="Path to markdown/text file containing 数据1/数据2 table blocks (optional).")
    parser.add_argument("-o","--outdir", default=".", help="Output directory for CSV/JSON files.")
    parser.add_argument("--topn", type=int, default=10, help="Top N solutions to save in summary CSV.")
    parser.add_argument("--coarsesteps", type=int, default=6, help="Coarse grid steps for X1/X2/X3.")
    parser.add_argument("--refinek", type=int, default=20, help="Number of coarse top candidates to refine.")
    args = parser.parse_args(argv)

    # load data
    df1 = df2 = None
    if args.input and os.path.exists(args.input):
        txt = read_file_text(args.input)
        blocks = extract_table_blocks_from_text(txt)
        if "数据1" in blocks:
            df1 = parse_table_like_block(blocks["数据1"])
        if "数据2" in blocks:
            df2 = parse_table_like_block(blocks["数据2"])

    # if missing, use example fallback data (you can replace these with real tables)
    if df1 is None:
        df1 = pd.DataFrame([
            ["A", 2, 33.2, 1, 2, 3],
            ["B", 1, 35, 0, 1, 1],
            ["C", 3, 10, 2, 0, 1],
            ["D", 1, 12, 0, 0, 0],
        ], columns=["分组","数量","P1","P2","P3","P4"])
        print("[Info] 数据1 未检测到，使用内置 fallback 示例（可用 -i 指定文件）")
    if df2 is None:
        df2 = pd.DataFrame([
            ["A", 1, 40],
            ["B", 1, 30],
            ["C", 1, 20],
            ["D", 1, 10],
        ], columns=["分组","数量","Percent"])
        print("[Info] 数据2 未检测到，使用内置 fallback 示例（可用 -i 指定文件）")

    targets1 = build_targets_from_df(df1, mode_hint="data1")
    targets2 = build_targets_from_df(df2, mode_hint="data2")
    print("Targets1:", targets1)
    print("Targets2:", targets2)

    # generate templates
    templates1 = generate_templates_from_targets(targets1, max_templates=200)
    templates2 = generate_templates_from_targets(targets2, max_templates=200)
    print("Generated templates:", len(templates1), len(templates2))

    # search params
    coarse_steps = args.coarsesteps
    top_k_refine = args.refinek

    # search
    print("Searching targets for 数据1 ...")
    results1 = search_targets(targets1, templates1, coarse_steps=coarse_steps, top_k_refine=top_k_refine)
    print("Searching targets for 数据2 ...")
    results2 = search_targets(targets2, templates2, coarse_steps=coarse_steps, top_k_refine=top_k_refine)

    # summarize and save
    os.makedirs(args.outdir, exist_ok=True)
    summary1 = summarize_top_results(results1, targets1, topN=args.topn)
    summary2 = summarize_top_results(results2, targets2, topN=args.topn)
    out_csv1 = os.path.join(args.outdir, "top_solutions_data1.csv")
    out_csv2 = os.path.join(args.outdir, "top_solutions_data2.csv")
    summary1.to_csv(out_csv1, index=False)
    summary2.to_csv(out_csv2, index=False)
    print(f"Saved top-{args.topn} CSVs to: {out_csv1}, {out_csv2}")

    # save all unique candidates to JSON for inspection
    def serialize_all(res):
        out = []
        for r in res:
            out.append({
                "assignment": r["assignment"],
                "X1": r["X1"], "X2": r["X2"], "X3": r["X3"], "X4": r["X4"],
                "D": r["D"], "E": r["E"],
                "achieved": r["achieved"],
                "by_bucket": r["by_bucket"]
            })
        return out

    all1_path = os.path.join(args.outdir, "all_solutions_data1.json")
    all2_path = os.path.join(args.outdir, "all_solutions_data2.json")
    with open(all1_path, "w", encoding="utf-8") as f:
        json.dump(serialize_all(results1), f, ensure_ascii=False, indent=2)
    with open(all2_path, "w", encoding="utf-8") as f:
        json.dump(serialize_all(results2), f, ensure_ascii=False, indent=2)
    print("Saved full candidate JSONs to:", all1_path, all2_path)

    # Print first few summary lines to console
    print("\nTop results for 数据1:")
    print(summary1.to_string(index=False))
    print("\nTop results for 数据2:")
    print(summary2.to_string(index=False))

if __name__ == "__main__":
    main()
