#!/usr/bin/env python3
"""Locate the runs behind the published tables and assemble ``paper_results/``.

Every number printed in Tables 2-4 of the manuscript is looked up in the stored
result CSVs. A value counts as traced only when a stored run reproduces it to
the precision at which it was published (two decimals for the percentages of
Tables 2-3). Nothing is inferred: values with no matching run are reported as
gaps rather than replaced by the closest available run.

Usage:
    python tools/collect_paper_results.py                 # report only
    python tools/collect_paper_results.py --write         # also build paper_results/
    python tools/collect_paper_results.py --extra-dir /path/to/synced/results

Exit status is non-zero when some published value could not be traced, so the
script doubles as a check that the repository really carries its own evidence.
"""
import argparse
import csv
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.paper_values import (  # noqa: E402
    DATASETS, PAPER_LABEL, TABLE2, TABLE3, TABLE4, CSV_METRIC, METHOD_LABEL,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "paper_results")

# paper_results/raw/ comes first: it holds the evidence published with the
# repository, so a clean clone can reproduce this audit without the working
# directories of the original machine.
DEFAULT_DIRS = {
    "pathboost": ["paper_results/raw", "PathBoost_results", "namshub_results/PathBoost_results"],
    "gnn": ["paper_results/raw", "GNN_results", "namshub_results/GNN_results"],
    "kernel": ["paper_results/raw", "Kernel_results", "namshub_results/Kernel_results"],
    "regression": ["paper_results/raw", "."],
}

FILE_GLOB = {
    "pathboost": "Sequential_PathBoost_Performance_*.csv",
    "gnn": "GNN_Performance_*.csv",
    "kernel": "Kernel_Performance_*.csv",
    "regression": "Sequential_PathBoost_Regression_Performance_*.csv",
}

SENTINELS = {"FAILED", "TIMEOUT", ""}

# Aggregated tables kept alongside the raw runs. When a raw run file is no
# longer available, a published value may still be traceable to one of these.
WIDE_SOURCES = [
    "paper_results/raw/merged_results_enriched_and_modified.csv",
    "results_csv_files/merged_results_enriched_and_modified.csv",
]

# Column carrying each Table 2 method inside the aggregated tables.
WIDE_COLUMN = {
    "PB": "PB_accuracy",
    "GNN": "GNN_accuracy",
    "WL": "Weisfeiler-Lehman subtree kernel_accuracy",
    "GR": "Graphlet kernel_accuracy",
    "SP": "Shortest-path kernel_accuracy",
}
WIDE_STD_COLUMN = {
    "PB": "PB_accuracy_std10",
    "GNN": "GNN_accuracy_std10",
    "WL": "Weisfeiler-Lehman subtree kernel_std_10",
    "GR": "Graphlet kernel_std_10",
    "SP": "Shortest-path kernel_std_10",
}


def find_match_wide(key, dataset, want_mean, want_std, tol=0.005):
    """Look the value up in an aggregated table; return (path, mean, std) or None."""
    col, std_col = WIDE_COLUMN.get(key), WIDE_STD_COLUMN.get(key)
    if col is None:
        return None
    for rel in WIDE_SOURCES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        try:
            with open(path, newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    if row.get("Dataset") != dataset or col not in row:
                        continue
                    mean_raw = to_float(row.get(col))
                    scale = percent_scale(mean_raw)
                    mean = None if mean_raw is None else mean_raw * scale
                    if mean is None or abs(mean - want_mean) > tol:
                        continue
                    std_raw = to_float(row.get(std_col)) if std_col in row else None
                    std = None if std_raw is None else std_raw * scale
                    return path, mean, std
        except Exception as exc:
            print(f"  ! unreadable: {rel} ({exc})", file=sys.stderr)
    return None


def to_float(raw):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def percent_scale(mean_raw):
    """PathBoost/GNN store accuracies as fractions, the kernels as percentages.

    The scale is decided from the mean and then applied to its standard
    deviation as well: a std is legitimately below 1 in either convention, so
    deciding per value would silently inflate it by a factor of 100.
    """
    return 100.0 if mean_raw is not None and abs(mean_raw) <= 1.0 else 1.0


def load_rows(method, extra_dirs):
    """Yield (path, row) for every non-sentinel row of a method's result CSVs."""
    dirs = list(DEFAULT_DIRS[method]) + [
        os.path.join(d, os.path.basename(x))
        for d in extra_dirs
        for x in DEFAULT_DIRS[method]
        if os.path.basename(x)
    ] + list(extra_dirs)

    seen = set()
    for d in dirs:
        pattern = os.path.join(d if os.path.isabs(d) else os.path.join(ROOT, d), FILE_GLOB[method])
        for path in sorted(glob.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            try:
                with open(path, newline="", encoding="utf-8-sig") as fh:
                    for row in csv.DictReader(fh):
                        if str(row.get("Mean", "")).strip() in SENTINELS:
                            continue
                        yield path, row
            except Exception as exc:
                print(f"  ! unreadable: {path} ({exc})", file=sys.stderr)


def find_match(method, dataset, metric, want_mean, want_std, extra_dirs, tol=0.005):
    """Return (path, mean, std) of a run reproducing the published value, else None."""
    fallback = None
    for path, row in load_rows(method, extra_dirs):
        if row.get("Dataset") != dataset or row.get("Metric") != metric:
            continue
        mean_raw = to_float(row.get("Mean"))
        scale = percent_scale(mean_raw)
        mean = None if mean_raw is None else mean_raw * scale
        if mean is None or abs(mean - want_mean) > tol:
            continue
        std_raw = to_float(row.get("Std_Top10"))
        std10 = None if std_raw is None else std_raw * scale
        if std10 is not None and abs(std10 - want_std) <= tol:
            return path, mean, std10
        if fallback is None:
            fallback = (path, mean, std10)
    return fallback


def trace_table(table, metric_of, extra_dirs, label):
    """Check one table; return (rows, traced, total)."""
    rows, traced, total = [], 0, 0
    for dataset in DATASETS:
        for key, published in table[dataset].items():
            if published is None:
                rows.append({
                    "Table": label, "Dataset": dataset,
                    "PaperName": PAPER_LABEL.get(dataset, dataset),
                    "Method": METHOD_LABEL[key],
                    "Published_Mean": "time limit", "Published_Std": "",
                    "Stored_Mean": "", "Stored_Std": "", "Source_File": "",
                    "Status": "time limit",
                })
                continue
            total += 1
            want_mean, want_std = published
            method, metric = metric_of(key)
            hit = find_match(method, dataset, metric, want_mean, want_std, extra_dirs)
            exact_hit = hit is not None and hit[2] is not None and abs(hit[2] - want_std) <= 0.005
            if not exact_hit and label == "Table 2":
                # Fall back to the aggregated tables kept in results_csv_files/
                hit = find_match_wide(key, dataset, want_mean, want_std) or hit
            if hit is None:
                status, src, sm, ss = "NOT TRACED", "", "", ""
            else:
                path, mean, std = hit
                src = os.path.relpath(path, ROOT)
                sm, ss = f"{mean:.2f}", "" if std is None else f"{std:.2f}"
                exact = std is not None and abs(std - want_std) <= 0.005
                status = "traced" if exact else "mean only"
                if exact:
                    traced += 1
            rows.append({
                "Table": label, "Dataset": dataset,
                "PaperName": PAPER_LABEL.get(dataset, dataset),
                "Method": METHOD_LABEL[key],
                "Published_Mean": f"{want_mean:.2f}", "Published_Std": f"{want_std:.2f}",
                "Stored_Mean": sm, "Stored_Std": ss, "Source_File": src,
                "Status": status,
            })
    return rows, traced, total


def trace_regression(extra_dirs, tol=0.0005):
    """Check Table 4 (alchemy_full, target 0): 'full' vs 'categorical_only'."""
    variant_suffix = {"Complete": "_full", "Restricted": "_categorical_only"}
    rows, traced, total = [], 0, 0
    for variant, published in TABLE4.items():
        dataset = "alchemy_full" + variant_suffix[variant]
        for metric in ("mae", "r2"):
            total += 1
            want_mean, _want_std = published[metric]
            hit, src = "", ""
            for path, row in load_rows("regression", extra_dirs):
                if row.get("Dataset") != dataset or row.get("Metric") != metric:
                    continue
                try:
                    value = float(row["Mean"])
                except (TypeError, ValueError):
                    continue
                if abs(value - want_mean) <= tol:
                    hit, src = f"{value:.4f}", os.path.relpath(path, ROOT)
                    break
            status = "traced" if hit else "NOT TRACED"
            if hit:
                traced += 1
            rows.append({
                "Table": "Table 4", "Dataset": dataset, "PaperName": variant,
                "Method": "PathBoost (regression)",
                "Published_Mean": f"{want_mean:.4f}", "Published_Std": "",
                "Stored_Mean": hit, "Stored_Std": "", "Source_File": src,
                "Status": status,
            })
    return rows, traced, total


def write_outputs(prov_rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_dir = os.path.join(OUT_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Tables 2 and 3, in the layout used in the paper.
    for label, table, keys, fname in (
        ("Table 2", TABLE2, ["PB", "GNN", "WL", "GR", "SP"], "table2_accuracy.csv"),
        ("Table 3", TABLE3, ["PB", "GNN"], "table3_f1_macro.csv"),
    ):
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Dataset", "PaperName"] + [METHOD_LABEL[k] for k in keys])
            for dataset in DATASETS:
                cells = []
                for k in keys:
                    v = table[dataset].get(k)
                    cells.append("time limit" if v is None else f"{v[0]:.2f} ± {v[1]:.2f}")
                w.writerow([dataset, PAPER_LABEL.get(dataset, dataset)] + cells)
        print(f"  wrote {os.path.relpath(path, ROOT)}")

    path = os.path.join(OUT_DIR, "table4_regression.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Variant", "MAE", "MAE_Std", "R2", "R2_Std", "Time_s"])
        for variant, v in TABLE4.items():
            w.writerow([variant, f"{v['mae'][0]:.4f}", f"{v['mae'][1]:.4f}",
                        f"{v['r2'][0]:.4f}", f"{v['r2'][1]:.4f}", v["time_s"]])
    print(f"  wrote {os.path.relpath(path, ROOT)}")

    path = os.path.join(OUT_DIR, "provenance.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prov_rows[0].keys()))
        w.writeheader()
        w.writerows(prov_rows)
    print(f"  wrote {os.path.relpath(path, ROOT)}")

    copied = 0
    for row in prov_rows:
        src = row.get("Source_File")
        if not src:
            continue
        abs_src = os.path.join(ROOT, src)
        dst = os.path.join(raw_dir, os.path.basename(src))
        if os.path.exists(abs_src) and not os.path.exists(dst):
            shutil.copy2(abs_src, dst)
            copied += 1
    print(f"  copied {copied} source run file(s) into {os.path.relpath(raw_dir, ROOT)}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="Build paper_results/ (default: report only)")
    ap.add_argument("--extra-dir", action="append", default=[], metavar="DIR",
                    help="Additional directory of result CSVs (repeatable), e.g. a server sync")
    args = ap.parse_args()

    extra = args.extra_dir
    print("Tracing published values to stored runs")
    searched = list(dict.fromkeys(sum(DEFAULT_DIRS.values(), []) + extra))
    print(f"  searching: {', '.join(searched)}\n")

    rows2, t2, n2 = trace_table(TABLE2, lambda k: CSV_METRIC[k], extra, "Table 2")
    rows3, t3, n3 = trace_table(
        TABLE3,
        lambda k: (CSV_METRIC[k][0], "f1_macro"),
        extra,
        "Table 3",
    )
    rows4, t4, n4 = trace_regression(extra)
    prov = rows2 + rows3 + rows4

    for label, traced, total in (("Table 2 (accuracy)", t2, n2),
                                 ("Table 3 (F1-macro)", t3, n3),
                                 ("Table 4 (regression)", t4, n4)):
        gap = total - traced
        flag = "" if gap == 0 else f"   <-- {gap} value(s) NOT traced"
        print(f"  {label:<24} {traced:>3}/{total:<3} traced{flag}")

    missing = [r for r in prov if r["Status"] == "NOT TRACED"]
    if missing:
        print(f"\n{len(missing)} published value(s) have no matching run in the searched directories:")
        by_method = {}
        for r in missing:
            by_method.setdefault(r["Method"], []).append(r["PaperName"])
        for method, datasets in by_method.items():
            print(f"  {method}: {', '.join(datasets)}")
        print("\nThese runs were produced on the compute servers. Sync them and re-run, e.g.:")
        print("  rsync -avz <user>@<server>:<path>/PathBoost_results/ ./PathBoost_results/")
        print("  python tools/collect_paper_results.py --write")

    if args.write:
        print("\nWriting paper_results/ ...")
        write_outputs(prov)

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
