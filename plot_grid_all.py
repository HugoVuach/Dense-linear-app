#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure: (scheds union) × 2 mappings.
Each subplot: GFLOP/s vs N, curves by NB, annotation "Best".
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scheduler orders (will filter based on what actually exists in the CSV)
SCHEDS_CPU_ONLY   = ["dm", "dmdap", "dmda"]
SCHEDS_HYBRID     = ["dmda", "dmdas", "heteroprio", "pheft"]
SCHEDULERS_ORDER  = ["dm", "dmdap", "dmda", "dmdas", "heteroprio", "pheft"]  # display order

MAPPINGS_ORDER = [
    ("4_cpu_only", "CPU-only (4 workers)"),
    ("hybrid",     "Hybrid (3 CPU + 1 GPU)"),
]

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expected = {"timestamp","scheduler","mapping","ncpu","ngpu","N","NB","run_idx","ms","exit_code","gflops","rel_error"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Data types
    for c in ["ncpu","ngpu","N","NB","run_idx","ms","exit_code","gflops","rel_error"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only successful runs
    df = df[df["exit_code"] == 0].copy()
    if df.empty:
        raise SystemExit("No data with exit_code==0 found in the CSV.")

    return df

def aggregate(df: pd.DataFrame, stat: str = "median") -> pd.DataFrame:
    agg_fun = "median" if stat == "median" else "mean"
    g = (df.groupby(["scheduler","mapping","N","NB"], as_index=False)
           .agg(gflops=("gflops", agg_fun)))
    return g

def plot_grid(g: pd.DataFrame, out_png: str, stat: str,
              peak_cpu: float, peak_hybrid: float):
    # Schedulers actually present in the data, keeping the global display order
    scheds_in_data = list(g["scheduler"].unique())
    scheds_present = [s for s in SCHEDULERS_ORDER if s in scheds_in_data]
    if not scheds_present:
        raise SystemExit("No expected scheduler found in the data.")

    # Figure: rows = #scheds_present, cols = 2 mappings
    nrows, ncols = len(scheds_present), len(MAPPINGS_ORDER)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.8*ncols, 2.9*nrows),
        sharey=False,
        squeeze=False
    )

    # List of NB values for a consistent legend
    all_nb = sorted(g["NB"].unique())

    # Prevent adding the peak performance line multiple times to the legend
    added_peak_label = {"cpu": False, "hybrid": False}

    for r, sched in enumerate(scheds_present):
        g_s = g[g["scheduler"] == sched]
        for c, (mapping_key, mapping_label) in enumerate(MAPPINGS_ORDER):
            ax = axes[r, c]
            ax.yaxis.set_tick_params(labelright=True)
            sub = g_s[g_s["mapping"] == mapping_key].copy()

            # NB curves
            any_plotted = False
            for nb in all_nb:
                dnb = sub[sub["NB"] == nb].sort_values("N")
                if dnb.empty:
                    continue
                ax.plot(dnb["N"], dnb["gflops"], marker="o", label=f"NB={nb}")
                any_plotted = True

            # Peak performance lines, by mapping
            if mapping_key == "4_cpu_only":
                lbl = "Peak (CPU-only)"
                ax.axhline(y=peak_cpu, linestyle=":", linewidth=1,
                           label=(lbl if not added_peak_label["cpu"] else None))
                added_peak_label["cpu"] = True
            elif mapping_key == "hybrid":
                lbl = "Peak (Hybrid)"
                ax.axhline(y=peak_hybrid, linestyle=":", linewidth=1,
                           label=(lbl if not added_peak_label["hybrid"] else None))
                added_peak_label["hybrid"] = True

            # Titles/axes
            ax.set_title(mapping_label if not sub.empty else f"{mapping_label}\n(no data)",
                         fontsize=12, pad=8)
            ax.set_xlabel("N")
            if c == 0:
                ax.set_ylabel(f"{sched}\nGFLOP/s", rotation=90)
            ax.grid(True, linestyle="--", alpha=0.35)

            # "Best" annotation
            if any_plotted:
                best_row = sub.loc[sub["gflops"].idxmax()]
                best_txt = f"Best: {best_row['gflops']:.2f} GFLOP/s (N={int(best_row['N'])}, NB={int(best_row['NB'])})"
                ax.annotate(
                    best_txt,
                    xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(2, -2), textcoords="offset points",
                    ha="left", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                )

    # Global legend (right side)
    handles, labels = [], []
    for r in range(nrows-1, -1, -1):
        for c in range(ncols-1, -1, -1):
            h, l = axes[r, c].get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            break
    if handles:
        fig.legend(handles, labels, title="NB sizes & Peak ref.",
                   loc="center right", bbox_to_anchor=(1.005, 0.5))

    # Global title + margins
    fig.suptitle(f"Cholesky Performance — {len(scheds_present)} schedulers × {len(MAPPINGS_ORDER)} mappings  (stat={stat})",
                 fontsize=14, y=0.995)
    plt.tight_layout(rect=[0.0, 0.0, 0.86, 0.965])
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[OK] Figure saved: {out_png}")
    plt.show()

def main():
    ap = argparse.ArgumentParser(description="Grid: schedulers × {4_cpu_only, hybrid} — GFLOP/s vs N (NB curves)")
    ap.add_argument("csv", nargs="?", default="results/bench.csv", help="Path to CSV (default: results/bench.csv)")
    ap.add_argument("--stat", choices=["median","mean"], default="median", help="Aggregation statistic for GFLOP/s (default: median)")
    ap.add_argument("--out", default="perf_grid.png", help="Output PNG filename")
    ap.add_argument("--peak-cpu", type=float, default=243.2, help="Peak CPU-only performance (GFLOP/s), default 243.2")
    ap.add_argument("--peak-hybrid", type=float, default=300.0, help="Peak Hybrid performance (GFLOP/s), default 300.0")
    args = ap.parse_args()

    df = load_and_prepare(args.csv)

    # Filter only the mappings you use
    df = df[df["mapping"].isin(["4_cpu_only", "hybrid"])].copy()
    if df.empty:
        raise SystemExit("No data for mapping ∈ {4_cpu_only, hybrid}.")

    g = aggregate(df, stat=args.stat)
    plot_grid(g, args.out, stat=args.stat,
              peak_cpu=args.peak_cpu, peak_hybrid=args.peak_hybrid)

if __name__ == "__main__":
    main()
