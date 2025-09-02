#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure: (scheds union) × 2 mappings.
Chaque subplot : GFLOP/s vs N, courbes par NB, annotation "Best".
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ordres (on filtrera selon ce qui existe réellement dans le CSV)
SCHEDS_CPU_ONLY   = ["dm", "dmdap", "dmda"]
SCHEDS_HYBRID     = ["dmda", "dmdas", "heteroprio", "pheft"]
SCHEDULERS_ORDER  = ["dm", "dmdap", "dmda", "dmdas", "heteroprio", "pheft"]  # ordre d’affichage

MAPPINGS_ORDER = [
    ("4_cpu_only", "CPU-only (4 workers)"),
    ("hybrid",     "Hybrid (3 CPU workers + 1 GPU, +1 CPU for CUDA)"),
]

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    expected = {"timestamp","scheduler","mapping","ncpu","ngpu","N","NB","run_idx","ms","exit_code","gflops","rel_error"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    # Types
    for c in ["ncpu","ngpu","N","NB","run_idx","ms","exit_code","gflops","rel_error"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Runs OK uniquement
    df = df[df["exit_code"] == 0].copy()
    if df.empty:
        raise SystemExit("Aucune donnée avec exit_code==0 dans le CSV.")

    return df

def aggregate(df: pd.DataFrame, stat: str = "median") -> pd.DataFrame:
    agg_fun = "median" if stat == "median" else "mean"
    g = (df.groupby(["scheduler","mapping","N","NB"], as_index=False)
           .agg(gflops=("gflops", agg_fun)))
    return g

def plot_grid(g: pd.DataFrame, out_png: str, stat: str,
              peak_cpu: float, peak_hybrid: float):
    # Schedulers réellement présents dans les données, respectant l’ordre global
    scheds_in_data = list(g["scheduler"].unique())
    scheds_present = [s for s in SCHEDULERS_ORDER if s in scheds_in_data]
    if not scheds_present:
        raise SystemExit("Aucun scheduler attendu n'est présent dans les données.")

    # Figure: rows = #scheds_present, cols = 2 mappings
    nrows, ncols = len(scheds_present), len(MAPPINGS_ORDER)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.8*ncols, 2.9*nrows),
        sharey=False,
        squeeze=False
    )

    # Liste NB pour une légende cohérente
    all_nb = sorted(g["NB"].unique())

    # Pour éviter d’ajouter la ligne de perf crête dans la légende plusieurs fois
    added_peak_label = {"cpu": False, "hybrid": False}

    for r, sched in enumerate(scheds_present):
        g_s = g[g["scheduler"] == sched]
        for c, (mapping_key, mapping_label) in enumerate(MAPPINGS_ORDER):
            ax = axes[r, c]
            ax.yaxis.set_tick_params(labelright=True)  # Valeurs à droite aussi

            # Pour la colonne Hybrid, afficher aussi l'axe Y "GFLOP/s" à droite
            if mapping_key == "hybrid":
                ax.set_ylabel("GFLOP/s", rotation=270, labelpad=15)
                ax.yaxis.set_label_position("right")

            sub = g_s[g_s["mapping"] == mapping_key].copy()

            # Traces NB
            any_plotted = False
            for nb in all_nb:
                dnb = sub[sub["NB"] == nb].sort_values("N")
                if dnb.empty:
                    continue
                ax.plot(dnb["N"], dnb["gflops"], marker="o", label=f"NB={nb}")
                any_plotted = True

            # Lignes de perf crête
            if mapping_key == "4_cpu_only":
                lbl = "Peak (CPU-only)"
                ax.axhline(y=peak_cpu, linestyle=":", linewidth=1, color="red",
                           label=(lbl if not added_peak_label["cpu"] else None))
                added_peak_label["cpu"] = True
            elif mapping_key == "hybrid":
                lbl = "Peak (Hybrid)"
                ax.axhline(y=peak_hybrid, linestyle=":", linewidth=1, color="purple",
                           label=(lbl if not added_peak_label["hybrid"] else None))
                added_peak_label["hybrid"] = True

            # Titres/axes
            ax.set_title(mapping_label if not sub.empty else f"{mapping_label}\n(sans données)",
                         fontsize=12, pad=8)
            ax.set_xlabel("N")
            if c == 0:
                ax.set_ylabel(f"{sched}\nGFLOP/s", rotation=90)
            ax.grid(True, linestyle="--", alpha=0.35)

            # Annotation "Best"
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

    # Légende globale (droite)
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
        fig.legend(handles, labels, title="Tailles NB & Réf. crête",
                   loc="center right", bbox_to_anchor=(1.005, 0.5))

    # Titre global + marges
    fig.suptitle(f"Performance Cholesky — {len(scheds_present)} schedulers × {len(MAPPINGS_ORDER)} mappings  (stat={stat})",
                 fontsize=14, y=0.995)
    plt.tight_layout(rect=[0.0, 0.0, 0.86, 0.965])
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[OK] Figure enregistrée : {out_png}")
    plt.show()

def main():
    ap = argparse.ArgumentParser(description="Grid: schedulers × {4_cpu_only, hybrid} — GFLOP/s vs N (courbes NB)")
    ap.add_argument("csv", nargs="?", default="results/bench.csv", help="Chemin du CSV (défaut: results/bench.csv)")
    ap.add_argument("--stat", choices=["median","mean"], default="median", help="Statistique d’agrégation des GFLOP/s (défaut: median)")
    ap.add_argument("--out", default="perf_grid.png", help="Nom du PNG de sortie")
    ap.add_argument("--peak-cpu", type=float, default=243.2, help="Perf crête CPU-only (GFLOP/s), défaut 243.2")
    ap.add_argument("--peak-hybrid", type=float, default=300.0, help="Perf crête Hybrid (GFLOP/s), défaut 300.0")
    args = ap.parse_args()

    df = load_and_prepare(args.csv)

    # Filtrer d’emblée aux mappings que tu utilises
    df = df[df["mapping"].isin(["4_cpu_only", "hybrid"])].copy()
    if df.empty:
        raise SystemExit("Aucune donnée pour mapping ∈ {4_cpu_only, hybrid}.")

    g = aggregate(df, stat=args.stat)
    plot_grid(g, args.out, stat=args.stat,
              peak_cpu=args.peak_cpu, peak_hybrid=args.peak_hybrid)

if __name__ == "__main__":
    main()
