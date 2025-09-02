#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

MAPPINGS_ORDER = [
    ("gpu_only", "01: GPU seul"),
    ("cpu_only", "10: CPU seul"),
    ("hybrid",   "11: Hybride"),
]

def load_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")
    df = pd.read_csv(csv_path)
    expected = {"timestamp","scheduler","mapping","ncpu","ngpu","N","NB","run_idx","ms","exit_code","gflops","rel_error"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")
    for c in ["ncpu","ngpu","N","NB","run_idx","ms","exit_code","gflops","rel_error"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # on garde les runs OK
    df = df[df["exit_code"] == 0].copy()
    if df.empty:
        raise SystemExit("Aucune donnée (exit_code==0).")
    return df

def aggregate_residuals(df: pd.DataFrame, stat: str = "median") -> pd.DataFrame:
    # agrège la résiduelle par (scheduler, mapping, N, NB)
    agg_fun = "median" if stat == "median" else "mean"
    g = (df.groupby(["scheduler","mapping","N","NB"], as_index=False)
           .agg(relerr=("rel_error", agg_fun)))
    return g

def plot_one_scheduler(g_s: pd.DataFrame, scheduler: str, outdir: str, stat: str):
    # figure 1×3, une colonne = un mapping
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    all_nb = sorted(g_s["NB"].unique())

    for ax, (mapping_key, mapping_label) in zip(axes, MAPPINGS_ORDER):
        sub = g_s[g_s["mapping"] == mapping_key].copy()
        if sub.empty:
            ax.set_title(f"{mapping_label} (aucune donnée)")
            ax.set_xlabel("N")
            ax.set_ylabel("||A-LLᵀ||_inf / ||A||_inf")
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.35)
            continue

        any_plot = False
        for nb in all_nb:
            d = sub[sub["NB"] == nb].sort_values("N")
            if d.empty: 
                continue
            ax.plot(d["N"], d["relerr"], marker="o", label=f"NB={nb}")
            any_plot = True

        ax.set_title(mapping_label)
        ax.set_xlabel("N")
        ax.set_ylabel("||A-LLᵀ||_inf / ||A||_inf")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

        if any_plot:
            best = sub.loc[sub["relerr"].idxmin()]
            ax.text(0.5, 1.02,
                    f"Best(min): {best['relerr']:.2e} (N={int(best['N'])}, NB={int(best['NB'])})",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=9)

    # légende commune (à droite)
    handles, labels = axes[-1].get_legend_handles_labels()
    if not handles:
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            if h: handles, labels = h, l; break
    if handles:
        fig.legend(handles, labels, title="Tailles NB", loc="center right", bbox_to_anchor=(1.02, 0.5))

    fig.suptitle(f"Erreur résiduelle — Scheduler: {scheduler} (stat={stat})", y=1.02, fontsize=14)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"residual_{scheduler}.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"[OK] {outpath}")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Génère un graphique de résiduelle par scheduler (3 sous-graphes: GPU/CPU/Hybride).")
    ap.add_argument("csv", nargs="?", default="results/bench.csv", help="Chemin du CSV (défaut: results/bench.csv)")
    ap.add_argument("--stat", choices=["median","mean"], default="median", help="Agrégation des runs par combinaison (défaut: median)")
    ap.add_argument("--outdir", default="plots_residuals", help="Dossier de sortie des PNG")
    args = ap.parse_args()

    df = load_df(args.csv)
    g = aggregate_residuals(df, stat=args.stat)

    schedulers = sorted(g["scheduler"].unique())
    if not schedulers:
        raise SystemExit("Aucun scheduler trouvé.")

    for sched in schedulers:
        g_s = g[g["scheduler"] == sched]
        plot_one_scheduler(g_s, sched, args.outdir, args.stat)

if __name__ == "__main__":
    main()
