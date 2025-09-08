"""
feature.py — Cross-Asset Joint Forecasting (panel multi-actifs)
---------------------------------------------------------------
- Parcourt un dossier de TSV/CSV OHLCV (date, open, high, close, low, volume)
- Union des dates, alignement par forward-fill (zéros avant 1ère obs si voulu)
- Extraction des features par actif via chart.ChartFeature (TA-Lib)
- Concaténation des blocs de features par actif => (F_total, T)
- Construction de fenêtres glissantes -> X: (N, F_total, window), y: (N, M)
- Standardisation par actif (z-score) calculée sur X_train uniquement
- Sérialisation en 2 pickles consécutifs (train_map puis test_map), format
  attendu par gossip.read_feature (qui transpose ensuite en [N, T, F])

Usage :
    python feature.py \
        --dataset_dir ./dataset \
        --window 30 \
        --prospective 1 \
        --days_for_test 700 \
        --selector "ROCP,OROCP,HROCP,LROCP,MACD,RSI,VROCP,BOLL,MA,VMA,PRICE_VOLUME,CROSS_PRICE" \
        --out ultimate_feature
"""

import os
import argparse
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


from rawdata import read_sample_data   # lit un TSV "\t" -> List[RawData] (date, open, high, close, low, volume)
from chart import ChartFeature         # on réutilise l’ingénierie de features unitaire

# --------------------
# Utilitaires internes
# --------------------

REQUIRED_COLS = {"date", "open", "high", "close", "low", "volume"}

def has_ohlcv_header(path: str, sep: str = "\t") -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().lower().split(sep)
        cols = {c.strip() for c in header}
        return REQUIRED_COLS.issubset(cols)
    except Exception:
        return False


# ---------------------------------------------------------------------
# Labels multi-cibles (mimique ChartFeature.make_label sur fenêtre future)
# ---------------------------------------------------------------------
def make_future_label_from_close(
    close: np.ndarray, start_idx: int, window: int, prospective: int
) -> float:
    """
    Reproduit la logique d'origine :
    label sur segment close[start : start+prospective] avec
    base = close[start] (qui est le dernier point *dans* la fenêtre).
    """
    left = start_idx + window - 1
    right = left + prospective  # inclus
    seg = close[left : right + 1]  # [left, ..., right], longueur prospective+1
    if seg.size <= 1:
        return 0.0

    ratio = 0.5
    decay = 0.9
    label = 0.0
    base = seg[0]
    for i in range(1, seg.size):
        label += (seg[i] / base - 1.0) * ratio
        ratio *= decay
    return label / (seg.size - 1)



# ---------------------------------------------------------------------
# Alignement : union des dates + forward-fill ; zéros avant 1re obs
# ---------------------------------------------------------------------
def forward_fill_align(
    all_dates: List[str],
    series: Dict[str, Tuple[float, float, float, float, float]],
    start_zero: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:    
    """
    series : dict(date -> (open, high, close, low, volume))
    Retourne 5 vecteurs numpy alignés sur all_dates (longueur T).
    - forward-fill après 1ère observation
    - avant 1ère observation :
        * start_zero=True  -> 0.0
        * start_zero=False -> valeur de la 1ère observation (backfill)
    """
    T = len(all_dates)
    O = np.full(T, np.nan, dtype=np.float64)
    H = np.full(T, np.nan, dtype=np.float64)
    C = np.full(T, np.nan, dtype=np.float64)
    L = np.full(T, np.nan, dtype=np.float64)
    V = np.full(T, np.nan, dtype=np.float64)

    last = None
    first_seen_idx = None

    for t, d in enumerate(all_dates):
        if d in series:
            last = series[d]
            if first_seen_idx is None:
                first_seen_idx = t
        if last is not None:
            O[t], H[t], C[t], L[t], V[t] = last

    # Avant la 1re observation
    if first_seen_idx is not None and first_seen_idx > 0:
        if start_zero:
            O[:first_seen_idx] = 0.0
            H[:first_seen_idx] = 0.0
            C[:first_seen_idx] = 0.0
            L[:first_seen_idx] = 0.0
            V[:first_seen_idx] = 0.0
        else:
            o0, h0, c0, l0, v0 = series[all_dates[first_seen_idx]]
            O[:first_seen_idx] = o0
            H[:first_seen_idx] = h0
            C[:first_seen_idx] = c0
            L[:first_seen_idx] = l0
            V[:first_seen_idx] = v0

    # Si aucune donnée : remplis zéro (rare, mais protège le pipeline)
    if first_seen_idx is None:
        O[:] = H[:] = C[:] = L[:] = V[:] = 0.0

    # Remplace résidus NaN (bords) par 0
    O = np.nan_to_num(O)
    H = np.nan_to_num(H)
    C = np.nan_to_num(C)
    L = np.nan_to_num(L)
    V = np.nan_to_num(V)

    return O, H, C, L, V


def build_windows_block(
    features_concat: np.ndarray,           # (F_total, T)
    closes_by_asset: List[np.ndarray],     # M arrays (T,)
    window: int,
    prospective: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide sur t = 0..T - window - prospective
    X[t] = features[:, t : t+window]        -> (F_total, window)
    y[t] = [label_m(t) for m in 1..M]       -> (M,)
    Retour : X_all (N, F_total, window), Y_all (N, M)
    """
    F_total, T = features_concat.shape
    M = len(closes_by_asset)
    last_start = T - window - prospective
    if last_start < 0:
        # pas assez de données pour 1 fenêtre
        return np.zeros((0, F_total, window), dtype=np.float32), np.zeros((0, M), dtype=np.float32)

    X = np.empty((last_start + 1, F_total, window), dtype=np.float32)
    Y = np.empty((last_start + 1, M), dtype=np.float32)

    for t in range(last_start + 1):
        X[t] = features_concat[:, t : t + window]
        for m in range(M):
            Y[t, m] = make_future_label_from_close(closes_by_asset[m], t, window, prospective)

    return X, Y


# ---------------------------------------------------------------------
# Standardisation par actif (z-score), calculée sur TRAIN uniquement
# ---------------------------------------------------------------------
def zscore_by_asset_block(
    X_tr: np.ndarray, X_te: np.ndarray, asset_feat_slices: List[Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X_* : (N, F_total, window)
    asset_feat_slices : liste de tranches (f0, f1) pour chaque actif dans F_total
    Standardise chaque bloc d'actif indépendamment avec stats de TRAIN.
    """
    Xtr = X_tr.copy()
    Xte = X_te.copy()

    for (f0, f1) in asset_feat_slices:
        # Moyenne/écart-type sur TRAIN, axes (batch, time) -> (N * window)
        mu = Xtr[:, f0:f1, :].mean(axis=(0, 2), keepdims=True)      # (1, f1-f0, 1)
        sd = Xtr[:, f0:f1, :].std(axis=(0, 2), keepdims=True) + 1e-8

        Xtr[:, f0:f1, :] = (Xtr[:, f0:f1, :] - mu) / sd
        Xte[:, f0:f1, :] = (Xte[:, f0:f1, :] - mu) / sd

    return Xtr, Xte


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Génération features panel multi-actifs (Cross-Asset).")
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(os.path.dirname(__file__), "dataset"))
    parser.add_argument("--window", type=int, default=30, help="Longueur de la fenêtre (T).")
    parser.add_argument("--prospective", type=int, default=1, help="Horizon futur pour le label.")
    parser.add_argument("--recall_period", type=int, default=2, help="Profondeur de rappel pour certaines features.")
    parser.add_argument("--days_for_test", type=int, default=700, help="Nb de fenêtres réservées au test.")
    parser.add_argument("--selector", type=str, default="ROCP,OROCP,HROCP,LROCP,MACD,RSI,VROCP,BOLL,MA,VMA,PRICE_VOLUME,CROSS_PRICE")
    parser.add_argument("--out", type=str, default="ultimate_feature", help="Fichier pickle de sortie.")
    parser.add_argument("--universe", type=str, default="", help="Liste de tickers CSV (sinon: tous les fichiers .csv/.tsv).")
    args = parser.parse_args()

    selector = [s.strip() for s in args.selector.split(",") if s.strip()]

    # 1) Lister les fichiers OHLCV et ignorer les fichiers méta
    all_files = []
    for fn in os.listdir(args.dataset_dir):
        if fn.lower().endswith((".csv", ".tsv")):
            path = os.path.join(args.dataset_dir, fn)
            if has_ohlcv_header(path, sep="\t"):
                all_files.append(fn)
            else:
                print(f"skip: {fn} (non-OHLCV ou méta)")
    if not all_files:
        raise FileNotFoundError(f"Aucun fichier OHLCV valide dans {args.dataset_dir}")

    # Filtre par univers explicite (liste de tickers sans extension)
    universe = [u.strip() for u in args.universe.split(",") if u.strip()]
    if universe:
        keep = set(universe)
        all_files = [fn for fn in all_files if os.path.splitext(fn)[0] in keep]
    if not all_files:
        raise RuntimeError("Après filtrage univers, aucun fichier restant.")

    # 2) Lire et construire l’index global de dates (union)
    assets: List[str] = []
    per_asset_series: Dict[str, Dict[str, Tuple[float, float, float, float, float]]] = {}
    all_dates_set = set()

    for fn in sorted(all_files):
        ticker = os.path.splitext(fn)[0]
        path = os.path.join(args.dataset_dir, fn)
        rows = read_sample_data(path)  # -> List[RawData], triée par .date (string 'YYYY-MM-DD')
        if not rows:
            print(f"warn: {fn} vide, ignoré.")
            continue
        assets.append(ticker)
        dd: Dict[str, Tuple[float, float, float, float, float]] = {}
        for r in rows:
            all_dates_set.add(r.date)
            dd[r.date] = (float(r.open), float(r.high), float(r.close), float(r.low), float(r.volume))
        per_asset_series[ticker] = dd

    if not assets:
        raise RuntimeError("Aucun actif valide lu.")

    all_dates = sorted(all_dates_set)  # union des dates
    T = len(all_dates)
    print(f"[INFO] Actifs={len(assets)} | Dates uniques={T}")

    # 3) Pour chaque actif : aligner, extraire features via ChartFeature, empiler les blocs
    features_blocks: List[np.ndarray] = []   # liste de (F_actif, T)
    closes_by_asset: List[np.ndarray] = []   # liste de close(T,) pour labels
    asset_feat_slices: List[Tuple[int, int]] = []  # tranches (f0,f1) de chaque actif dans la concat finale
    f_cursor = 0

    for ticker in assets:
        series = per_asset_series[ticker]
        O, H, C, L, V = forward_fill_align(all_dates, series, start_zero=False)

        print(f"[DBG][{ticker}] any NaN/Inf in OHLCV:",
              np.isnan(O).any() or np.isinf(O).any(),
              np.isnan(H).any() or np.isinf(H).any(),
              np.isnan(C).any() or np.isinf(C).any(),
              np.isnan(L).any() or np.isinf(L).any(),
              np.isnan(V).any() or np.isinf(V).any())
        print(f"[DBG][{ticker}] zeros in close:", np.count_nonzero(C == 0.0))


        # Extracteur unitaire (hérite de ChartFeature historique)
        cf = ChartFeature(selector)
        cf.recall_period = args.recall_period
        cf.prospective = args.prospective

        # Extrait par type -> self.feature: liste de vecteurs longueur T
        cf.extract(open_prices=O, close_prices=C, high_prices=H, low_prices=L, volumes=V)
        F_actif = len(cf.feature)
        feat_block = np.asarray(cf.feature, dtype=np.float64)  # (F_actif, T)

        # === DBG (features bloc) ===
        print(f"[DBG][{ticker}] any NaN/Inf in FEATURES:",
              np.isnan(feat_block).any(), np.isinf(feat_block).any(),
              "min/max:", np.nanmin(feat_block), np.nanmax(feat_block))

        features_blocks.append(feat_block)
        closes_by_asset.append(C.astype(np.float64))
        asset_feat_slices.append((f_cursor, f_cursor + F_actif))
        f_cursor += F_actif

        print(f"[{ticker}] F_actif={F_actif} | T={T}")

    # Concatène tous les blocs : (F_total, T)
    features_concat = np.concatenate(features_blocks, axis=0)
    F_total = features_concat.shape[0]
    M = len(assets)
    print(f"[INFO] F_total={F_total} | M={M}")

    # === DBG (concat) ===
    print("[DBG] any NaN/Inf in features_concat:",
          np.isnan(features_concat).any(), np.isinf(features_concat).any(),
          "min/max:", np.nanmin(features_concat), np.nanmax(features_concat))

    # 4) Fenêtres + labels multi-cibles
    X_all, Y_all = build_windows_block(features_concat, closes_by_asset,
                                       window=args.window, prospective=args.prospective)
    N = X_all.shape[0]
    print(f"[INFO] N_windows={N} | X_all={X_all.shape} | Y_all={Y_all.shape}")


    # === DBG (fenêtres brutes) ===
    print("[DBG] any NaN/Inf in X_all:",
          np.isnan(X_all).any(), np.isinf(X_all).any(),
          "min/max:", np.nanmin(X_all), np.nanmax(X_all))
    print("[DBG] any NaN/Inf in Y_all:",
          np.isnan(Y_all).any(), np.isinf(Y_all).any(),
          "min/max:", np.nanmin(Y_all), np.nanmax(Y_all))


    # 5) Split train/test en *fenêtres*
    cut = max(0, N - args.days_for_test)
    X_tr, X_te = X_all[:cut], X_all[cut:]
    Y_tr, Y_te = Y_all[:cut], Y_all[cut:]
    print(f"[SPLIT] Train={X_tr.shape[0]} | Test={X_te.shape[0]} (cut={cut})")

    # === DBG (avant standardisation) ===
    if X_tr.size:
        print("[DBG] any NaN/Inf in X_tr:",
              np.isnan(X_tr).any(), np.isinf(X_tr).any(),
              "min/max:", np.nanmin(X_tr), np.nanmax(X_tr))
    if X_te.size:
        print("[DBG] any NaN/Inf in X_te:",
              np.isnan(X_te).any(), np.isinf(X_te).any(),
              "min/max:", np.nanmin(X_te), np.nanmax(X_te))

    # 6) Standardisation par actif (z-score) calculée sur X_tr seulement
    X_tr_std, X_te_std = zscore_by_asset_block(X_tr, X_te, asset_feat_slices)

    # === DBG (après standardisation) ===
    if X_tr_std.size:
        print("[DBG] any NaN/Inf in X_tr_std:",
              np.isnan(X_tr_std).any(), np.isinf(X_tr_std).any(),
              "min/max:", np.nanmin(X_tr_std), np.nanmax(X_tr_std))
    if X_te_std.size:
        print("[DBG] any NaN/Inf in X_te_std:",
              np.isnan(X_te_std).any(), np.isinf(X_te_std).any(),
              "min/max:", np.nanmin(X_te_std), np.nanmax(X_te_std))

    # 7) Dump pickle au format attendu par gossip.read_feature (features en [N,F,T])
    meta = {
        "window": args.window,
        "prospective": args.prospective,
        "recall_period": args.recall_period,
        "selector": selector,
        "assets": assets,
        "F_total": F_total,
        "M": M,
        "N_total_windows": N,
        "days_for_test": args.days_for_test,
        "dates_span": (all_dates[0] if all_dates else None, all_dates[-1] if all_dates else None),
    }

    with open(args.out, "wb") as fp:
        train_map = {
            "code": "PANEL",
            "feature": X_tr_std.astype(np.float32),  # (N_tr, F_total, window)
            "label": Y_tr.astype(np.float32),        # (N_tr, M)
            "assets": assets,
            "meta": meta
        }
        pickle.dump(train_map, fp, protocol=pickle.HIGHEST_PROTOCOL)

        test_map = {
            "code": "PANEL",
            "feature": X_te_std.astype(np.float32),  # (N_te, F_total, window)
            "label": Y_te.astype(np.float32),        # (N_te, M)
            "assets": assets,
            "meta": meta
        }
        pickle.dump(test_map, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Écrit → {os.path.abspath(args.out)}")
    print(f"     Train: X={X_tr_std.shape}, y={Y_tr.shape}")
    print(f"     Test : X={X_te_std.shape}, y={Y_te.shape}")
    print(f"     Actifs: {assets}")


if __name__ == "__main__":
    main()