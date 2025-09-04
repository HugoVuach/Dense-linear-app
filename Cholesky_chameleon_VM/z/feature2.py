# feature.py — Cross-Asset Joint Forecasting
# ---------------------------------------------------------------
# - Lit tous les fichiers du dossier dataset/ (ou un sous-ensemble via --universe)
# - Aligne les dates (union) avec forward-fill et 0 au démarrage
# - Extrait des features TA-Lib par actif via ChartFeature.extract(...)
# - Concatène les blocs de features par actif : F_total = sum(F_actif)
# - Fenêtre glissante : X windows de shape (F_total, T_window) pour le dump
# - Labels multi-cibles : y ∈ R^{N_windows × M} (un rendement futur par actif)
# - Standardisation par actif (sur les fenêtres d’entraînement uniquement)
# - Sauvegarde 2 objets pickle (train_set, test_set) dans 'ultimate_feature'
#
# Sorties (par map pickle) :
#   {
#     "code": "PANEL",
#     "feature": np.ndarray (N,F_total,T_window),   # dump au format [N,F,T]
#     "label":   np.ndarray (N,M),                  # multi-target
#     "assets":  [ticker_1, ..., ticker_M],
#     "meta":    {...}
#   }

import os
import argparse
import pickle
import numpy as np
from collections import defaultdict

from rawdata import read_sample_data   # lit un TSV "\t" -> List[RawData] (date, open, high, close, low, volume)
from chart import ChartFeature         # on réutilise l’ingénierie de features unitaire

# --------------------
# Utilitaires internes
# --------------------

def make_label_from_prices(prices_slice: np.ndarray) -> float:
    """
    Reproduit ChartFeature.make_label :
    - base = prices_slice[0]
    - rendement(s) futur(s) pondéré(s) (ratio=0.5, decay=0.9), moyenne normalisée
    """
    ratio = 0.5
    decay = 0.9
    label = 0.0
    for i in range(1, len(prices_slice)):
        label += (prices_slice[i] / prices_slice[0] - 1.0) * ratio
        ratio *= decay
    return label / max(1, (len(prices_slice) - 1))


def forward_fill_align(all_dates, series_by_date, start_zero=True):
    """
    Aligne une série OHLCV sur all_dates (triées croissant).
    - series_by_date: dict(date -> (o,h,c,l,v))
    - forward-fill des trous ; si start_zero=True, initialise à 0 jusqu'à la 1ère valeur connue.
    """
    O, H, C, L, V = [], [], [], [], []
    last = None
    for d in all_dates:
        if d in series_by_date:
            last = series_by_date[d]
        if last is None:
            if start_zero:
                O.append(0.0); H.append(0.0); C.append(0.0); L.append(0.0); V.append(0.0)
            else:
                O.append(np.nan); H.append(np.nan); C.append(np.nan); L.append(np.nan); V.append(np.nan)
        else:
            o, h, c, l, v = last
            O.append(o); H.append(h); C.append(c); L.append(l); V.append(v)
    return (np.asarray(O, dtype=np.float64),
            np.asarray(H, dtype=np.float64),
            np.asarray(C, dtype=np.float64),
            np.asarray(L, dtype=np.float64),
            np.asarray(V, dtype=np.float64))


def build_windows_block(features_concat_FxT, closes_concat_list, window, prospective):
    """
    Construit des fenêtres X (liste de blocs [F_total, window]) et labels Y (liste de vecteurs longueur M).
    - features_concat_FxT : np.ndarray (F_total, T)
    - closes_concat_list  : list[np.ndarray(T,)] — un close par actif (dans le même ordre que les blocs)
    - returns: X_windows (N, F_total, window), Y_targets (N, M)
    """
    F_total, T = features_concat_FxT.shape
    M = len(closes_concat_list)
    # Nombre de fenêtres valides si le label regarde 'prospective' pas dans le futur
    N = max(0, T - window - prospective + 1)
    X_list, Y_list = [], []

    for p in range(N):
        # slice temporel
        x = features_concat_FxT[:, p:p + window]           # (F_total, window)
        # labels multi-actifs
        y_vec = []
        t0 = p + window - 1
        # label basé sur close[t0 : t0+prospective]
        for c in closes_concat_list:
            y_val = make_label_from_prices(c[t0:t0 + prospective + 1])
            y_vec.append(y_val)
        X_list.append(x)
        Y_list.append(y_vec)

    X = np.asarray(X_list, dtype=np.float32)   # (N, F_total, window)
    Y = np.asarray(Y_list, dtype=np.float32)   # (N, M)
    return X, Y


def zscore_by_asset_block(X_train, X_test, asset_feat_slices, eps=1e-8):
    """
    Standardisation par actif (z-score) sur les fenêtres d'entraînement uniquement.
    - X_train, X_test: np.ndarray (N, F_total, window)
    - asset_feat_slices: liste de tranches [(f0,f1), ...] pour chaque actif dans F_total
    """
    Xtr = X_train.copy()
    Xte = X_test.copy()
    for (f0, f1) in asset_feat_slices:
        # concatène toutes les valeurs de ce bloc sur N et T
        block_train = Xtr[:, f0:f1, :].reshape(-1)
        mu = np.nanmean(block_train)
        sd = np.nanstd(block_train)
        sd = sd if sd > eps else 1.0
        Xtr[:, f0:f1, :] = (Xtr[:, f0:f1, :] - mu) / sd
        Xte[:, f0:f1, :] = (Xte[:, f0:f1, :] - mu) / sd
    return Xtr, Xte


# --------------------
# Programme principal
# --------------------

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

    # 1) Lister les fichiers
    all_files = []
    for fn in os.listdir(args.dataset_dir):
        if fn.lower().endswith((".csv", ".tsv")):
            all_files.append(fn)
    if not all_files:
        raise FileNotFoundError(f"Aucun fichier .csv/.tsv dans {args.dataset_dir}")

    universe = [u.strip() for u in args.universe.split(",") if u.strip()]
    if universe:
        keep = set(universe)
        all_files = [fn for fn in all_files if os.path.splitext(fn)[0] in keep]

    # 2) Lire et construire un index global de dates
    assets = []
    per_asset_series = {}  # ticker -> dict(date -> (o,h,c,l,v))
    all_dates_set = set()

    for fn in sorted(all_files):
        ticker = os.path.splitext(fn)[0]
        path = os.path.join(args.dataset_dir, fn)
        rows = read_sample_data(path)  # -> List[RawData], triée par .date (string 'YYYY-MM-DD')
        if not rows:
            continue
        assets.append(ticker)
        dd = {}
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
    features_blocks = []      # liste de (F_actif, T)
    closes_by_asset = []      # liste de close(T,) pour labels
    asset_feat_slices = []    # mémorise les tranches (f0,f1) de chaque actif dans la concat finale
    f_cursor = 0

    for ticker in assets:
        series = per_asset_series[ticker]
        O, H, C, L, V = forward_fill_align(all_dates, series, start_zero=True)

        # Configurer l’extracteur unitaire
        cf = ChartFeature(selector)
        cf.recall_period = args.recall_period
        cf.prospective = args.prospective

        # Extrait par type -> self.feature: liste de vecteurs longueur T
        cf.extract(open_prices=O, close_prices=C, high_prices=H, low_prices=L, volumes=V)
        F_actif = len(cf.feature)
        feat_block = np.asarray(cf.feature, dtype=np.float64)  # (F_actif, T)
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

    # 4) Fenêtres + labels multi-cibles
    X_all, Y_all = build_windows_block(features_concat, closes_by_asset,
                                       window=args.window, prospective=args.prospective)
    N = X_all.shape[0]
    print(f"[INFO] N_windows={N} | X_all={X_all.shape} | Y_all={Y_all.shape}")

    # 5) Split train/test en *fenêtres*
    cut = max(0, N - args.days_for_test)
    X_tr, X_te = X_all[:cut], X_all[cut:]
    Y_tr, Y_te = Y_all[:cut], Y_all[cut:]

    # 6) Standardisation par actif (z-score) calculée sur X_tr seulement
    X_tr_std, X_te_std = zscore_by_asset_block(X_tr, X_te, asset_feat_slices)

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
        "days_for_test": args.days_for_test
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
