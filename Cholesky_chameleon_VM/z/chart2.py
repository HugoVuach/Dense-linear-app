import numpy as np
import talib
import pandas as pd
import math
from collections import defaultdict


class ChartFeature:
    """
    Gestionnaire d'extraction de features techniques pour UN SEUL actif.
    On garde la logique de l'ancien code (RSI, MACD, etc.)
    """

    def __init__(self, selector, recall_period=2, prospective=1):
        self.selector = selector
        self.supported = {
            "ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP",
            "BOLL", "MA", "VMA", "PRICE_VOLUME", "CROSS_PRICE"
        }
        self.feature = []
        self.prospective = prospective  # horizon de prédiction
        self.recall_period = recall_period

    def make_label(self, prices):
        """ Rendement futur pondéré (moyenne décroissante). """
        ratio = 0.5
        decay = 0.9
        label = 0.0
        for i in range(1, len(prices)):
            label = label + (prices[i] / prices[0] - 1.0) * ratio
            ratio = ratio * decay
        return label / (len(prices) - 1)

    def shift(self, xs, n):
        """ Décalage temporel pour créer des features croisés. """
        if n > 0:
            return np.r_[np.full(n, np.nan), xs[:-n]]
        elif n == 0:
            return xs
        else:
            return np.r_[xs[-n:], np.full(-n, np.nan)]

    def extract(self, open_prices, close_prices, high_prices, low_prices, volumes):
        """ Extraction brute des features d’un actif. """
        self.feature = []
        for feature_type in self.selector:
            if feature_type in self.supported:
                self.extract_by_type(
                    feature_type,
                    open_prices=open_prices, close_prices=close_prices,
                    high_prices=high_prices, low_prices=low_prices, volumes=volumes
                )
            else:
                print(f"[WARN] feature non supportée: {feature_type}")
        return self.feature

    def extract_by_type(self, feature_type, open_prices, close_prices, high_prices, low_prices, volumes):
        """
        Même logique que ton code original, mais je laisse inchangé
        (trop long à recopier ici).
        ↪ Tu gardes toutes les features ROCP, RSI, MACD, etc.
        """
        # ⚠️ garde ton contenu original ici
        pass


def align_and_merge(raw_data_dict, selector, window=30, with_label=True, flatten=False):
    """
    ⚡ Fonction centrale pour Cross-Asset Joint Forecasting
    - Prend un dict {ticker: raw_data[]} où raw_data est une liste d’objets avec date/open/high/low/close/volume
    - Aligne toutes les séries sur un calendrier global
    - Calcule les features et labels multi-actifs
    """

    # === 1. Transformer en DataFrames ===
    dfs = {}
    for code, series in raw_data_dict.items():
        df = pd.DataFrame([{
            "date": item.date,
            "open": item.open,
            "high": item.high,
            "low": item.low,
            "close": item.close,
            "volume": float(item.volume)
        } for item in series])
        df = df.sort_values("date").set_index("date")
        dfs[code] = df

    # === 2. Construire le calendrier global (union de toutes les dates) ===
    all_dates = sorted(set().union(*[df.index for df in dfs.values()]))

    # === 3. Reindexer chaque actif sur ce calendrier (forward-fill puis 0 init) ===
    for code in dfs:
        dfs[code] = dfs[code].reindex(all_dates).fillna(method="ffill").fillna(0.0)

    # === 4. Extraire features par actif ===
    features_by_asset = {}
    labels_by_asset = {}
    for code, df in dfs.items():
        cf = ChartFeature(selector)
        feat = cf.extract(
            df["open"].values, df["close"].values,
            df["high"].values, df["low"].values,
            df["volume"].values
        )
        feat = np.asarray(feat)  # shape (F, T)
        features_by_asset[code] = feat

        if with_label:
            labels = []
            closes = df["close"].values
            for t in range(window, len(closes) - cf.prospective):
                future = closes[t: t + cf.prospective + 1]
                labels.append(cf.make_label(future))
            labels_by_asset[code] = np.asarray(labels)

    # === 5. Standardiser par actif ===
    for code in features_by_asset:
        f = features_by_asset[code]
        f = np.nan_to_num(f)
        f = (f - f.mean(axis=1, keepdims=True)) / (f.std(axis=1, keepdims=True) + 1e-8)
        features_by_asset[code] = f

    # === 6. Construire tenseur final (N, window, M×F) ===
    asset_codes = list(features_by_asset.keys())
    M = len(asset_codes)
    F = features_by_asset[asset_codes[0]].shape[0]
    T = features_by_asset[asset_codes[0]].shape[1]

    X = []
    y = []
    for t in range(window, T):
        block_t = []
        y_t = []
        for code in asset_codes:
            f = features_by_asset[code][:, t - window:t]  # (F, window)
            block_t.append(f.T)  # transpose pour (window, F)
            if with_label:
                y_t.append(labels_by_asset[code][t - window])
        X.append(np.concatenate(block_t, axis=1))  # concat bloc par actif → (window, M×F)
        if with_label:
            y.append(y_t)
    X = np.asarray(X)  # (N, window, M×F)
    y = np.asarray(y) if with_label else None  # (N, M)

    return X, y, asset_codes
