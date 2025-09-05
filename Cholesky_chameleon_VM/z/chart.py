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

        if feature_type == 'ROCP':
            for i in range(1, self.recall_period):
                self.feature.append(talib.ROCP(close_prices, timeperiod=i) / float(i))
        if feature_type == 'OROCP':
            for i in range(1, self.recall_period):
                self.feature.append(talib.ROCP(open_prices, timeperiod=i) / float(i))
        if feature_type == 'HROCP':
            for i in range(1, self.recall_period):
                self.feature.append(talib.ROCP(high_prices, timeperiod=i) / float(i))
        if feature_type == 'LROCP':
            for i in range(1, self.recall_period):
                self.feature.append(talib.ROCP(low_prices, timeperiod=i) / float(i))
        if feature_type == 'MACD':
            macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            norm_signal = numpy.minimum(numpy.maximum(numpy.nan_to_num(signal), -1), 1)
            norm_hist = numpy.minimum(numpy.maximum(numpy.nan_to_num(hist), -1), 1)
            norm_macd = numpy.minimum(numpy.maximum(numpy.nan_to_num(macd), -1), 1)

            zero = numpy.asarray([0])
            macdrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(macd)))), -1),
                                     1)
            signalrocp = numpy.minimum(
                numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(signal)))), -1), 1)
            histrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(hist)))), -1),
                                     1)

            self.feature.append(norm_macd)
            self.feature.append(norm_signal)
            self.feature.append(norm_hist)

            self.feature.append(macdrocp)
            self.feature.append(signalrocp)
            self.feature.append(histrocp)
        if feature_type == 'RSI':
            rsi6 = talib.RSI(close_prices, timeperiod=6)
            rsi12 = talib.RSI(close_prices, timeperiod=12)
            rsi24 = talib.RSI(close_prices, timeperiod=24)
            rsi6rocp = talib.ROCP(rsi6 + 100., timeperiod=1)
            rsi12rocp = talib.ROCP(rsi12 + 100., timeperiod=1)
            rsi24rocp = talib.ROCP(rsi24 + 100., timeperiod=1)
            self.feature.append(rsi6 / 100.0 - 0.5)
            self.feature.append(rsi12 / 100.0 - 0.5)
            self.feature.append(rsi24 / 100.0 - 0.5)
            self.feature.append(rsi6rocp)
            self.feature.append(rsi12rocp)
            self.feature.append(rsi24rocp)
        if feature_type == 'VROCP':
            for i in range(1, self.recall_period):
                self.feature.append(numpy.arctan(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=i))))
        if feature_type == 'BOLL':
            upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            self.feature.append((upperband - close_prices) / close_prices)
            self.feature.append((middleband - close_prices) / close_prices)
            self.feature.append((lowerband - close_prices) / close_prices)
        if feature_type == 'MA':
            ma5 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=5))
            ma10 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=10))
            ma20 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=20))
            ma30 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=30))
            ma60 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=60))
            ma90 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=90))
            ma120 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=120))
            ma180 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=180))
            ma360 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=360))
            ma720 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=720))
            ma5rocp = talib.ROCP(ma5, timeperiod=1)
            ma10rocp = talib.ROCP(ma10, timeperiod=1)
            ma20rocp = talib.ROCP(ma20, timeperiod=1)
            ma30rocp = talib.ROCP(ma30, timeperiod=1)
            ma60rocp = talib.ROCP(ma60, timeperiod=1)
            ma90rocp = talib.ROCP(ma90, timeperiod=1)
            ma120rocp = talib.ROCP(ma120, timeperiod=1)
            ma180rocp = talib.ROCP(ma180, timeperiod=1)
            ma360rocp = talib.ROCP(ma360, timeperiod=1)
            ma720rocp = talib.ROCP(ma720, timeperiod=1)
            self.feature.append(ma5rocp)
            self.feature.append(ma10rocp)
            self.feature.append(ma20rocp)
            self.feature.append(ma30rocp)
            self.feature.append(ma60rocp)
            self.feature.append(ma90rocp)
            self.feature.append(ma120rocp)
            self.feature.append(ma180rocp)
            self.feature.append(ma360rocp)
            self.feature.append(ma720rocp)
            self.feature.append((ma5 - close_prices) / close_prices)
            self.feature.append((ma10 - close_prices) / close_prices)
            self.feature.append((ma20 - close_prices) / close_prices)
            self.feature.append((ma30 - close_prices) / close_prices)
            self.feature.append((ma60 - close_prices) / close_prices)
            self.feature.append((ma90 - close_prices) / close_prices)
            self.feature.append((ma120 - close_prices) / close_prices)
            self.feature.append((ma180 - close_prices) / close_prices)
            self.feature.append((ma360 - close_prices) / close_prices)
            self.feature.append((ma720 - close_prices) / close_prices)
        if feature_type == 'VMA':
            ma5 = numpy.nan_to_num(talib.MA(volumes, timeperiod=5))
            ma10 = numpy.nan_to_num(talib.MA(volumes, timeperiod=10))
            ma20 = numpy.nan_to_num(talib.MA(volumes, timeperiod=20))
            ma30 = numpy.nan_to_num(talib.MA(volumes, timeperiod=30))
            ma60 = numpy.nan_to_num(talib.MA(volumes, timeperiod=60))
            ma90 = numpy.nan_to_num(talib.MA(volumes, timeperiod=90))
            ma120 = numpy.nan_to_num(talib.MA(volumes, timeperiod=120))
            ma180 = numpy.nan_to_num(talib.MA(volumes, timeperiod=180))
            ma360 = numpy.nan_to_num(talib.MA(volumes, timeperiod=360))
            ma720 = numpy.nan_to_num(talib.MA(volumes, timeperiod=720))
            ma5rocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(ma5, timeperiod=1)))
            ma10rocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(ma10, timeperiod=1)))
            ma20rocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(ma20, timeperiod=1)))
            ma30rocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(ma30, timeperiod=1)))
            ma60rocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(ma60, timeperiod=1)))
            ma90rocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(ma90, timeperiod=1)))
            ma120rocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(ma120, timeperiod=1)))
            ma180rocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(ma180, timeperiod=1)))
            ma360rocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(ma360, timeperiod=1)))
            ma720rocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(ma720, timeperiod=1)))
            self.feature.append(ma5rocp)
            self.feature.append(ma10rocp)
            self.feature.append(ma20rocp)
            self.feature.append(ma30rocp)
            self.feature.append(ma60rocp)
            self.feature.append(ma90rocp)
            self.feature.append(ma120rocp)
            self.feature.append(ma180rocp)
            self.feature.append(ma360rocp)
            self.feature.append(ma720rocp)
            self.feature.append(numpy.tanh(numpy.nan_to_num((ma5 - volumes) / (volumes + 1))))
            self.feature.append(numpy.tanh(numpy.nan_to_num((ma10 - volumes) / (volumes + 1))))
            self.feature.append(numpy.tanh(numpy.nan_to_num((ma20 - volumes) / (volumes + 1))))
            self.feature.append(numpy.tanh(numpy.nan_to_num((ma30 - volumes) / (volumes + 1))))
            self.feature.append(numpy.tanh(numpy.nan_to_num((ma60 - volumes) / (volumes + 1))))
            self.feature.append(numpy.tanh(numpy.nan_to_num((ma90 - volumes) / (volumes + 1))))
            self.feature.append(numpy.tanh(numpy.nan_to_num((ma120 - volumes) / (volumes + 1))))
            self.feature.append(numpy.tanh(numpy.nan_to_num((ma180 - volumes) / (volumes + 1))))
            self.feature.append(numpy.tanh(numpy.nan_to_num((ma360 - volumes) / (volumes + 1))))
            self.feature.append(numpy.tanh(numpy.nan_to_num((ma720 - volumes) / (volumes + 1))))
        if feature_type == 'PRICE_VOLUME':
            rocp = talib.ROCP(close_prices, timeperiod=1)
            # norm_volumes = (volumes - numpy.mean(volumes)) / math.sqrt(numpy.var(volumes))
            # vrocp = talib.ROCP(norm_volumes + numpy.max(norm_volumes) - numpy.min(norm_volumes), timeperiod=1)
            vrocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=1)))
            pv = rocp * vrocp
            self.feature.append(pv)
        if feature_type == 'CROSS_PRICE':
            for i in range(0, self.recall_period - 1):
                shift_open = self.shift(open_prices, i)
                self.feature.append(numpy.minimum(numpy.maximum(numpy.nan_to_num((close_prices - shift_open) / shift_open), -1), 1))
                self.feature.append(numpy.minimum(numpy.maximum(numpy.nan_to_num((high_prices - shift_open) / shift_open), -1), 1))
                self.feature.append(numpy.minimum(numpy.maximum(numpy.nan_to_num((low_prices - shift_open) / shift_open), -1), 1))
                shift_close = self.shift(close_prices, i)
                self.feature.append(numpy.minimum(numpy.maximum(numpy.nan_to_num((open_prices - shift_close) / shift_close), -1), 1))
                self.feature.append(numpy.minimum(numpy.maximum(numpy.nan_to_num((high_prices - shift_close) / shift_close), -1), 1))
                self.feature.append(numpy.minimum(numpy.maximum(numpy.nan_to_num((low_prices - shift_close) / shift_close), -1), 1))
                shift_high = self.shift(high_prices, i)
                self.feature.append(numpy.minimum(numpy.maximum(numpy.nan_to_num((close_prices - shift_high) / shift_high), -1), 1))
                self.feature.append(numpy.minimum(numpy.maximum(numpy.nan_to_num((open_prices - shift_high) / shift_high), -1), 1))
                self.feature.append(numpy.minimum(numpy.maximum(numpy.nan_to_num((low_prices - shift_high) / shift_high), -1), 1))
                shift_low = self.shift(low_prices, i)
                self.feature.append(numpy.minimum(numpy.maximum(numpy.nan_to_num((close_prices - shift_low) / shift_low), -1), 1))
                self.feature.append(numpy.minimum(numpy.maximum(numpy.nan_to_num((high_prices - shift_low) / shift_low), -1), 1))
                self.feature.append(numpy.minimum(numpy.maximum(numpy.nan_to_num((open_prices - shift_low) / shift_low), -1), 1))
        pass


def align_and_merge(raw_data_dict, selector, window=30, with_label=True, flatten=False):
    """
    ⚡ Fonction centrale pour Cross-Asset Joint Forecasting
    - Prend un dict {ticker: raw_data[]} où raw_data est une liste d’objets avec date/open/high/low/close/volume
    - Aligne toutes les séries sur un calendrier global
    - Calcule les features et labels multi-actifs
    Input : dict d’actifs → DataFrame avec OHLCV
    Output X : fenêtres glissantes de features alignés (multi-actifs, multi-features)
    Output y : labels construits selon ton labeler (ex. %change du close de AAPL)
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
