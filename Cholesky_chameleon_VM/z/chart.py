import numpy 
import talib
import pandas as pd
import math
from collections import defaultdict


# ---------------------------------------------------------------------
# Debug helper (unique)
# ---------------------------------------------------------------------
def dbg(tag, arr):
    """
    Affiche un résumé robuste d'un array (shape, NaN/Inf/Zéros, min/max/mean/std).
    Utilisation: après chaque calcul d'un vecteur de features.
    """
    a = numpy.asarray(arr)
    n = a.size
    n_nan  = int(numpy.isnan(a).sum())
    n_inf  = int(numpy.isinf(a).sum())
    n_zero = int((a == 0).sum()) if a.dtype.kind in "fc" else 0  # seulement float/complex
    amin = float(numpy.nanmin(a)) if n_nan < n else float("nan")
    amax = float(numpy.nanmax(a)) if n_nan < n else float("nan")
    amean = float(numpy.nanmean(a)) if n_nan < n else float("nan")
    astd  = float(numpy.nanstd(a))  if n_nan < n else float("nan")
    print(f"[DBG] {tag:22s} shape={a.shape}, nan={n_nan}/{n}, inf={n_inf}, zero={n_zero}, "
          f"min={amin:.4g}, max={amax:.4g}, mean={amean:.4g}, std={astd:.4g}")



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
        base = prices[0]
        if base == 0:
            return 0.0
        for i in range(1, len(prices)):
            label = label + (prices[i] / base - 1.0) * ratio
            ratio = ratio * decay
        return label / (len(prices) - 1)

    def shift(self, xs, n):
        """ Décalage temporel pour créer des features croisés. """
        if n > 0:
            return numpy.r_[numpy.full(n, numpy.nan), xs[:-n]]
        elif n == 0:
            return xs
        else:
            return numpy.r_[xs[-n:], numpy.full(-n, numpy.nan)]

    def extract(self, open_prices, close_prices, high_prices, low_prices, volumes):
        """ Extraction brute des features d’un actif. """
        self.feature = []
        
        # sanity-check de base
        if numpy.any(close_prices == 0.0):
            print(f"[WARN] close_prices contient {int((close_prices == 0.0).sum())} zéros → attention aux divisions.")

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
                arr = talib.ROCP(close_prices, timeperiod=i) / float(i)
                self.feature.append(arr); dbg(f"ROCP/{i}", arr)

        if feature_type == 'OROCP':
            for i in range(1, self.recall_period):
                arr = talib.ROCP(open_prices, timeperiod=i) / float(i)
                self.feature.append(arr); dbg(f"OROCP/{i}", arr)

        if feature_type == 'HROCP':
            for i in range(1, self.recall_period):
                arr = talib.ROCP(high_prices, timeperiod=i) / float(i)
                self.feature.append(arr); dbg(f"HROCP/{i}", arr)

        if feature_type == 'LROCP':
            for i in range(1, self.recall_period):
                arr = talib.ROCP(low_prices, timeperiod=i) / float(i)
                self.feature.append(arr); dbg(f"LROCP/{i}", arr)

        if feature_type == 'MACD':
            macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            norm_signal = numpy.clip(numpy.nan_to_num(signal), -1, 1)
            norm_hist = numpy.clip(numpy.nan_to_num(hist), -1, 1)
            norm_macd = numpy.clip(numpy.nan_to_num(macd), -1, 1)

            zero = numpy.asarray([0])
            macdrocp = numpy.clip(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(macd)))), -1, 1)
            signalrocp = numpy.clip(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(signal)))), -1, 1)
            histrocp = numpy.clip(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(hist)))), -1, 1)

            self.feature.append(norm_macd);   dbg("MACD", norm_macd)
            self.feature.append(norm_signal); dbg("MACD_signal", norm_signal)
            self.feature.append(norm_hist);   dbg("MACD_hist", norm_hist)
            self.feature.append(macdrocp);    dbg("MACD_rocp", macdrocp)
            self.feature.append(signalrocp);  dbg("MACD_signal_rocp", signalrocp)
            self.feature.append(histrocp);    dbg("MACD_hist_rocp", histrocp)

        if feature_type == 'RSI':
            rsi6 = talib.RSI(close_prices, timeperiod=6)
            rsi12 = talib.RSI(close_prices, timeperiod=12)
            rsi24 = talib.RSI(close_prices, timeperiod=24)
            rsi6rocp = talib.ROCP(rsi6 + 100., timeperiod=1)
            rsi12rocp = talib.ROCP(rsi12 + 100., timeperiod=1)
            rsi24rocp = talib.ROCP(rsi24 + 100., timeperiod=1)

            f1 = rsi6  / 100.0 - 0.5;  self.feature.append(f1); dbg("RSI6_n",  f1)
            f2 = rsi12 / 100.0 - 0.5;  self.feature.append(f2); dbg("RSI12_n", f2)
            f3 = rsi24 / 100.0 - 0.5;  self.feature.append(f3); dbg("RSI24_n", f3)
            self.feature.append(rsi6rocp);   dbg("RSI6_rocp",  rsi6rocp)
            self.feature.append(rsi12rocp);  dbg("RSI12_rocp", rsi12rocp)
            self.feature.append(rsi24rocp);  dbg("RSI24_rocp", rsi24rocp)

        if feature_type == 'VROCP':
            for i in range(1, self.recall_period):
                arr = numpy.arctan(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=i)))
                self.feature.append(arr); dbg(f"VROCP/{i}", arr)        
    
        if feature_type == 'BOLL':
            upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            denom = numpy.where(close_prices == 0, 1.0, close_prices)
            f1 = (upperband  - close_prices) / denom
            f2 = (middleband - close_prices) / denom
            f3 = (lowerband  - close_prices) / denom
            self.feature.append(f1); dbg("BOLL_up",  f1)
            self.feature.append(f2); dbg("BOLL_mid", f2)
            self.feature.append(f3); dbg("BOLL_low", f3)

    
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

            for name, m in [("ma5", ma5), ("ma10", ma10), ("ma20", ma20), ("ma30", ma30),
                            ("ma60", ma60), ("ma90", ma90), ("ma120", ma120), ("ma180", ma180),
                            ("ma360", ma360), ("ma720", ma720)]:
                r = talib.ROCP(m, timeperiod=1)
                self.feature.append(r); dbg(f"{name}_rocp", r)
            
            denom = numpy.where(close_prices == 0, 1.0, close_prices)
            for name, m in [("ma5", ma5), ("ma10", ma10), ("ma20", ma20), ("ma30", ma30),
                            ("ma60", ma60), ("ma90", ma90), ("ma120", ma120), ("ma180", ma180),
                            ("ma360", ma360), ("ma720", ma720)]:
                d = (m - close_prices) / denom
                self.feature.append(d); dbg(f"{name}_minus_close", d)

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

            for name, m in [("vma5", ma5), ("vma10", ma10), ("vma20", ma20), ("vma30", ma30),
                            ("vma60", ma60), ("vma90", ma90), ("vma120", ma120), ("vma180", ma180),
                            ("vma360", ma360), ("vma720", ma720)]:
                r = numpy.tanh(numpy.nan_to_num(talib.ROCP(m, timeperiod=1)))
                self.feature.append(r); dbg(f"{name}_rocp", r)

            
            denom = volumes + 1.0
            for name, m in [("vma5", ma5), ("vma10", ma10), ("vma20", ma20), ("vma30", ma30),
                            ("vma60", ma60), ("vma90", ma90), ("vma120", ma120), ("vma180", ma180),
                            ("vma360", ma360), ("vma720", ma720)]:
                d = numpy.tanh(numpy.nan_to_num((m - volumes) / denom))
                self.feature.append(d); dbg(f"{name}_minus_vol", d)


        if feature_type == 'PRICE_VOLUME':
            rocp  = talib.ROCP(close_prices, timeperiod=1)
            vrocp = numpy.tanh(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=1)))
            pv = rocp * vrocp
            self.feature.append(pv); dbg("PRICE_VOLUME", pv)

        if feature_type == 'CROSS_PRICE':
            for i in range(0, self.recall_period - 1):
                shift_open  = self.shift(open_prices,  i)
                shift_close = self.shift(close_prices, i)
                shift_high  = self.shift(high_prices,  i)
                shift_low   = self.shift(low_prices,   i)
                
                
                denom = numpy.where(shift_open == 0, 1.0, shift_open)
                f1 = numpy.clip(numpy.nan_to_num((close_prices - shift_open) / denom), -1, 1)
                f2 = numpy.clip(numpy.nan_to_num((high_prices  - shift_open) / denom), -1, 1)
                f3 = numpy.clip(numpy.nan_to_num((low_prices   - shift_open) / denom), -1, 1)
                self.feature.append(f1); dbg(f"CROSS o{i}: close-open", f1)
                self.feature.append(f2); dbg(f"CROSS o{i}: high-open",  f2)
                self.feature.append(f3); dbg(f"CROSS o{i}: low-open",   f3)

                denom = numpy.where(shift_close == 0, 1.0, shift_close)
                f4 = numpy.clip(numpy.nan_to_num((open_prices  - shift_close) / denom), -1, 1)
                f5 = numpy.clip(numpy.nan_to_num((high_prices  - shift_close) / denom), -1, 1)
                f6 = numpy.clip(numpy.nan_to_num((low_prices   - shift_close) / denom), -1, 1)
                self.feature.append(f4); dbg(f"CROSS c{i}: open-close",  f4)
                self.feature.append(f5); dbg(f"CROSS c{i}: high-close",  f5)
                self.feature.append(f6); dbg(f"CROSS c{i}: low-close",   f6)

                denom = numpy.where(shift_high == 0, 1.0, shift_high)
                f7 = numpy.clip(numpy.nan_to_num((close_prices - shift_high) / denom), -1, 1)
                f8 = numpy.clip(numpy.nan_to_num((open_prices  - shift_high) / denom), -1, 1)
                f9 = numpy.clip(numpy.nan_to_num((low_prices   - shift_high) / denom), -1, 1)
                self.feature.append(f7); dbg(f"CROSS h{i}: close-high", f7)
                self.feature.append(f8); dbg(f"CROSS h{i}: open-high",  f8)
                self.feature.append(f9); dbg(f"CROSS h{i}: low-high",   f9)

                denom = numpy.where(shift_low == 0, 1.0, shift_low)
                f10 = numpy.clip(numpy.nan_to_num((close_prices - shift_low) / denom), -1, 1)
                f11 = numpy.clip(numpy.nan_to_num((high_prices  - shift_low) / denom), -1, 1)
                f12 = numpy.clip(numpy.nan_to_num((open_prices  - shift_low) / denom), -1, 1)
                self.feature.append(f10); dbg(f"CROSS l{i}: close-low", f10)
                self.feature.append(f11); dbg(f"CROSS l{i}: high-low",  f11)
                self.feature.append(f12); dbg(f"CROSS l{i}: open-low",  f12)
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
        feat = numpy.asarray(feat)  # shape (F, T)
        features_by_asset[code] = feat

        if with_label:
            labels = []
            closes = df["close"].values
            for t in range(window, len(closes) - cf.prospective):
                future = closes[t: t + cf.prospective + 1]
                labels.append(cf.make_label(future))
            labels_by_asset[code] = numpy.asarray(labels)

    # === 5. Standardiser par actif ===
    for code in features_by_asset:
        f = features_by_asset[code]
        f = numpy.nan_to_num(f)
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
        X.append(numpy.concatenate(block_t, axis=1))  # concat bloc par actif → (window, M×F)
        if with_label:
            y.append(y_t)
    X = numpy.asarray(X)  # (N, window, M×F)
    y = numpy.asarray(y) if with_label else None  # (N, M)

    return X, y, asset_codes
