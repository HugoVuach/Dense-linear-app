#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère un dossier `dataset/` rempli de fichiers TSV compatibles avec rawdata.py :
colonnes : date\topen\thigh\tclose\tlow\tvolume

Univers disponibles :
  - indices   : grands indices US (Yahoo '^' -> nom de fichier nettoyé)
  - bluechips : leaders US (SP100)
  - sectors   : actions emblématiques par secteur (diversifié)
  - etf       : ETF représentatifs (marché & secteurs)

Exemples d’usage :
  python generate_dataset.py --universe indices --start 2012-01-01 --end 2025-09-01
  python generate_dataset.py --universe bluechips
  python generate_dataset.py --universe etf --outdir dataset_us
  python generate_dataset.py --universe sectors --tickers AAPL,MSFT,NVDA,AMZN   # override manuel


  python generate_dataset.py --universe etf --start 2015-01-01
python generate_dataset.py --universe indices --start 2000-01-01
python generate_dataset.py --universe bluechips --start 2010-01-01
python generate_dataset.py --universe sector --start 2012-01-01

Dépendances :
  pip install yfinance pandas numpy
"""

import argparse
import os
import re
import sys
from typing import List, Dict

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Veuillez installer yfinance :  pip install yfinance", file=sys.stderr)
    sys.exit(1)


# ---------- 1) Univers par défaut ----------

DEFAULT_UNIVERSES: Dict[str, List[str]] = {
    # Yahoo Finance indices (préfixe ^). On les garde tels quels pour le download,
    # mais on nettoie les noms de fichiers sur disque.
    "indices": [
        "^GSPC",   # S&P 500
        "^NDX",    # Nasdaq 100
        "^DJI",    # Dow Jones
        "^RUT",    # Russell 2000
        "^VIX",    # Volatility Index
    ],
    # Blue chips (leaders liquides)
    "l": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META",
        "NVDA", "TSLA", "JPM", "JNJ", "XOM",
        "BRK-B",  # Berkshire (Yahoo utilise BRK-B)
        "PG",
    ],
    # Actions emblématiques par secteur (diversifié)
    "sectors": [
        # Tech
        "AAPL", "MSFT", "NVDA", "AVGO",
        # Finance
        "JPM", "GS", "BAC",
        # Énergie
        "XOM", "CVX",
        # Santé
        "JNJ", "PFE", "UNH",
        # Conso discrétionnaire
        "AMZN", "HD", "MCD",
        # Industriels
        "CAT", "BA",
        # Conso de base
        "PG", "KO",
        # Télécom/Comm Services
        "GOOGL", "META",
        # Utilities
        "NEE", "DUK",
        # Real Estate
        "PLD", "AMT",
    ],
    # ETF représentatifs
    "etf": [
        "SPY", "QQQ", "IWM",               # marché
        "XLK", "XLF", "XLE", "XLV",        # secteurs
        "XLY", "XLP", "XLI", "XLU", "XLRE",
        "GLD", "SLV",                      # métaux
        "TLT", "IEF",                      # obligations US
        "DIA"                              # Dow ETF
    ],
}


# ---------- 2) Utilitaires ----------

def sanitize_filename(ticker: str) -> str:
    """
    Nettoie le ticker pour un nom de fichier sûr :
      - Remplace tout caractère non alphanumérique par '_'
      - Exemple : '^GSPC' -> '_GSPC', 'BRK-B' -> 'BRK_B'
    """
    return re.sub(r'\W+', '_', ticker).strip('_')


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Télécharge OHLCV via yfinance. Utilise Ajd Close pour 'close'.
    Renvoie un DataFrame avec colonnes : date, open, high, close, low, volume (trié).
    """
    # yfinance: .history() renvoie un DF indexé par datetime
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"Aucune donnée pour {ticker} sur [{start}, {end}]")

    # Colonnes attendues
    # Si 'Adj Close' indisponible (rare), fallback sur 'Close'
    if "Adj Close" in df.columns:
        close = df["Adj Close"]
    else:
        close = df["Close"]

    out = pd.DataFrame({
        "date": df.index.tz_localize(None).date,  # dates naïves
        "open": df["Open"].astype(float),
        "high": df["High"].astype(float),
        "close": close.astype(float),   # prix ajusté
        "low": df["Low"].astype(float),
        "volume": df["Volume"].astype(np.float64),  # volume peut être grand
    })

    # Nettoyage / tri / drop NaN
    out = out.dropna().sort_values("date")
    # Assure ordre & types
    out["date"] = out["date"].astype(str)  # format YYYY-MM-DD
    out["open"] = out["open"].astype(float)
    out["high"] = out["high"].astype(float)
    out["close"] = out["close"].astype(float)
    out["low"] = out["low"].astype(float)
    out["volume"] = out["volume"].astype(float)

    return out


def save_tsv(df: pd.DataFrame, out_path: str) -> None:
    """
    Écrit le DataFrame au format TSV attendu par rawdata.py
    (en-tête : date, open, high, close, low, volume ; séparateur '\t').
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False,
              columns=["date", "open", "high", "close", "low", "volume"])


# ---------- 3) Main ----------

def main():
    parser = argparse.ArgumentParser(description="Génère un dossier dataset/ selon l'univers choisi.")
    parser.add_argument("--universe",
                        type=str,
                        choices=["indices", "bluechips", "sectors", "etf"],
                        required=True,
                        help="Type d’univers à générer.")
    parser.add_argument("--tickers",
                        type=str,
                        default=None,
                        help="Liste manuelle (comma-separated) pour override, ex: AAPL,MSFT,NVDA")
    parser.add_argument("--start", type=str, default="2012-01-01", help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end",   type=str, default=None,         help="Date de fin (YYYY-MM-DD), défaut = aujourd’hui")
    parser.add_argument("--outdir", type=str, default="dataset",   help="Répertoire de sortie")
    parser.add_argument("--fail_fast", action="store_true",
                        help="Arrêter au premier échec (par défaut on continue et on log les erreurs)")
    args = parser.parse_args()

    # Détermination de la liste de tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = DEFAULT_UNIVERSES[args.universe]

    print(f"Univers : {args.universe}")
    print(f"Tickers ({len(tickers)}) : {', '.join(tickers)}")
    print(f"Période : {args.start} → {args.end or 'today'}")
    print(f"Output  : {args.outdir}")
    os.makedirs(args.outdir, exist_ok=True)

    # Téléchargement & écriture
    ok, ko = [], []
    for t in tickers:
        try:
            df = fetch_ohlcv(t, start=args.start, end=args.end)
            if df.empty:
                raise ValueError("DataFrame vide après nettoyage.")
            fname = sanitize_filename(t) + ".csv"
            out_path = os.path.join(args.outdir, fname)
            save_tsv(df, out_path)
            print(f"✔ {t:10s} -> {out_path}  ({len(df)} lignes)")
            ok.append(t)
        except Exception as e:
            msg = f"✖ {t:10s} -> erreur : {e}"
            print(msg, file=sys.stderr)
            ko.append((t, str(e)))
            if args.fail_fast:
                break

    # Résumé
    print("\n=== RÉSUMÉ ===")
    print(f"Succès : {len(ok)} / {len(tickers)}")
    if ko:
        print("Échecs :")
        for t, err in ko:
            print(f" - {t}: {err}")

    # Petit fichier meta (facultatif mais utile)
    try:
        meta = pd.DataFrame({
            "ticker": tickers,
            "file": [os.path.join(args.outdir, sanitize_filename(t) + ".csv") for t in tickers],
            "universe": args.universe
        })
        meta.to_csv(os.path.join(args.outdir, "_meta_universe.tsv"), sep="\t", index=False)
    except Exception as e:
        print(f"[WARN] impossible d’écrire _meta_universe.tsv : {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

