# --- rawdata_io.py ----------------------------------------------------
from typing import List, Dict
from dataclasses import dataclass
import csv
from datetime import datetime

# Même structure et ordre d'arguments que dans ton projet :
# RawData(date, open, high, close, low, volume)
@dataclass
class RawData:
    date: str
    open: float
    high: float
    close: float
    low: float
    volume: float


def read_rawdata_tsv(path: str, sep: str = "\t") -> List[RawData]:
    """
    Lit un fichier tab-séparé avec en-tête :
        date  open  high  close  low  volume
    Renvoie : List[RawData] triée par date (ISO 'YYYY-MM-DD').
    - Tolère espaces, lignes vides, et timestamps ISO (YYYY-MM-DD HH...).
    - Ignore les lignes invalides.
    """
    required = ["date", "open", "high", "close", "low", "volume"]
    rows: List[RawData] = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=sep)

        header = next(reader, None)
        if header is None:
            raise ValueError(f"Fichier vide : {path}")

        cols = [h.strip().lower() for h in header]
        name2idx: Dict[str, int] = {}
        for k in required:
            if k not in cols:
                raise ValueError(
                    f"Colonne manquante dans {path}. "
                    f"Trouvées={set(cols)} ; attendues ⊇ {set(required)}"
                )
            name2idx[k] = cols.index(k)

        for parts in reader:
            # skip lignes vides / blanches
            if not parts or all(not p.strip() for p in parts):
                continue

            try:
                date_raw = parts[name2idx["date"]].strip()
                if not date_raw or date_raw.lower() == "date":
                    continue

                # parse date robuste → ISO 'YYYY-MM-DD'
                if any(ch in date_raw for ch in ("T", " ", "Z", "+")):
                    d = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
                    date_iso = d.date().isoformat()
                else:
                    d = datetime.strptime(date_raw, "%Y-%m-%d")
                    date_iso = d.date().isoformat()

                o = float(parts[name2idx["open"]])
                h = float(parts[name2idx["high"]])
                c = float(parts[name2idx["close"]])
                l = float(parts[name2idx["low"]])
                v = float(parts[name2idx["volume"]])

                rows.append(RawData(date_iso, o, h, c, l, v))
            except Exception:
                # ligne invalide -> on ignore
                continue

    # Tri (ISO YYYY-MM-DD => tri lexicographique OK)
    rows.sort(key=lambda r: r.date)
    return rows


def load_raw_data_dict(file_map: Dict[str, str], sep: str = "\t") -> Dict[str, List[RawData]]:
    """
    Convertit un mapping {ticker: chemin_tsv} en {ticker: [RawData, ...]}.
    Prêt à être passé à align_and_merge(raw_data_dict=...).
    """
    return {sym: read_rawdata_tsv(path, sep=sep) for sym, path in file_map.items()}
