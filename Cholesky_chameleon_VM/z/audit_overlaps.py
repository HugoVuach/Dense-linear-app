# audit_overlaps.py
# Usage :
#   python audit_overlaps.py                 # audit du dossier ./dataset
#   python audit_overlaps.py --dir dataset   # dossier personnalisé
#   python audit_overlaps.py --csv rapport.csv  # export CSV des overlaps

import os
import argparse
from itertools import combinations
from typing import List, Tuple, Dict, Optional
from rawdata import read_sample_data  # <-- votre parser existant (TSV -> List[RawData])

def audit_file(path: str) -> Optional[Tuple[str, str, int]]:
    """
    Lit un fichier via read_sample_data et retourne (date_debut, date_fin, n_lignes).
    Si le fichier est vide/invalide, retourne None.
    """
    try:
        rows = read_sample_data(path)
    except Exception as e:
        print(f"[WARN] Lecture impossible: {path} ({e})")
        return None
    if not rows:
        print(f"[WARN] Fichier vide: {path}")
        return None
    start = rows[0].date
    end   = rows[-1].date
    n     = len(rows)
    return (start, end, n)

def overlap_range(a_start: str, a_end: str, b_start: str, b_end: str) -> Optional[Tuple[str, str]]:
    """
    Calcule l'intersection [max(start), min(end)] si elle existe.
    Les dates sont au format 'YYYY-MM-DD' (tri lexicographique OK).
    """
    start = max(a_start, b_start)
    end   = min(a_end, b_end)
    if start <= end:
        return (start, end)
    return None

def main():
    parser = argparse.ArgumentParser(description="Audit des plages temporelles et recouvrements de fichiers OHLCV.")
    parser.add_argument("--dir", type=str, default="dataset", help="Dossier contenant les .csv (TSV) à auditer.")
    parser.add_argument("--ext", type=str, default=".csv", help="Extension des fichiers à scanner (par défaut .csv).")
    parser.add_argument("--csv", type=str, default=None, help="Chemin d'export CSV des recouvrements (optionnel).")
    args = parser.parse_args()

    dataset_dir = args.dir
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dossier introuvable: {dataset_dir}")

    files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(args.ext.lower())]
    files.sort()
    if not files:
        print(f"Aucun fichier *{args.ext} trouvé dans {dataset_dir}")
        return

    print("=== Audit des plages par fichier ===")
    info: Dict[str, Tuple[str, str, int]] = {}  # filename -> (start, end, n)
    for fname in files:
        fpath = os.path.join(dataset_dir, fname)
        res = audit_file(fpath)
        if res is None:
            continue
        start, end, n = res
        info[fname] = (start, end, n)
        print(f"- {fname:30s}  {start}  →  {end}   ({n} lignes)")

    if not info:
        print("Aucune donnée exploitable.")
        return

    print("\n=== Recouvrements entre fichiers (paires) ===")
    overlaps: List[Tuple[str, str, str, str]] = []  # (fileA, fileB, ov_start, ov_end)
    for fa, fb in combinations(sorted(info.keys()), 2):
        a_start, a_end, _ = info[fa]
        b_start, b_end, _ = info[fb]
        ov = overlap_range(a_start, a_end, b_start, b_end)
        if ov:
            overlaps.append((fa, fb, ov[0], ov[1]))
            print(f"* {fa}  ↔  {fb}   ->   overlap: {ov[0]}  →  {ov[1]}")

    if not overlaps:
        print("Aucun recouvrement détecté.")

    # Export CSV optionnel
    if args.csv:
        try:
            import csv
            with open(args.csv, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["file_a", "file_b", "overlap_start", "overlap_end"])
                for fa, fb, s, e in overlaps:
                    w.writerow([fa, fb, s, e])
            print(f"\nCSV des recouvrements écrit : {args.csv}")
        except Exception as e:
            print(f"[WARN] Échec d'écriture CSV ({args.csv}) : {e}")

    # Bonus : résumé par "code" (nom sans suffixe)
    print("\n=== Résumé par code (nom avant extension) ===")
    by_code: Dict[str, List[Tuple[str, str, str]]] = {}
    for fname, (s, e, n) in info.items():
        code = fname[:fname.rfind(".")] if "." in fname else fname
        by_code.setdefault(code, []).append((fname, s, e))
    for code, entries in by_code.items():
        if len(entries) <= 1:
            continue
        entries.sort(key=lambda t: (t[1], t[2]))
        print(f"[{code}]")
        for (fn, s, e) in entries:
            print(f"  - {fn:30s}  {s} → {e}")
        # Chevauchements intra-code
        for i in range(1, len(entries)):
            prev = entries[i-1]
            cur  = entries[i]
            ov = overlap_range(prev[1], prev[2], cur[1], cur[2])
            if ov:
                print(f"    ! overlap : {prev[0]}  ↔  {cur[0]}   {ov[0]} → {ov[1]}")

if __name__ == "__main__":
    main()
