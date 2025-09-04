# predict_with_windpuller.py
# ---------------------------------------
# Prédit avec un modèle entraîné (model_30.h5)
# en réutilisant TA classe WindPuller.

import numpy as np

# === imports locaux : même pipeline que le train ===
from rawdata import read_sample_data          # lit le CSV -> liste de RawData triée
from chart import extract_feature             # construit les features identiques au train
from windpuller import WindPuller             # ta classe (avec load_model / predict)

# === même config que l'entraînement ===
SELECTOR = [
    "ROCP","OROCP","HROCP","LROCP","MACD","RSI",
    "VROCP","BOLL","MA","VMA","PRICE_VOLUME","CROSS_PRICE"
]
WINDOW = 30                     # Longueur de séquence utilisée au train
MODEL_STEM = "model_30"         # <- passe le nom SANS extension si ta load_model ajoute ".h5"
CSV_PATH = r"Z\BTC\kaggle 1\deeptrad_lstm\dataset\000001.csv"  # à adapter

def build_X_from_csv(csv_path, selector=SELECTOR, window=WINDOW):
    """
    Construit X au bon format (N, window, features) en réappliquant exactement
    le pipeline d'entraînement (extract_feature avec flatten=False puis transpose).
    Renvoie (X, y) où y sont les labels 'retour' si tu veux une comparaison rapide.
    """
    raw = read_sample_data(csv_path)  # lit et trie par date
    # flatten=False -> shape (N, F, T)
    moving_features, moving_labels = extract_feature(
        raw_data=raw,
        selector=selector,
        window=window,
        with_label=True,
        flatten=False
    )
    # (N, F, T) -> (N, T, F) pour LSTM Keras
    X = np.transpose(np.asarray(moving_features), (0, 2, 1))
    y = np.asarray(moving_labels)
    return X, y

if __name__ == "__main__":
    # 1) Charger le modèle via TA classe
    wp = WindPuller.load_model(MODEL_STEM)   # chargera 'model_30.h5'

    # 2) Construire X (même pipeline que le train)
    X, y = build_X_from_csv(CSV_PATH)

    # 3) Vérifs de forme par rapport au modèle
    _, T_expected, F_expected = wp.model.input_shape
    assert X.shape[1] == T_expected, f"Time steps attendus {T_expected}, reçus {X.shape[1]}"
    assert X.shape[2] == F_expected, f"Features attendues {F_expected}, reçues {X.shape[2]}"
    print("✅ X prêt pour la prédiction :", X.shape)

    # 4) Prédire avec TA méthode
    preds = wp.predict(X, batch_size=32, verbose=0).reshape(-1)

    # 5) Affichage rapide
    print("Extrait des prédictions :", preds[:10])
    if y is not None and len(y) == len(preds):
        print("Premières (pred, label):")
        for i in range(min(10, len(preds))):
            print(f"{i+1:02d}: pred={preds[i]: .6f}   label={y[i]: .6f}")

    # 6) Exemple : prédiction uniquement sur la dernière fenêtre
    last_pred = wp.predict(X[-1:, :, :], verbose=0).item()
    print("Prédiction dernière fenêtre :", last_pred)
