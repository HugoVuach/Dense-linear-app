# gossip2.py — Cross-Asset Joint Forecasting (train/eval/predict)
# ---------------------------------------------------------------
# - Lit 'ultimate_feature' (pickle: train_map + test_map)
# - Convertit X: [N,F,T] -> [N,T,F], y: [N,M]
# - Construit WindPuller(n_outputs=M)
# - Entraîne + checkpoint (poids) + sauvegarde modèle complet .h5
# - Évalue + exporte prédictions multi-actifs

import os
import sys
import argparse
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from windpuller import WindPuller, risk_estimation, directional_accuracy, pearson_corr


class DataSet(object):
    def __init__(self, images, labels):
        self._images = images.astype(np.float32)   # (N,T,F)
        self._labels = labels.astype(np.float32)   # (N,M)

    @property
    def images(self): return self._images
    @property
    def labels(self): return self._labels


def read_feature(path: str):
    """Lit le pickle 'ultimate_feature' (2 objets: train_map, test_map).
       Convertit 'feature' de [N,F,T] -> [N,T,F]."""
    with open(path, "rb") as fp:
        try:
            train_map = pickle.load(fp)
            test_map  = pickle.load(fp)
        except EOFError:
            raise RuntimeError(f"Fichier '{path}' invalide ou incomplet (pickle).")

    trainX = np.transpose(np.asarray(train_map["feature"]), (0, 2, 1))
    testX  = np.transpose(np.asarray(test_map["feature"]),  (0, 2, 1))
    trainY = np.asarray(train_map["label"])
    testY  = np.asarray(test_map["label"])

    assets = train_map.get("assets", None)
    meta   = train_map.get("meta", {})

    print(f"[INFO] Train X={trainX.shape}, y={trainY.shape} | Test X={testX.shape}, y={testY.shape}")
    if assets:
        print(f"[INFO] Actifs ({len(assets)}): {assets}")

    # ====== DEBUG start ======
    print("[DBG] any NaN/Inf in Train X:",
          np.isnan(trainX).any(), np.isinf(trainX).any())
    print("[DBG] any NaN/Inf in Train Y:",
          np.isnan(trainY).any(), np.isinf(trainY).any())
    print("[DBG] any NaN/Inf in Test  X:",
          np.isnan(testX).any(), np.isinf(testX).any())
    print("[DBG] any NaN/Inf in Test  Y:",
          np.isnan(testY).any(), np.isinf(testY).any())

    # Bornes (nan-safe)
    try:
        print("[DBG] Train X min/max:", np.nanmin(trainX), np.nanmax(trainX))
        print("[DBG] Train Y min/max:", np.nanmin(trainY), np.nanmax(trainY))
        print("[DBG] Test  X min/max:", np.nanmin(testX),  np.nanmax(testX))
        print("[DBG] Test  Y min/max:", np.nanmin(testY),  np.nanmax(testY))
    except ValueError:
        # (peut arriver si arrays vides)
        print("[DBG] min/max non disponibles (arrays vides)")
    # ====== DEBUG end ======

    
    return DataSet(trainX, trainY), DataSet(testX, testY), assets, meta


def cumulative_return_matrix(y_true, y_pred):
    """CR par actif : cumprod(1 + y_true * y_pred) - 1, sur l'axe N (temps)."""
    step = 1.0 + y_true * y_pred
    return np.cumprod(step, axis=0) - 1.0


def train_cmd(args):
    features_path = args.features
    train_set, test_set, assets, meta = read_feature(features_path)

    T, F = train_set.images.shape[1], train_set.images.shape[2]
    M = train_set.labels.shape[1]
    input_shape = (T, F)

    model_basename = f"model_panel_{T}"
    full_model_path = model_basename + ".h5"
    ckpt_weights_path = model_basename + ".best.weights.h5"

    # Modèle
    wp = WindPuller(
        input_shape=input_shape,
        n_outputs=M,
        lr=args.lr,
        n_layers=args.layers,
        n_hidden=args.hidden,
        rate_dropout=args.dropout,
        loss=risk_estimation  # garder la perte signée par défaut
    )
    wp.build_model()

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=ckpt_weights_path,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        )
    ]

    # Entraînement
    wp.fit(
        train_set.images, train_set.labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        verbose=1,
        validation_data=(test_set.images, test_set.labels),
        callbacks=callbacks
    )

    # Évalues (dernier)
    scores_last = wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print("Test (last) ->", dict(zip(wp.model.metrics_names, scores_last)))

    # Charge meilleurs poids et sauvegarde modèle complet
    wp.model.load_weights(ckpt_weights_path)
    scores_best = wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print("Test (best ckpt) ->", dict(zip(wp.model.metrics_names, scores_best)))

    wp.model.save(full_model_path)
    print(f"[OK] Modèle sauvegardé : {full_model_path}")

    # Prédictions & exports
    pred = wp.predict(test_set.images, batch_size=1024)   # (N_te, M)
    cr   = cumulative_return_matrix(test_set.labels, pred)

    np.savetxt("pred_test.tsv", pred, fmt="%.6f", delimiter="\t")
    np.savetxt("y_true_test.tsv", test_set.labels, fmt="%.6f", delimiter="\t")
    np.savetxt("cumret_test.tsv", cr, fmt="%.6f", delimiter="\t")

    if assets:
        with open("assets.txt", "w") as f:
            for a in assets: f.write(a + "\n")

    print("[OK] Exports : pred_test.tsv, y_true_test.tsv, cumret_test.tsv, assets.txt")


def eval_cmd(args):
    features_path = args.features
    model_path_no_ext = args.model.replace(".h5", "")

    train_set, test_set, assets, meta = read_feature(features_path)
    custom = {
        'risk_estimation': risk_estimation,
        'directional_accuracy': directional_accuracy,
        'pearson_corr': pearson_corr
    }
    model = tf.keras.models.load_model(model_path_no_ext + ".h5", custom_objects=custom)
    scores = model.evaluate(test_set.images, test_set.labels, verbose=0)
    print("Test ->", dict(zip(model.metrics_names, scores)))

    pred = model.predict(test_set.images, batch_size=1024)
    cr   = cumulative_return_matrix(test_set.labels, pred)

    np.savetxt("pred_test.tsv", pred, fmt="%.6f", delimiter="\t")
    np.savetxt("y_true_test.tsv", test_set.labels, fmt="%.6f", delimiter="\t")
    np.savetxt("cumret_test.tsv", cr, fmt="%.6f", delimiter="\t")
    if assets:
        with open("assets.txt", "w") as f:
            for a in assets: f.write(a + "\n")
    print("[OK] Évaluation & exports terminés.")


def predict_cmd(args):
    """Identique à eval mais sans impression de métriques, si tu veux séparer."""
    return eval_cmd(args)


def main():
    p = argparse.ArgumentParser(description="Cross-Asset Joint Forecasting — entraînement/évaluation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="Entrainer le modèle multi-actifs")
    pt.add_argument("--features", type=str, required=True, help="Chemin vers le pickle ultimate_feature")
    pt.add_argument("--epochs", type=int, default=100)
    pt.add_argument("--batch_size", type=int, default=512)
    pt.add_argument("--lr", type=float, default=8e-4)
    pt.add_argument("--layers", type=int, default=2)
    pt.add_argument("--hidden", type=int, default=96)
    pt.add_argument("--dropout", type=float, default=0.3)

    # eval / predict
    for name in ("eval", "predict"):
        pe = sub.add_parser(name, help="Évaluer un modèle sauvegardé sur le test set")
        pe.add_argument("--features", type=str, required=True)
        pe.add_argument("--model", type=str, required=True, help="Fichier .h5 (ou basename)")

    args = p.parse_args()

    if args.cmd == "train":
        train_cmd(args)
    elif args.cmd in ("eval", "predict"):
        eval_cmd(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

    """ pour laancer
    python gossip2.py train \
  --features ./ultimate_feature \
  --epochs 120 \
  --batch_size 512 \
  --lr 5e-4 \
  --layers 2 \
  --hidden 64 \
  --dropout 0.3

    """
