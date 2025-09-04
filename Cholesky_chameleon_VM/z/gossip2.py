# gossip.py — Cross-Asset Joint Forecasting (train/eval)
# ------------------------------------------------------
# - Lit 'ultimate_feature' (2 pickles train/test)
# - Convertit en DataSet-like (N,T,F) + labels (N,M)
# - Construit WindPuller avec n_outputs=M
# - Entraîne, checkpoint poids, sauvegarde modèle complet .h5
# - Évalue & exporte les prédictions multi-actifs

import os
import sys
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint

from windpuller import WindPuller, risk_estimation, directional_accuracy, pearson_corr


class DataSet(object):
    """Simple conteneur pour rester compatible avec ton code existant."""
    def __init__(self, images, labels):
        self._images = images.astype(np.float32)   # (N,T,F)
        self._labels = labels.astype(np.float32)   # (N,M)

    @property
    def images(self): return self._images
    @property
    def labels(self): return self._labels


def read_feature(path: str):
    """
    Lit le pickle 'ultimate_feature' (2 objets : train_map, test_map).
    Convertit 'feature' de [N,F,T] -> [N,T,F].
    """
    with open(path, "rb") as fp:
        try:
            train_map = pickle.load(fp)
            test_map  = pickle.load(fp)
        except EOFError:
            raise RuntimeError("Fichier 'ultimate_feature' invalide ou incomplet.")

    # X: [N,F,T] -> [N,T,F]
    trainX = np.transpose(np.asarray(train_map["feature"]), (0, 2, 1))
    testX  = np.transpose(np.asarray(test_map["feature"]),  (0, 2, 1))
    trainY = np.asarray(train_map["label"])
    testY  = np.asarray(test_map["label"])

    assets = train_map.get("assets", None)
    meta   = train_map.get("meta", {})

    print(f"[INFO] Train X={trainX.shape}, y={trainY.shape} | Test X={testX.shape}, y={testY.shape}")
    if assets:
        print(f"[INFO] Actifs ({len(assets)}): {assets}")
    return DataSet(trainX, trainY), DataSet(testX, testY), assets, meta


def cumulative_return_matrix(y_true, y_pred):
    """
    CR élémentaire par actif : cumprod(1 + y_true * y_pred) - 1, sur l'axe temps (N).
    - y_true : (N,M)
    - y_pred : (N,M)
    -> renvoie (N,M) cumulant dans le temps pour chaque actif
    """
    # Évite overflow en découpant si besoin ; ici on reste simple :
    step = 1.0 + y_true * y_pred
    cr = np.cumprod(step, axis=0) - 1.0
    return cr


def make_model(nb_epochs=50, batch_size=256, lr=8e-4, n_layers=2, n_hidden=64, rate_dropout=0.3, loss=risk_estimation):
    train_set, test_set, assets, meta = read_feature("./ultimate_feature")

    T, F = train_set.images.shape[1], train_set.images.shape[2]
    M = train_set.labels.shape[1]
    input_shape = (T, F)
    model_basename = f"model_{T}"
    full_model_path = model_basename + ".h5"
    ckpt_weights_path = model_basename + ".best.weights.h5"

    # Modèle
    wp = WindPuller(input_shape=input_shape, n_outputs=M, lr=lr, n_layers=n_layers,
                    n_hidden=n_hidden, rate_dropout=rate_dropout, loss=loss)
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
        batch_size=batch_size,
        epochs=nb_epochs,
        shuffle=True,
        verbose=1,
        validation_data=(test_set.images, test_set.labels),
        callbacks=callbacks
    )

    # Évalue (dernier)
    scores_last = wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print("Test (last) ->", dict(zip(wp.model.metrics_names, scores_last)))

    # Charge meilleurs poids puis sauvegarde modèle complet
    wp.model.load_weights(ckpt_weights_path)
    scores_best = wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print("Test (best ckpt) ->", dict(zip(wp.model.metrics_names, scores_best)))

    wp.model.save(full_model_path)
    print(f"[OK] Modèle sauvegardé : {full_model_path}")

    # Prédictions & exports
    pred = wp.predict(test_set.images, batch_size=1024)
    # pred shape: (N_te, M)
    cr = cumulative_return_matrix(test_set.labels, pred)  # (N_te, M)

    # Exports simples
    np.savetxt("pred_test.tsv", pred, fmt="%.6f", delimiter="\t")
    np.savetxt("y_true_test.tsv", test_set.labels, fmt="%.6f", delimiter="\t")
    np.savetxt("cumret_test.tsv", cr, fmt="%.6f", delimiter="\t")

    if assets:
        with open("assets.txt", "w") as f:
            for a in assets:
                f.write(a + "\n")

    print("[OK] Exports : pred_test.tsv, y_true_test.tsv, cumret_test.tsv, assets.txt")


def evaluate_only(model_path_no_ext="model_30"):
    train_set, test_set, assets, meta = read_feature("./ultimate_feature")
    custom = {'risk_estimation': risk_estimation,
              'directional_accuracy': directional_accuracy,
              'pearson_corr': pearson_corr}
    model = tf.keras.models.load_model(model_path_no_ext + ".h5", custom_objects=custom)

    scores = model.evaluate(test_set.images, test_set.labels, verbose=0)
    print("Test ->", dict(zip(model.metrics_names, scores)))

    pred = model.predict(test_set.images, batch_size=1024)
    cr = cumulative_return_matrix(test_set.labels, pred)
    np.savetxt("pred_test.tsv", pred, fmt="%.6f", delimiter="\t")
    np.savetxt("y_true_test.tsv", test_set.labels, fmt="%.6f", delimiter="\t")
    np.savetxt("cumret_test.tsv", cr, fmt="%.6f", delimiter="\t")
    if assets:
        with open("assets.txt", "w") as f:
            for a in assets:
                f.write(a + "\n")
    print("[OK] Évaluation & exports terminés.")


if __name__ == "__main__":
    op = "train"
    if len(sys.argv) > 1:
        op = sys.argv[1].strip().lower()

    if op == "train":
        make_model(
            nb_epochs=60,
            batch_size=512,
            lr=8e-4,
            n_layers=2,
            n_hidden=96,
            rate_dropout=0.4,
            loss=risk_estimation
        )
    elif op == "eval" or op == "evaluate":
        # Exemple : python gossip.py eval
        evaluate_only("model_30")
    else:
        print("Usage: gossip.py [train | eval]")
