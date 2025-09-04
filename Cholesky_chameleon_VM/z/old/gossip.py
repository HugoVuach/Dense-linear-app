# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model

from windpuller import WindPuller, risk_estimation, pairwise_logit
from dataset import DataSet
from feature import extract_from_file
import pickle


def read_ultimate(path, input_shape):
    """(Optionnel) lecteur de features à plat, non utilisé dans le flux pickle actuel."""
    ultimate_features = np.loadtxt(path + "ultimate_feature." + str(input_shape[0]))
    ultimate_features = np.reshape(ultimate_features, [-1, input_shape[0], input_shape[1]])
    ultimate_labels = np.loadtxt(path + "ultimate_label." + str(input_shape[0]))
    train_set = DataSet(ultimate_features, ultimate_labels)

    test_features = np.loadtxt(path + "ultimate_feature.test." + str(input_shape[0]))
    test_features = np.reshape(test_features, [-1, input_shape[0], input_shape[1]])
    test_labels = np.loadtxt(path + "ultimate_label.test." + str(input_shape[0]))
    test_set = DataSet(test_features, test_labels)
    return train_set, test_set


def read_feature(path):
    """
    Lit le fichier binaire 'ultimate_feature' (pickle) produit par feature.py,
    concatène train/test de tous les instruments, et transpose en [batch, time, feat].
    """
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    with open(path, "rb") as fp:
        while True:
            try:
                train_map = pickle.load(fp)
                test_map = pickle.load(fp)
                train_features.extend(train_map["feature"])
                train_labels.extend(train_map["label"])
                test_features.extend(test_map["feature"])
                test_labels.extend(test_map["label"])
                print(f"read {train_map['code']} successfully. ")
            except Exception:
                break

    # On passe de [batch, feat, time] à [batch, time, feat]
    trainX = np.transpose(np.asarray(train_features), [0, 2, 1])
    testX  = np.transpose(np.asarray(test_features),  [0, 2, 1])
    trainY = np.asarray(train_labels)
    testY  = np.asarray(test_labels)
    return DataSet(trainX, trainY), DataSet(testX, testY)


def read_separate_feature(path):
    """
    Version 'séparée' : un DataSet par ticker (train/test).
    """
    train_sets, test_sets = {}, {}
    with open(path, "rb") as fp:
        while True:
            try:
                train_map = pickle.load(fp)
                test_map = pickle.load(fp)
                train_sets[train_map["code"]] = DataSet(
                    np.transpose(np.asarray(train_map["feature"]), [0, 2, 1]),
                    np.asarray(train_map["label"])
                )
                test_sets[test_map["code"]] = DataSet(
                    np.transpose(np.asarray(test_map["feature"]), [0, 2, 1]),
                    np.asarray(test_map["label"])
                )
                print(f"read {train_map['code']} successfully. ")
            except Exception:
                break
    return train_sets, test_sets


def calculate_cumulative_return(labels, pred):
    """Accumulation simple : produit des (1 + y_t * p_t) - 1."""
    cr = []
    if len(labels) <= 0:
        return cr
    cr.append(1.0 * (1.0 + labels[0] * pred[0]))
    for l in range(1, len(labels)):
        cr.append(cr[l - 1] * (1 + labels[l] * pred[l]))
    for i in range(len(cr)):
        cr[i] = cr[i] - 1
    return cr


def turnover(pred):
    """Somme des variations absolues de la position (proxy de turnover)."""
    t = 0.0
    for l in range(1, len(pred)):
        t += abs(pred[l] - pred[l - 1])
    return t


def evaluate_model(model_path_no_ext, code, input_shape=[30, 61]):
    """
    Charge un modèle complet .h5 et évalue sur un code spécifique (après extraction ad hoc).
    """
    # Génère features pour ce code uniquement
    extract_from_file(os.path.join("dataset", f"{code}.csv"), code)
    train_set, test_set = read_feature(f"./{code}_feature")

    # Charge le modèle complet (architecture + poids + compile)
    model_file = model_path_no_ext + ".h5"
    model = load_model(model_file, custom_objects={'risk_estimation': risk_estimation})

    scores = model.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('test accuracy:', scores[1])

    pred = model.predict(test_set.images, batch_size=1024)
    pred = np.reshape(pred, [-1])

    cr = calculate_cumulative_return(test_set.labels, pred)
    print("changeRate\tpositionAdvice\tprincipal\tcumulativeReturn")
    for i in range(len(test_set.labels)):
        print(f"{test_set.labels[i]}\t{pred[i]}\t{cr[i] + 1.}\t{cr[i]}")
    print("turnover:", turnover(pred))


def make_model(nb_epochs=100, batch_size=128, lr=0.01, n_layers=1, n_hidden=14,
               rate_dropout=0.3, loss=risk_estimation):
    """
    Entraîne sur le fichier 'ultimate_feature' (tous tickers concaténés),
    sauvegarde checkpoints (poids) + modèle complet .h5, et (optionnel) recharge pour vérifier.
    """
    # 1) Charger les features/labels
    train_set, test_set = read_feature("./ultimate_feature")
    input_shape = [train_set.images.shape[1], train_set.images.shape[2]]
    model_basename = f"model_{input_shape[0]}"
    full_model_path = model_basename + ".h5"
    ckpt_weights_path = model_basename + ".best.weights.h5"

    # 2) Construire le modèle
    wp = WindPuller(input_shape=input_shape, lr=lr, n_layers=n_layers,
                    n_hidden=n_hidden, rate_dropout=rate_dropout, loss=loss)
    wp.build_model()

    # 3) Callbacks : checkpoint = POIDS UNIQUEMENT (fiable avec Keras 3)
    callbacks = [
        # Désactive TensorBoard si tu n’en as pas besoin
        # TensorBoard(log_dir="logs", histogram_freq=0),
        ModelCheckpoint(
            filepath=ckpt_weights_path,
            save_best_only=True,
            save_weights_only=True,   # <--- évite les problèmes de save_model(options)
            monitor="val_loss",
            mode="min",
            verbose=1
        )
    ]

    # 4) Entraînement
    wp.fit(
        train_set.images, train_set.labels,
        batch_size=batch_size,
        epochs=nb_epochs,
        shuffle=True,
        verbose=1,
        validation_data=(test_set.images, test_set.labels),
        callbacks=callbacks
    )

    # 5) Évaluation avec les derniers poids en mémoire
    scores = wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss (last):', scores[0])
    print('Test accuracy (last):', scores[1])

    # 6) Sauvegarder le modèle COMPLET (.h5)
    #    -> Architecture + Poids + Compile (robuste pour re-load ultérieur)
    wp.model.save(full_model_path)   # pas d'options exotiques

    # 7) (Optionnel) Recharger le meilleur checkpoint (poids) AVANT de sérialiser en .h5
    #    Si tu veux que 'model_XX.h5' contienne VRAIMENT les meilleurs poids :
    #    a) recharge les meilleurs poids
    wp.model.load_weights(ckpt_weights_path)
    #    b) re-évalue
    best_scores = wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss (best ckpt):', best_scores[0])
    print('Test accuracy (best ckpt):', best_scores[1])
    #    c) resauvegarde le modèle complet avec les meilleurs poids
    wp.model.save(full_model_path)
    print(f"✅ Modèle complet sauvegardé : {full_model_path}")
    print(f"✅ Poids checkpoint sauvegardés : {ckpt_weights_path}")

    # 8) (Optionnel) Test de rechargement complet (pour s’assurer que tout marche)
    reloaded = load_model(full_model_path, custom_objects={'risk_estimation': risk_estimation})
    reload_scores = reloaded.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Reloaded Test loss:', reload_scores[0])
    print('Reloaded Test accuracy:', reload_scores[1])

    # 9) (Optionnel) Export des prédictions
    pred = reloaded.predict(test_set.images, batch_size=1024)
    pred = np.reshape(pred, [-1])
    result = np.array([pred, test_set.labels]).T
    with open('output.' + str(input_shape[0]), 'w') as fp:
        for row in result:
            fp.write(f"{row[0]}\t{row[1]}\n")


def make_separate_model(nb_epochs=100, batch_size=128, lr=0.01, n_layers=1,
                        n_hidden=14, rate_dropout=0.3, input_shape=[30, 73]):
    """
    Variante : un modèle par ticker (charge + entraîne + sauvegarde).
    """
    train_sets, test_sets = read_separate_feature("./ultimate_feature")

    for code, train_set in train_sets.items():
        test_set = test_sets[code]
        input_shape = [train_set.images.shape[1], train_set.images.shape[2]]
        model_basename = f"model.{code}"
        full_model_path = model_basename + ".h5"
        ckpt_weights_path = model_basename + ".best.weights.h5"

        print("input_shape:", input_shape)
        print("train images shape:", train_set.images.shape)

        wp = WindPuller(input_shape=input_shape, lr=lr, n_layers=n_layers,
                        n_hidden=n_hidden, rate_dropout=rate_dropout)
        wp.build_model()

        callbacks = [
            # TensorBoard(log_dir=f"logs/{code}", histogram_freq=0),
            ModelCheckpoint(
                filepath=ckpt_weights_path,
                save_best_only=True,
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
                verbose=1
            )
        ]

        wp.fit(
            train_set.images, train_set.labels,
            batch_size=batch_size,
            epochs=nb_epochs,
            shuffle=False,
            verbose=1,
            validation_data=(test_set.images, test_set.labels),
            callbacks=callbacks
        )

        # Sauvegarde finale avec les meilleurs poids
        wp.model.load_weights(ckpt_weights_path)
        wp.model.save(full_model_path)

        scores = wp.evaluate(test_set.images, test_set.labels, verbose=0)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        # Vérif rechargement
        reloaded = load_model(full_model_path, custom_objects={'risk_estimation': risk_estimation})
        scores2 = reloaded.evaluate(test_set.images, test_set.labels, verbose=0)
        print('Reloaded Test loss:', scores2[0])
        print('Reloaded Test accuracy:', scores2[1])

        pred = reloaded.predict(test_set.images, batch_size=1024)
        pred = np.reshape(pred, [-1])
        result = np.array([pred, test_set.labels]).transpose()
        with open('output.' + str(input_shape[0]), 'w') as fp:
            for i in range(result.shape[0]):
                fp.write(f"{result[i][0]}\t{result[i][1]}\n")


if __name__ == '__main__':
    operation = "train"
    if len(sys.argv) > 1:
        operation = sys.argv[1]

    if operation == "train":
        # Entraînement global (tous tickers concaténés)
        make_model(
            nb_epochs=100,
            batch_size=512,
            lr=0.0004,
            n_layers=1,
            n_hidden=14,
            rate_dropout=0.5,
            loss=risk_estimation  # ou pairwise_logit
        )
    elif operation == "predict":
        # Exemple : évaluer un modèle entraîné sur un seul code
        evaluate_model("model_30", "000001")
    else:
        print("Usage: gossip.py [train | predict]")
