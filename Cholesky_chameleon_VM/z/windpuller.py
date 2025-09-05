# windpuller.py — Cross-Asset Joint Forecasting (multi-output)
# ------------------------------------------------------------
# - Entrée : (T, F_total)
# - Sortie : vecteur R^{M} (un signal/retour par actif)
# - Perte : risk_estimation (maximiser y_true * y_pred)
# - Metrics : MAE/MSE + Directional Accuracy (sign match)
# - Activation finale : tanh (signal signé [-1, 1])
# - Sauvegarde/chargement robustes avec custom_objects

from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers, models, initializers, backend as K
from tensorflow.keras.optimizers import RMSprop


# ----------- Loss & metrics ------------

def risk_estimation(y_true, y_pred):
    """
    Perte 'profit' négative : -E[y_true * y_pred]
    Fonctionne pour y ∈ R^{N×M}. On prend la moyenne sur tous les éléments.
    """
    return -100.0 * K.mean(y_true * y_pred)


def directional_accuracy(y_true, y_pred):
    """
    Proportion d'accord de signe (sur toutes les sorties).
    """
    return K.mean(K.cast(K.equal(K.sign(y_true), K.sign(y_pred)), K.floatx()))


def pearson_corr(y_true, y_pred, eps=1e-8):
    """
    Corrélation de Pearson (moyenne sur les colonnes).
    """
    y_true_c = y_true - K.mean(y_true, axis=0, keepdims=True)
    y_pred_c = y_pred - K.mean(y_pred, axis=0, keepdims=True)
    num = K.sum(y_true_c * y_pred_c, axis=0)
    den = K.sqrt(K.sum(K.square(y_true_c), axis=0) * K.sum(K.square(y_pred_c), axis=0)) + eps
    r = num / den
    return K.mean(r)


# ------------- Modèle ------------------

class WindPuller(object):
    def __init__(self,
                 input_shape: Optional[tuple] = None,    # (T, F_total)
                 n_outputs: int = 1,                     # M (nombre d'actifs cibles)
                 lr: float = 1e-3,
                 n_layers: int = 2,
                 n_hidden: int = 64,
                 rate_dropout: float = 0.2,
                 loss=risk_estimation):
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.lr = lr
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.rate_dropout = rate_dropout
        self.loss = loss
        self.model: Optional[tf.keras.Model] = None

    def build_model(self):
        assert self.input_shape is not None, "input_shape doit être (T, F_total)"
        print(f"initializing... lr={self.lr}, n_layers={self.n_layers}, n_hidden={self.n_hidden}, dropout={self.rate_dropout}, outputs={self.n_outputs}")

        x_in = layers.Input(shape=self.input_shape)

        x = layers.GaussianNoise(stddev=0.01)(x_in)

        # Empilement LSTM
        for i in range(max(0, self.n_layers - 1)):
            x = layers.LSTM(
                units=self.n_hidden * 2,
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid',        # plus moderne que hard_sigmoid
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                dropout=self.rate_dropout,
                recurrent_dropout=0.0                   # laisser 0.0 pour permettre l'implémentation optimisée
            )(x)
            x = layers.Dropout(self.rate_dropout)(x)

        x = layers.LSTM(
            units=self.n_hidden,
            return_sequences=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            dropout=self.rate_dropout,
            recurrent_dropout=0.0
        )(x)

        x = layers.Dropout(self.rate_dropout)(x)

        # Tête multi-sorties
        out = layers.Dense(
            self.n_outputs,
            activation='tanh',                         # signal signé [-1,1]
            kernel_initializer=initializers.glorot_uniform()
        )(x)

        self.model = models.Model(inputs=x_in, outputs=out)

        opt = RMSprop(learning_rate=self.lr)
        self.model.compile(
            loss=self.loss,
            optimizer=opt,
            metrics=[directional_accuracy, 'mae', 'mse', pearson_corr]
        )

    def fit(self, x, y, batch_size=64, epochs=100, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0, **kwargs):
        return self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            **kwargs
        )

    def evaluate(self, x, y, batch_size=64, verbose=1, sample_weight=None, **kwargs):
        return self.model.evaluate(x, y, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight, **kwargs)

    def predict(self, x, batch_size=64, verbose=0):
        return self.model.predict(x, batch_size=batch_size, verbose=verbose)

    def save(self, path_no_ext: str):
        self.model.save(path_no_ext + ".h5")

    @staticmethod
    def load_model(path_no_ext: str):
        custom = {
            'risk_estimation': risk_estimation,
            'directional_accuracy': directional_accuracy,
            'pearson_corr': pearson_corr
        }
        obj = WindPuller()
        obj.model = tf.keras.models.load_model(path_no_ext + ".h5", custom_objects=custom)
        # Déduit n_outputs si besoin
        obj.n_outputs = obj.model.output_shape[-1]
        obj.input_shape = obj.model.input_shape[1:]
        return obj
