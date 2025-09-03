
from tensorflow.keras.layers import Dense, LSTM, Activation, BatchNormalization, Dropout,GaussianNoise
from tensorflow.keras import initializers
from relu_activation import ReLU
from relu_activation import BiReLU

from renormalization import BatchRenormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Constant
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K


def risk_estimation(y_true, y_pred):
    return -100. * K.mean(y_true * y_pred)


def pairwise_logit(y_true, y_pred):
    loss_mat = K.log(1 + K.exp(K.sign(K.transpose(y_true) - y_true) * (y_pred - K.transpose(y_pred))))
    return K.mean(K.mean(loss_mat))


class WindPuller(object):
    def __init__(self, input_shape = None, lr=0.01, n_layers=2, n_hidden=8, rate_dropout=0.2, loss=risk_estimation):
        self.input_shape = input_shape
        self.lr = lr
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.rate_dropout = rate_dropout
        self.loss = loss
        self.model = None

    def build_model(self):
        print("initializing..., learing rate %s, n_layers %s, n_hidden %s, dropout rate %s." % (
            self.lr, self.n_layers, self.n_hidden, self.rate_dropout))
        self.model = Sequential()
        self.model.add(GaussianNoise(stddev=0.01, input_shape=self.input_shape))
        # self.model.add(LocallyConnected1D(self.input_shape[1] * 2, 3))
        # self.model.add(Dropout(rate=0.5))
        for i in range(0, self.n_layers - 1):
            self.model.add(LSTM(self.n_hidden * 4, return_sequences=True, activation='tanh',
                                recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                                recurrent_initializer='orthogonal', bias_initializer='zeros',
                                dropout=self.rate_dropout, recurrent_dropout=self.rate_dropout))
             
        self.model.add(LSTM(self.n_hidden, return_sequences=False, activation='tanh',
                            recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                            recurrent_initializer='orthogonal', bias_initializer='zeros',
                            dropout=self.rate_dropout, recurrent_dropout=self.rate_dropout))
        
        self.model.add(Dense(1, kernel_initializer=initializers.glorot_uniform()))
        # self.model.add(BatchNormalization(axis=-1, moving_mean_initializer=Constant(value=0.5),
        #               moving_variance_initializer=Constant(value=0.25)))

        if self.loss == risk_estimation:
            self.model.add(BatchNormalization(axis=-1, beta_initializer='ones'))
            self.model.add(ReLU(alpha=0.0, max_value=1.0))
            
        opt = RMSprop(learning_rate=self.lr)
        self.model.compile(loss=self.loss,
                           optimizer=opt,
                           metrics=['accuracy'])

    def fit(self, x, y, batch_size=32, epochs=100, verbose=1, callbacks=None,
        validation_split=0., validation_data=None, shuffle=True,
        class_weight=None, sample_weight=None, initial_epoch=0, **kwargs):
        self.model.fit(
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


    def save(self, path):
        self.model.save(path + '.h5', save_format='h5')

    @staticmethod
    def load_model(path):
        wp = WindPuller()
        wp.model = load_model(path + '.h5', custom_objects={'risk_estimation': risk_estimation})

        return wp

    def evaluate(self, x, y, batch_size=32, verbose=1,
                 sample_weight=None, **kwargs):
        return self.model.evaluate(x, y, batch_size, verbose,
                                   sample_weight)

    def predict(self, x, batch_size=32, verbose=0):
        return self.model.predict(x, batch_size, verbose)