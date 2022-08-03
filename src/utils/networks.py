import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense,Concatenate, Flatten

def gaussian_likelihood(n_actions, lib='keras'): # for keras custom loss
    if lib == 'keras':
        _exp = K.exp
        _log = K.log
        _sum = K.sum
    else:
        _exp = np.exp
        _log = np.log
        _sum = np.sum

    def fn(actions,pred, log_std=False):
        if log_std == False:
            log_std = -0.5 * np.ones(n_actions, dtype=np.float32)

        pre_sum = -0.5 * (((actions-pred)/(_exp(log_std)+1e-8))**2 + 2*log_std + _log(2*np.pi))

        return _sum(pre_sum, axis=1)

    return fn



def discrete_actor_output(self,common_layer,n_actions):
    return Dense(n_actions, activation="softmax")(common_layer)

def continous_actor_output(self,common_layer,n_actions):
    sigma = Dense(n_actions, activation="softplus", name="sigma")(common_layer)
    mu = Dense(n_actions, activation="tanh" , name='mu')(common_layer)
    
    return Concatenate(axis=-1)([mu,sigma])

def MultiLayerPerceptron(policy="mlp"):
    layers = []
    if type(policy) == str:
        if policy == "mlp":
            layers.append(Dense(256, activation='relu', name="mlp_dense_layer_0"))
            layers.append(Dense(256, activation='relu', name="mlp_dense_layer_1"))
    else:
        for i,layer in enumerate(policy):
            layer._name = 'mlp_custom_layer_{}'.format(i)
            layers.append(layer)
            
    return layers

def CommonLayer(X_input, policy="mlp", rename=True):
    if rename:
        # Shared CNN layers:
        if type(policy) == str:
            if policy=="CNN":
                X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh", name="shared_cnn_conv_1d_layer_0")(X_input)
                X = MaxPooling1D(pool_size=2, name="shared_cnn_max_polling_1d_layer_0")(X)
                X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh", name="shared_cnn_conv_1d_layer_1")(X)
                X = MaxPooling1D(pool_size=2, name="shared_cnn_max_polling_1d_layer_1")(X)
                X = Flatten(name="shared_cnn_flatten_layer_0")(X)

            # Shared LSTM layers:
            elif policy=="LSTM":
                X = LSTM(512, return_sequences=True,name="shared_lstm_layer_0")(X_input)
                X = LSTM(256,name="shared_lstm_layer_1")(X)

            # Shared Dense layers:
            else:
                X = Flatten(name="shared_mlp_flatten_layer")(X_input)
                X = Dense(512, activation="relu", name="shared_mlp_dense_layer_0")(X)
        else:
            for i,layer in enumerate(policy):
                if i == 0:
                    X = layer(X_input)
                else:
                    X = layer(X)

                X._name = 'shared_custom_layer_{}'.format(i)
    else:
        # Shared CNN layers:
        if type(policy) == str:
            if policy=="CNN":
                X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
                X = MaxPooling1D(pool_size=2)(X)
                X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)
                X = MaxPooling1D(pool_size=2)(X)
                X = Flatten()(X)

            # Shared LSTM layers:
            elif policy=="LSTM":
                X = LSTM(512, return_sequences=True)(X_input)
                X = LSTM(256)(X)

            # Shared Dense layers:
            else:
                X = Flatten()(X_input)
                X = Dense(512, activation="relu")(X)
        else:
            for i,layer in enumerate(policy):
                if i == 0:
                    X = layer(X_input)
                else:
                    X = layer(X)


    return X