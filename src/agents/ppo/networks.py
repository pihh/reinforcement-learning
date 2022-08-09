import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from src.utils.networks import CommonLayer
from src.utils.policy import gaussian_likelihood

### CONTINUOUS

class PpoActorContinuous:
    def __init__(self, 
        	observation_shape, 
            action_space, 
            learning_rate,
            optimizer,
            loss_clipping = 0.2,
            policy="mlp"
        ):

        self.action_space = action_space
        self.loss_clipping = loss_clipping
        self.log_std = -0.5 * np.ones(self.action_space , dtype=np.float32)

        self.gaussian_likelihood = gaussian_likelihood(self.log_std, lib="keras")

        X_input = Input(observation_shape)
        X = CommonLayer(X_input,policy)
        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)

        output = Dense(self.action_space, activation="tanh")(X)

        self.model = Model(inputs = X_input, outputs = output)
        self.model.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=learning_rate))

    def ppo_loss(self, y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]

        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + self.loss_clipping)*advantages, (1.0 - self.loss_clipping)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss


    def predict(self, state):
        return self.model.predict(state)

### DISCRETE
class PpoActorDiscrete:
    def __init__(self, 
        observation_shape, 
        action_space, 
        learning_rate, 
        optimizer,
        loss_clipping=0.2,
        loss_entropy=0.001,
        policy="mlp"):

        self.action_space = action_space
        self.loss_clipping = loss_clipping
        self.loss_entropy = loss_entropy

        X_input = Input(observation_shape)
        # X = CommonLayer(X_input,self.policy)
        X = CommonLayer(X_input,policy)
        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_space, activation="softmax")(X)

        self.model = Model(inputs = X_input, outputs = output)
        self.model.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=learning_rate))

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        #advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        advantages,  actions, prediction_picks = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - self.loss_clipping, max_value=1 + self.loss_clipping) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = self.loss_entropy * K.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, state):
        return self.model.predict(state)

# PPO Critic for discrete or continuous only differs in the initializer
class PpoCritic:
    def __init__(self, 
        observation_shape, 
        learning_rate, 
        optimizer,
        loss_function_version=1, 
        loss_clipping=0.2,
        kernel_initializer=False,
        action_space_mode="discrete", 
        policy ="mlp"):

        self.loss_clipping = loss_clipping

        X_input = Input(shape=observation_shape) 
        old_values = Input(shape=(1,))

        if kernel_initializer == False:
            if action_space_mode == "discrete":
                kernel_initializer = 'he_uniform'
            else:
                kernel_initializer=tf.random_normal_initializer(stddev=0.01)

        if loss_function_version == 1:
            loss_function = self.ppo_loss
        else:
            loss_function = self.ppo_loss_2(old_values)

        V = CommonLayer(X_input,policy)
        V = Dense(512, activation="relu", kernel_initializer=kernel_initializer)(V)
        V = Dense(256, activation="relu", kernel_initializer=kernel_initializer)(V)
        V = Dense(64, activation="relu", kernel_initializer=kernel_initializer)(V)
        value = Dense(1, activation=None)(V)

        self.model = Model(inputs=[X_input, old_values], outputs = value)
        self.model.compile(loss=[loss_function], optimizer=optimizer(learning_rate=learning_rate))

    def ppo_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def ppo_loss_2(self, values):
        def loss(y_true, y_pred):

            clipped_value_loss = values + K.clip(y_pred - values, -self.loss_clipping, self.loss_clipping)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2

            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
            return value_loss
        return loss

    def predict(self, state):
        return self.model.predict([state, np.zeros((state.shape[0], 1))])


