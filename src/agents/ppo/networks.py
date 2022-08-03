import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from src.utils.networks import gaussian_likelihood
from src.utils.networks import CommonLayer


def critic_ppo2_loss(values, loss_clipping):
    def loss(y_true, y_pred):
        clipped_value_loss = values + K.clip(y_pred - values, loss_clipping, loss_clipping)
        v_loss1 = (y_true - clipped_value_loss) ** 2
        v_loss2 = (y_true - y_pred) ** 2
            
        value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
        #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss
    return loss

def critic_ppo_loss(y_true, y_pred):
    value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
    return value_loss

def actor_ppo_loss_discrete(n_actions, loss_clipping, loss_entropy):

    def fn(y_true,y_pred):

        print(y_true, y_pred)
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+n_actions], y_true[:, 1+n_actions:]

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
            
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - loss_clipping, max_value=1 + loss_clipping) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = loss_entropy * K.mean(entropy)
            
        total_loss = actor_loss - entropy

        return total_loss
    return fn

def actor_ppo_loss_continuous(n_actions,loss_clipping):

    _gaussian_likelihood = gaussian_likelihood(n_actions,lib="keras")

    def fn(y_true,y_pred):
        advantages, logp_old_ph, actions = y_true[:, :1], y_true[:, 1:1+n_actions], y_true[:, 1+n_actions]
        # self.loss_clipping = clipping_val
        logp = _gaussian_likelihood(actions, y_pred,log_std=False)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + loss_clipping)*advantages, (1.0 - loss_clipping)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss
    return fn




def act(actor,n_actions, std=False, log_std=False, continuous=False):
 
    _gaussian_likelihood = gaussian_likelihood(n_actions,lib="numpy")
    
    def fn(state):
        if not continuous:
            prediction = actor.predict(state)[0]
            action = np.random.choice(n_actions, p=prediction)
            action_onehot = np.zeros([n_actions])
            action_onehot[action] = 1
            return action, action_onehot, prediction
        else:
            
            # Use the network to predict the next action to take, using the model
            pred = actor.predict(state)

            low, high = -1.0, 1.0 # -1 and 1 are boundaries of tanh

            action = pred + np.random.uniform(low, high, size=pred.shape) * std
            action = np.clip(action, low, high)
            
            logp_t = _gaussian_likelihood(action, pred, log_std)
            # print('act',std, log_std, continuous)
            # print('logp_t',logp_t,'action',action, 'pred',pred)
            # print()
            return action[0], action[0], logp_t

    return fn

def get_actor_model(
    input_shape, 
    n_actions, 
    loss_clipping, 
    loss_entropy, 
    optimizer=Adam, 
    learning_rate=0.00025,
    kernel_initializer=False,
    policy="mlp",
    continuous=False):

    if kernel_initializer == False:
        if continuous:
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)
        else:
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)

    X_input = Input(input_shape)
    X = CommonLayer(X_input,policy=policy)

    X = Dense(512, activation="relu", kernel_initializer=kernel_initializer)(X_input)
    X = Dense(256, activation="relu", kernel_initializer=kernel_initializer)(X)
    X = Dense(64, activation="relu", kernel_initializer=kernel_initializer)(X)

    if continuous:
        output = Dense(n_actions,activation="tanh")(X)
        loss_function = actor_ppo_loss_continuous(n_actions, loss_clipping)
    else:
        output = Dense(n_actions, activation="softmax")(X)
        loss_function = actor_ppo_loss_discrete(n_actions, loss_clipping, loss_entropy)

    model = Model(inputs = X_input, outputs = output)
    model.compile(loss=loss_function, optimizer=optimizer(learning_rate=learning_rate))

    return model


def get_critic_model(
    input_shape,
    loss_clipping, 
    optimizer=Adam, 
    learning_rate=0.00025,
    kernel_initializer=False,
    policy="mlp", 
    continuous=False):
    
    if kernel_initializer == False:
        if continuous:
            kernel_initializer = 'he_uniform'
        else:
            kernel_initializer = tf.random_normal_initializer(stddev=0.01)

    X_input = Input(input_shape)
    Old_values = Input(shape=(1,))
    V = CommonLayer(X_input,policy=policy)
    
    V = Dense(512, activation="relu", kernel_initializer=kernel_initializer)(X_input)
    V = Dense(256, activation="relu", kernel_initializer=kernel_initializer)(V)
    V = Dense(64, activation="relu", kernel_initializer=kernel_initializer)(V)
    value = Dense(1, activation=None)(V)

    model = Model(inputs=[X_input, Old_values], outputs = value)
    model.compile(loss=critic_ppo2_loss(Old_values,loss_clipping), optimizer=optimizer(learning_rate=learning_rate))

    return model