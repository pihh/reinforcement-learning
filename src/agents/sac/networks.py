import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from src.utils.networks import MultiLayerPerceptron
from src.agents.sac.buffer import ReplayBuffer


print('@TODO implement Common Layer instead of MLP')
class CriticNetwork(Model):
    def __init__(self,
                policy="mlp",
                name='critic'
        ):
        super(CriticNetwork, self).__init__()
        
        
        self.model_name = name
        self.fc = MultiLayerPerceptron(policy=policy)
        self.q = Dense(1, activation=None)
        self._name = name


    def call(self, state, action):
        X = tf.concat([state, action], axis=1)
        for layer in self.fc:
            X = layer(X)
            
        q = self.q(X)
        return q

class ValueNetwork(Model):
    def __init__(self,
                 policy="mlp",
                 name='value',  
        ):
        super(ValueNetwork, self).__init__()
        

        self.model_name = name

        self.fc = MultiLayerPerceptron(policy=policy)
        self.v = Dense(1, activation=None)
        self._name = name

    def call(self, state):
        X = state
        for layer in self.fc:
            X = layer(X)

        v = self.v(X)

        return v

class ActorNetwork(Model):
    def __init__(self, 
            policy="mlp",
            n_actions=2,
            max_action=1, 
            name='actor', 
    ):
        super(ActorNetwork, self).__init__()

        self.model_name = name
        self.max_action = max_action
        self.noise = 1e-6

        self.fc = MultiLayerPerceptron(policy=policy)
        
        self.mu = Dense(n_actions, activation=None)
        self.sigma = Dense(n_actions, activation=None)
        self._name = name

    def call(self, state):
        X = state
        for layer in self.fc:
            X = layer(X)

        mu = self.mu(X)
        sigma = self.sigma(X)
        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.sample() # + something else if you want to implement
        else:
            actions = probabilities.sample()

        action = tf.math.tanh(actions)*self.max_action
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action,2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs
