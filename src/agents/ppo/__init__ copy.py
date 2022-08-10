import shutup
shutup.please()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1:cpu, 0:first gpu
import random
import gym
import pybullet_envs
import pylab
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras import backend as K
import time
import copy


from threading import Thread, Lock
from multiprocessing import Process, Pipe, cpu_count


tf.compat.v1.disable_eager_execution() # usually using this for fastest performance

# Helpers
def gaussian_likelihood(log_std, lib="keras"): # for keras custom loss
    _exp = K.exp
    _log = K.log
    _sum = K.sum
    if lib == "numpy":
        _exp = np.exp
        _log = np.log
        _sum = np.sum

    def fn(actions,pred):
        pre_sum = -0.5 * (((actions-pred)/(_exp(log_std)+1e-8))**2 + 2*log_std + _log(2*np.pi))
        return _sum(pre_sum, axis=1)

    return fn

# Environment
class Environment(Process):
    def __init__(self, env_idx, agent_connection, env_name, state_size, action_size):
        super(Environment, self).__init__()
        self.env = gym.make(env_name)
        self.env_idx = env_idx
        self.agent_connection = agent_connection
        self.state_size = state_size
        self.action_size = action_size

    def run(self):
        super(Environment, self).run()
        state = self.env.reset()

        state = np.reshape(state, [1, self.state_size])

        self.agent_connection.send(state)
        while True:

            action = self.agent_connection.recv()

            state, reward, done, info = self.env.step(action)
            state = np.reshape(state, [1, self.state_size])

            if done:
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

            self.agent_connection.send([state, reward, done, info])

# Continuous
class PpoActorContinuous:
    def __init__(self, input_shape, action_space, lr, optimizer,loss_clipping = 0.2):

        self.action_space = action_space
        self.loss_clipping = loss_clipping
        self.log_std = -0.5 * np.ones(self.action_space , dtype=np.float32)

        self.gaussian_likelihood = gaussian_likelihood(self.log_std, lib="keras")

        X_input = Input(input_shape)


        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_space, activation="tanh")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=lr))

    def ppo_loss(self, y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]

        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + self.loss_clipping)*advantages, (1.0 - self.loss_clipping)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss


    def predict(self, state):
        return self.Actor.predict(state)

# Discrete
class PpoActorDiscrete:
    def __init__(self, input_shape, action_space, lr, optimizer,loss_clipping=0.2,loss_entropy=0.001):

        self.action_space = action_space
        self.loss_clipping = loss_clipping
        self.loss_entropy = loss_entropy

        X_input = Input(input_shape)


        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_space, activation="softmax")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=lr))

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
        return self.Actor.predict(state)


# PPO Critic for discrete or continuous only differs in the initializer
class PpoCritic:
    def __init__(self, input_shape, action_space, lr, optimizer,loss_function_version=1, loss_clipping=0.2,kernel_initializer=False,continuous_action_space=False):

        self.loss_clipping = loss_clipping

        X_input = Input(input_shape)
        old_values = Input(shape=(1,))

        if kernel_initializer == False:
            if continuous_action_space == False:
                kernel_initializer = 'he_uniform'
            else:
                kernel_initializer=tf.random_normal_initializer(stddev=0.01)

        if loss_function_version == 1:
            loss_function = self.ppo_loss
        else:
            loss_function = self.ppo_loss_2(old_values)

        V = Dense(512, activation="relu", kernel_initializer=kernel_initializer)(X_input)
        V = Dense(256, activation="relu", kernel_initializer=kernel_initializer)(V)
        V = Dense(64, activation="relu", kernel_initializer=kernel_initializer)(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=[X_input, old_values], outputs = value)
        self.Critic.compile(loss=[loss_function], optimizer=optimizer(learning_rate=lr))

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
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])

class PpoBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states=[]
        self.next_states=[]
        self.actions=[]
        self.rewards=[]
        self.predictions=[]
        self.dones=[]

# PPO PPOAgent
class PpoAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, env_name, training_batch=4000, epochs=80, episodes=1000,lr=0.00025,shuffle=False,target_kl = 0.01, continuous_action_space=False, n_workers=cpu_count()):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name
        self.env = gym.make(env_name)

        self.target_kl = target_kl
        self.n_workers = n_workers


        if continuous_action_space:
            self.action_size = self.env.action_space.shape[0]
        else:
            self.action_size = self.env.action_space.n

        self.state_size = self.env.observation_space.shape
        self.EPISODES = episodes # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = lr
        self.epochs = epochs # training epochs
        self.shuffle = shuffle
        self.training_batch = training_batch
        #self.optimizer = RMSprop
        self.optimizer = Adam
        self.replay_count = 0
        self.continuous_action_space=continuous_action_space

        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        if continuous_action_space:
            self.Actor= PpoActorContinuous(self.state_size, self.action_size, lr=self.lr, optimizer = self.optimizer,loss_clipping = 0.2)
        else:
            self.Actor= PpoActorDiscrete(self.state_size, self.action_size, lr=self.lr, optimizer = self.optimizer,loss_clipping=0.2,loss_entropy=0.001)

        self.Critic = PpoCritic(self.state_size, self.action_size, lr=self.lr, optimizer = self.optimizer,loss_clipping=0.2,kernel_initializer=False,continuous_action_space=continuous_action_space)

        # do not change bellow
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)

        # Bind gaussian likelihood
        self.gaussian_likelihood = gaussian_likelihood(self.log_std, lib="numpy")

    def act_batch(self,states):
        if self.continuous_action_space:
            # Use the networker to predict the next action to take, using the model
            prediction = self.Actor.predict(states)

            low, high = -1.0, 1.0 # -1 and 1 are boundaries of tanh
            action = prediction + np.random.uniform(low, high, size=prediction.shape) * self.std
            action = np.clip(action, low, high)

            logp_t = self.gaussian_likelihood(action, prediction)

            return action.astype(np.float32), action.astype(np.float32) , logp_t.astype(np.float32)
        else:
            predictions_list = self.Actor.predict(states)
            actions_list = [np.random.choice(self.action_size, p=i) for i in predictions_list]
            actions_onehot_list = [np.zeros([self.action_size]) for i in predictions_list]

            for i,action_list in enumerate(actions_list):
                actions_onehot_list[i][actions_list[i]]= 1

            return actions_list, actions_onehot_list, predictions_list

    def choose_action(self, state):
        if self.continuous_action_space:
            # Use the networker to predict the next action to take, using the model
            prediction = self.Actor.predict(state)

            low, high = -1.0, 1.0 # -1 and 1 are boundaries of tanh
            action = prediction + np.random.uniform(low, high, size=prediction.shape) * self.std
            action = np.clip(action, low, high)

            logp_t = self.gaussian_likelihood(action, prediction)

            return action[0], action , logp_t[0]
        else:
            prediction = self.Actor.predict(state)[0]
            action = np.random.choice(self.action_size, p=prediction)
            action_onehot = np.zeros([self.action_size])
            action_onehot[action] = 1
            return action, action_onehot, prediction

    def reshape_state(self,state,n_workers=1):
        return np.reshape(state, [n_workers, self.state_size[0]])

    def replay(self, buffer):
        # reshape memory to appropriate shape for training
        states = np.vstack(buffer.states)
        next_states = np.vstack(buffer.next_states)
        actions = np.vstack(buffer.actions)
        predictions = np.vstack(buffer.predictions)
        rewards = buffer.rewards
        dones = buffer.dones

        # Get Critic networker predictions
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array pack all advantages, predictions and actions to y_true and when they are received in custom loss function we unpack it

        y_true = np.hstack([advantages, actions, predictions])

        # training Actor and Critic networkers
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        if self.continuous_action_space:
            # calculate loss parameters (should be done in loss, but couldn't find workering way how to do that with disabled eager execution)
            pred = self.Actor.predict(states)
            #log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
            #logp = self.gaussian_likelihood(actions, pred, log_std)
            logp = self.gaussian_likelihood(actions, pred)
            approx_kl = np.mean(predictions - logp)
            approx_ent = np.mean(-logp)
            print()
            print('approx_kl',approx_kl)
            print('approx_ent',approx_ent)
            print()
        self.replay_count += 1

        buffer.reset()

    ### Equal fns
    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.90, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def load(self):
#         self.Actor.Actor.load_weights(self.Actor_name)
#         self.Critic.Critic.load_weights(self.Critic_name)
        pass

    def save(self):
#         self.Actor.Actor.save_weights(self.Actor_name)
#         self.Critic.Critic.save_weights(self.Critic_name)
        pass

    def checkpoint(self, score, episode):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        saving = False
        # saving best models
        if self.average_[-1] >= self.max_average:
            self.max_average = self.average_[-1]
            self.save()
            # decrease learning rate every saved model
            self.lr *= 0.95
            K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
            saving = True
            print()
            print('New record')
            print()

        if str(episode)[-2:] == "00":# much faster than episode % 100
            # Do some logging
            pass

        return self.average_[-1]

    def run_batch(self):
        state = self.env.reset()
        state = self.reshape_state(state)
        done, score = False, 0
        while True:
            # Instantiate or reset games memory
            buffer = PpoBuffer()

            for t in range(self.training_batch):
                #self.env.render()
                # Actor picks an action
                action, action_data, prediction = self.choose_action(state)

                # Retrieve new state, reward, and whether the state is terminal

                next_state, reward, done, _ = self.env.step(action)
                next_state = self.reshape_state(next_state)
                # Memorize (state, next_states, action, reward, done, logp_ts) for training
                buffer.states.append(state)
                buffer.next_states.append(next_state)

                buffer.actions.append(action_data)
                buffer.rewards.append(reward)
                buffer.dones.append(done)

                buffer.predictions.append(prediction)

                # Update current state shape
                state = next_state
                score += reward
                if done:
                    self.episode += 1
                    average = self.checkpoint(score, self.episode)
                    #if str(self.episode)[-2:] == "00":
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, ''))
                    state, done, score = self.env.reset(), False, 0
                    state = self.reshape_state(state) #np.reshape(state, [1, self.state_size[0]])


            if self.episode >= self.EPISODES:
                break

            self.replay(buffer)


        self.env.close()

    def run_multiprocesses(self, n_workers = False, epochs=False):
        if n_workers == False:
            n_workers = self.n_workers
        if epochs == False:
            epochs = self.epochs

        workers, environment_connections, agent_connections = [], [], []
        for idx in range(n_workers):

            environment_connection, agent_connection = Pipe()
            worker = Environment(idx, agent_connection, self.env_name, self.state_size[0], self.action_size)
            worker.start()
            workers.append(worker)
            environment_connections.append(environment_connection)
            agent_connections.append(agent_connection)

        buffers =       [PpoBuffer() for _ in range(n_workers)]
        score =         [0 for _ in range(n_workers)]
        state =         [0 for _ in range(n_workers)]

        for worker_id, environment_connection in enumerate(environment_connections):
            state[worker_id] = environment_connection.recv()
            print(worker_id)

        while self.episode < self.EPISODES:
            action_list, action_data_list, prediction_list = self.act_batch(self.reshape_state(state,n_workers=n_workers))

            for worker_id, environment_connection in enumerate(environment_connections):
                #print('action_list[worker_id].shape',action_list[worker_id].shape,action_list[worker_id].dtype)
                environment_connection.send(action_list[worker_id])
                buffers[worker_id].actions.append(action_data_list[worker_id])
                buffers[worker_id].predictions.append(prediction_list[worker_id])

            for worker_id, environment_connection in enumerate(environment_connections):
                next_state, reward, done, _ = environment_connection.recv()

                buffers[worker_id].states.append(state[worker_id])
                buffers[worker_id].next_states.append(next_state)
                buffers[worker_id].rewards.append(reward)
                buffers[worker_id].dones.append(done)
                state[worker_id] = next_state
                score[worker_id] += reward

                if done:
                    average = self.checkpoint(score[worker_id], self.episode)
                    print("episode: {}/{}, worker: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, worker_id, score[worker_id], average, ''))
                    # self.writer.add_scalar(f'workers:{n_workers}/score_per_episode', score[worker_id], self.episode)
                    # self.writer.add_scalar(f'workers:{n_workers}/learning_rate', self.lr, self.episode)
                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1

            for worker_id in range(n_workers):
                if len(buffers[worker_id].states) >= self.training_batch:
                    self.replay(buffers[worker_id])


        # terminating processes after while loop
        workers.append(worker)
        for worker in workers:
            worker.terminate()
            print('TERMINATED:', worker)
            worker.join()

    def learn(self, timesteps=-1, plot_results=True, reset=True, success_threshold=False, log_level=1, log_every=50 , success_threshold_lookback=100):
        success_threshold = self.on_learn_start(timesteps,success_threshold,reset,success_threshold_lookback)
        pass
if __name__ == '__main__':
    # discrete_agent = PpoAgent('CartPole-v1', training_batch=4000,epochs=20,lr=3e-4,episodes=2000, continuous_action_space=False)
    # discrete_agent.run_multiprocesses(n_workers=2)
    # discrete_agent = PpoAgent('LunarLander-v2', training_batch=4000,epochs=20,lr=3e-4,episodes=2000, continuous_action_space=False)
    # discrete_agent.run_multiprocesses(n_workers=2)
    continuous_agent = PpoAgent('BipedalWalker-v3',training_batch=1000, epochs=10,episodes=10000, continuous_action_space=True)
    continuous_agent.run_multiprocesses(n_workers=8)
