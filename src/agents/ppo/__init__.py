import os


# CPU first for multiprocessing
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 


import copy
import numpy as np
import tensorflow as tf

from tensorboardX import SummaryWriter

from tensorflow.keras.optimizers import Adam #, RMSprop#, Adagrad, Adadelta
from tensorflow.keras import backend as K



from multiprocessing import Pipe, cpu_count

from src.agents.agent import Agent
from src.agents.ppo.worker import Worker
from src.agents.ppo.networks import PpoActorContinuous
from src.agents.ppo.networks import PpoActorDiscrete
from src.agents.ppo.networks import PpoCritic
from src.agents.ppo.buffer import PpoBuffer
from src.utils.policy import gaussian_likelihood

# Faster performance
tf.compat.v1.disable_eager_execution() 

class PpoAgent(Agent):
    def __init__(self,
                environment,
                gamma = 0.99,
                policy="mlp",
                batch_size=4000, 
                epochs=80, 
                episodes=100000,
                shuffle=False,
                target_kl = 0.01, 
                n_workers=cpu_count(),
                critic_loss_function_version=1,
                loss_clipping=0.2,
                loss_entropy=0.001,
                actor_optimizer=Adam,
                actor_learning_rate=0.00025,
                critic_optimizer=Adam,
                critic_learning_rate=0.00025,
                ):
        super(PpoAgent, self).__init__(environment,args=locals())
        
        # Args
        self.gamma=gamma
        self.policy=policy
        self.batch_size=batch_size
        self.epochs=epochs 
        self.episodes=episodes
        self.actor_optimizer=actor_optimizer
        self.actor_learning_rate=actor_learning_rate
        self.critic_optimizer=critic_optimizer
        self.critic_learning_rate=critic_learning_rate
        self.shuffle=shuffle
        self.target_kl=target_kl 
        self.n_workers=n_workers
        self.critic_loss_function_version=critic_loss_function_version
        self.loss_clipping=loss_clipping
        self.loss_entropy=loss_entropy


        # Bootstrap
        self.__init_networks()
        self.__init_buffers()
        self._add_models_to_config([self.actor.model,self.critic.model])
        self._init_tensorboard()

        # do not change bellow
        self.log_std = -0.5 * np.ones(self.n_actions, dtype=np.float32)
        self.std = np.exp(self.log_std)

        # Bind gaussian likelihood
        self.gaussian_likelihood = gaussian_likelihood(self.log_std, lib="numpy")
        
    def __init_networks(self):
        if self.action_space_mode == "continuous":
            self.actor = PpoActorContinuous(
                observation_shape=self.observation_shape, 
                action_space=self.n_actions, 
                learning_rate=self.actor_learning_rate,
                optimizer=self.actor_optimizer,
                loss_clipping=self.loss_clipping,
                policy=self.policy
            )
        else:
            self.actor = PpoActorDiscrete(
                observation_shape=self.observation_shape, 
                action_space=self.n_actions, 
                learning_rate=self.actor_learning_rate, 
                optimizer=self.actor_optimizer,
                loss_clipping=self.loss_clipping,
                loss_entropy=self.loss_entropy,
                policy=self.policy
            )

        self.critic = PpoCritic(
            observation_shape=self.observation_shape,
            learning_rate=self.critic_learning_rate, 
            optimizer=self.critic_optimizer,
            loss_function_version=self.critic_loss_function_version, 
            loss_clipping=self.loss_clipping,
            action_space_mode=self.action_space_mode, 
            policy=self.policy
        )

    def __init_buffers(self):
        self.buffer = PpoBuffer()
        
    def act_batch(self,states):
        
        if self.action_space_mode=="continuous":
            # Use the networker to predict the next action to take, using the model
            prediction = self.actor.predict(states)

            print(prediction.shape)
            print(prediction)

            low, high = self.action_lower_bounds, self.action_upper_bounds # -1 and 1 are boundaries of tanh
            action = prediction + np.random.uniform(low, high, size=prediction.shape) * self.std
            action = np.clip(action, low, high)

            logp_t = self.gaussian_likelihood(action, prediction)

            return action.astype(np.float32), action.astype(np.float32) , logp_t.astype(np.float32)
        else:
            states = np.array(states)
            predictions_list = self.actor.predict(states)

            actions_list = [np.random.choice(self.n_actions, p=p) for p in predictions_list]
            actions_onehot_list = [np.zeros([self.n_actions]) for p in predictions_list]

            for i, action_list in enumerate(actions_list):
                actions_onehot_list[i][action_list]= 1

            return actions_list, actions_onehot_list, predictions_list

    def act(self, state):
        if self.action_space_mode=="continuous":
            # Use the networker to predict the next action to take, using the model
            prediction = self.actor.predict(state)

            low, high = -1.0, 1.0 # -1 and 1 are boundaries of tanh
            action = prediction + np.random.uniform(low, high, size=prediction.shape) * self.std
            action = np.clip(action, low, high)

            logp_t = self.gaussian_likelihood(action, prediction)

            return action[0], action , logp_t[0]
        else:
            prediction = self.actor.predict(state)[0]

            action = np.random.choice(self.n_actions, p=prediction)
            action_onehot = np.zeros([self.n_actions])
            action_onehot[action] = 1
            return action, action_onehot, prediction

    def reshape_state(self,state,n_workers=1):
        return np.expand_dims(state,axis=0)
        #return np.reshape(state, [n_workers, self.observation_shape])

    def replay(self, buffer):
        # reshape memory to appropriate shape for training
        states = np.vstack(buffer.states)
        next_states = np.vstack(buffer.next_states)
        actions = np.vstack(buffer.actions)
        predictions = np.vstack(buffer.predictions)
        rewards = buffer.rewards
        dones = buffer.dones

        # Get Critic networker predictions
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array pack all advantages, predictions and actions to y_true and when they are received in custom loss function we unpack it

        y_true = np.hstack([advantages, actions, predictions])

        # training Actor and Critic networkers
        a_loss = self.actor.model.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.critic.model.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        if self.action_space_mode=="continuous":
            # calculate loss parameters (should be done in loss, but couldn't find workering way how to do that with disabled eager execution)
            pred = self.actor.predict(states)
            #log_std = -0.5 * np.ones(self.n_actions, dtype=np.float32)
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
    
    def checkpoint(self, score, episode):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-self.success_threshold_lookback:]) / len(self.scores_[-self.success_threshold_lookback:]))
        # saving = False
        # # saving best models
        if len(self.average_) > self.success_threshold_lookback:
            if self.average_[-1] >= self.learning_max_score:
                
                self.learning_max_score = self.average_[-1]
                self.save()

                # decrease learning rate every saved model
                self.actor_learning_rate *= 0.95
                self.critic_learning_rate *= 0.95
                K.set_value(self.actor.model.optimizer.learning_rate, self.actor_learning_rate)
                K.set_value(self.critic.model.optimizer.learning_rate, self.critic_learning_rate)
    
                print()
                print('New record ', self.learning_max_score)
                print('Learning rate decreased')
                print()

        return self.average_[-1]

    def save(self):
        self._save_weights([
            {"name": "ppo-actor","model":self.actor.model},
            {"name": "ppo-critic","model":self.critic.model},
        ])
    
    def load(self):
        self._load_weights([
            {"name": "ppo-actor","model":self.actor.model},
            {"name": "ppo-critic","model":self.critic.model},
        ])

    def run_batch(self,
        timesteps=-1, 
        log_level=1, 
        log_every=50, 
        success_strict=False,
        success_threshold=False, 
        success_threshold_lookback=100, 
        plot_results=True, 
        reset=True):

        success_threshold = self.on_learn_start(timesteps,success_threshold,reset,success_threshold_lookback, success_strict)

        timestep = 0
        episode = 0
        score = 0
        done = False

        state = self.env.reset()
        state = self.reshape_state(state)
            
        while self.learning_condition(timesteps,timestep):  # Run until solved
            
            # Instantiate or reset games memory
            buffer = PpoBuffer()

            # Fill training buffer 
            for _ in range(self.batch_size):
 
                action, action_data, prediction = self.act(state)

                # print('action',action)
                # print('state',state)
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
                    # Episode ended
                    episode += 1
                    # Step reward, tensorboard log score, print progress
                    self.on_learn_episode_end(score,log_every,log_level,success_threshold)
                    
                    # Verify if learning success conditions are met
                    if self.did_finnish_learning(success_threshold,episode):
                        break

                    score=0
                    done=False
                    
                    state=self.env.reset() 
                    state = self.reshape_state(state) #np.reshape(state, [1, self.observation_shape[0]])

                timestep +=1

            # Buffer full, time to learn something
            self.replay(buffer)

        # End of training
        self.env.close()
        
        if plot_results:
            self.plot_learning_results()

    def run_multiprocesses(self, 
        timesteps=-1, 
        log_level=1, 
        log_every=50, 
        success_strict=False,
        success_threshold=False, 
        success_threshold_lookback=100, 
        n_workers=cpu_count(),
        plot_results=True, 
        reset=True
    ):
        
        print('* NOTE: multiprocess refactored not tested yet')

        timestep = 0
        episode = 0

        success_threshold = self.on_learn_start(
            timesteps,
            success_threshold,
            reset,
            success_threshold_lookback,
            success_strict)

        # Setup trackers
        workers=[] 
        environment_connections = [] 
        agent_connections = []

        # Create pipes
        for idx in range(n_workers):
            # Start connection pipe
            environment_connection, agent_connection = Pipe()

            # Boot worker
            worker = Worker(idx, self._environment,agent_connection, self.observation_shape, self.n_actions)
            worker.start()
            workers.append(worker)

            # Boot connection
            environment_connections.append(environment_connection)
            agent_connections.append(agent_connection)

        # Create buffers and scores
        buffers =       [PpoBuffer()    for _ in range(n_workers)]
        score =         [0              for _ in range(n_workers)]
        state =         [0              for _ in range(n_workers)]

        def shutdown(workers):
            for worker in workers:
                #self.env.close()
                worker.terminate()
                print('Worker {} has shut down'.format(worker))
                worker.join()

        # Start comunication
        for worker_id, environment_connection in enumerate(environment_connections):
            state[worker_id] = environment_connection.recv()

        # Start learning 
        while self.learning_condition(timesteps,timestep):  
            
            # Run until solved
            for _ in range(self.batch_size):
                
                action_list, action_data_list, prediction_list = self.act_batch(state)
           
                for worker_id, environment_connection in enumerate(environment_connections):
                    environment_connection.send(action_list[worker_id])
                    buffers[worker_id].actions.append(action_data_list[worker_id])
                    buffers[worker_id].predictions.append(prediction_list[worker_id])

                for worker_id, environment_connection in enumerate(environment_connections):
                    next_state, reward, done, _ = environment_connection.recv()

                    buffers[worker_id].states.append(state[worker_id])
                    buffers[worker_id].next_states.append(next_state)
                    buffers[worker_id].rewards.append(reward)
                    buffers[worker_id].dones.append(done)
                    
                    # Update current state
                    state[worker_id] = next_state
                    score[worker_id] += reward

                    if done:
                        # Episode ended
                        episode += 1
                        # Step reward, tensorboard log score, print progress
                        self.on_learn_episode_end(score[worker_id],log_every,log_level,success_threshold)
                        
                        
                        # Reset score
                        score[worker_id] = 0
                        
                    timestep +=1

            # Verify if learning success conditions are met
            if self.did_finnish_learning(success_threshold,episode):
                #shutdown(workers)
                break

            print()
            print('* Will replay')
            for worker_id in range(n_workers):
                self.replay(buffers[worker_id])
                print('* Worker {} finnished learning phase'.format(worker_id))
            print('* Will resume')
            print()

        # Ended training
        shutdown(workers)

        if plot_results:
            self.plot_learning_results()

    def learn(self, 
        timesteps=-1, 
        log_level=1, 
        log_every=50, 
        success_strict=False,
        success_threshold=False, 
        success_threshold_lookback=100, 
        plot_results=True, 
        reset=True
    ):

        # Instantiate plot memory
        self.scores_ = []
        self.episodes_ = []
        self.average_ = []
        
        self.replay_count = 0
        self.timestep = 0
        self.episode = 0

        self.log_every = log_every

        if self.n_workers == 1:
            self.run_batch(
                timesteps=timesteps, 
                plot_results=plot_results, 
                reset=reset, 
                success_threshold=success_threshold, 
                log_level=log_level, 
                log_every=log_every, 
                success_threshold_lookback=success_threshold_lookback, 
                success_strict=success_strict)
                
        elif self.n_workers > 1:
            self.run_multiprocesses(timesteps=timesteps, 
                plot_results=plot_results, 
                reset=reset, 
                success_threshold=success_threshold, 
                log_level=log_level, 
                log_every=log_every, 
                success_threshold_lookback=success_threshold_lookback, 
                success_strict=success_strict, 
                n_workers=self.n_workers)
        

