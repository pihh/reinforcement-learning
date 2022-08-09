from pydoc import describe
import gym 
import numpy as np

from multiprocessing import Process

# Worker
class Worker(Process):
    def __init__(self, worker_id, environment, agent_connection,  observation_shape=False, n_actions=False):
        super(Worker, self).__init__()

        print('* Booting worker {}'.format(worker_id))

        # Create it's own environment
        self.env = environment(describe=False)

        # Args
        self.worker_id = worker_id
        self.agent_connection = agent_connection
        # self.observation_shape = observation_shape
        # self.n_actions = n_actions

    def run(self):
        super(Worker, self).run()
        state = self.env.reset()
        #state = self.reshape_state(state)
        #print('worker sending ',state)
        self.agent_connection.send(state)
        while True:
            #print('recv')
            action = self.agent_connection.recv()

            #print('action',action)
            state, reward, done, info = self.env.step(action)
            #state = self.reshape_state(state)

            if done:
                state = self.env.reset()
                #state = self.reshape_state(state)
            self.agent_connection.send([state, reward, done, info])