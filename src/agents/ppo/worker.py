import gym 
import numpy as np

from multiprocessing import Process

# Worker
class Worker(Process):
    def __init__(self, environment, agent_connection, worker_idx, state_size, action_size):
        super(Worker, self).__init__()

        self.env = environment()
        self.worker_idx = worker_idx
        self.agent_connection = agent_connection
        self.state_size = state_size
        self.action_size = action_size

    def run(self):
        super(Worker, self).run()
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