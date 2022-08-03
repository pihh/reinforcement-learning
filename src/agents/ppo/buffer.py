class PpoBuffer():
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.predictions = []
        self.dones = [] 

    def remember(self,state,next_state,action,reward,done,prediction):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action_onehot)
        self.rewards.append(reward)
        self.dones.append(done)
        self.predictions.append(prediction)