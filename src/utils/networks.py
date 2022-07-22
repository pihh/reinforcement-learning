from tensorflow.keras.layers import Input, Dense,Concatenate

def discrete_actor_output(self,common_layer,n_actions):
    return Dense(n_actions, activation="softmax")(common_layer)

def continous_actor_output(self,common_layer,n_actions):
    sigma = Dense(n_actions, activation="softplus", name="sigma")(common_layer)
    mu = Dense(n_actions, activation="tanh" , name='mu')(common_layer)
    
    return Concatenate(axis=-1)([mu,sigma])