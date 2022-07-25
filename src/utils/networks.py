from tensorflow.keras.layers import Input, Dense,Concatenate

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