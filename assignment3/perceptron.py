# perceptron.py
# Author: Henry Yang

import numpy as np

############################################
# Base class for a multilayer Perceptron   #
############################################

class PerceptronLayer:
    def __init__(self, input, output, activation = np.tanh):
        self.neurons = output
        weights = np.random.normal(0,1,size=(output,input))
        if output == 1:
            weights = weights.reshape(output,input)
        
        self.weights = weights
        self.thresholds = np.zeros(output)
        self.activation = activation
    
    def feed(self, input_pattern):
        # Assuming that input_patterns are colon vectors

        synapse_in = self.weights @ input_pattern
        synapse_in = np.transpose(synapse_in) - self.thresholds
        return np.transpose(self.activation(synapse_in))

##
# The Config object is a dictionary that contains the keys input, output, and hidden
# Input is an integer
# Output is a dictionary with the keys output, and activation
# hidden is a list of similar dictionaries as output

class Perceptron:
    def __init__(self, config):
        self.layers = construct_layers(config)
    
    def feed_forward(self, input_pattern):
        states = []
        current_state = np.transpose(input_pattern)
        for l in self.layers:
           
            states.append(np.transpose(current_state))
            current_state = l.feed(current_state)
        
        return np.transpose(current_state), states

    def train(self, epochs, batchsize, 
        learning_rate, train_data, train_targets,
        val_data, val_targets):
        data_count = train_data.shape[0]
        indices = np.arange(data_count)
        val_energy_vec = np.zeros(epochs)
        for i in range(epochs):
            batches = np.reshape(np.random.permutation(indices),(-1,batchsize))
            for batch in batches:
                self.backprop_update(learning_rate, train_data[batch], train_targets[batch],batchsize)
            
            output, _ = self.feed_forward(val_data)
            val_energy = energy_function(output,val_targets)
            print("current validation energy: %s" % val_energy)
            val_energy_vec[i] = val_energy

        return val_energy_vec
    
    def backprop_update(self, learning_rate, data, targets, batchsize):

        output, states = self.feed_forward(data)
        list_lsp = list(zip(self.layers,states))
        delta = 1/batchsize * np.sum((targets - output),axis=0) * -1
        current_states = 1/batchsize * np.sum(output,axis=0)
        for l,s in reversed(list_lsp):
            diffg = differentiate(l.activation,current_states)
            delta = delta * diffg
            pstates = np.sum(s,axis=0) * 1/batchsize
            pstates = pstates.reshape(1,np.size(pstates))
            dw = delta.reshape(np.size(delta),1) @ pstates
           
            delta = np.transpose(l.weights) @ np.transpose(delta)
            l.weights -= learning_rate * dw
            l.thresholds -= learning_rate * delta.flatten() * (-1)
            delta = np.transpose(delta)
            current_states = pstates

   
            
def construct_layers(config):
    layers = []
    before = config['input']
    if config['hidden']:
        for c in config['hidden']:
            current = c['count']
            layers.append(PerceptronLayer(before,current,c['activation']))
            before = current
    
    output = config['output']
    layers.append(PerceptronLayer(before,output['count'],output['activation']))
    return layers

def energy_function(output, targets):
    return 1/2 * np.sum((targets - output)**2)

def differentiate(activation,current_states):
       gradients = {
           'tanh'    : (1 - current_states ** 2)
       }
       return gradients[activation.__name__]