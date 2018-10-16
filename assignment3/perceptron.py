# perceptron.py
# Author: Henry Yang

import numpy as np

############################################
# Base class for a multilayer Perceptron   #
############################################

class PerceptronLayer:
    def __init__(self, input, output, activation = np.tanh):
        self.weights = np.random.normal(0,1,size=(output,input))
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
        current_state = input_pattern
        for l in self.layers:
            current_state = l.feed(current_state)
            states.append(current_state)
        
        return current_state,states

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