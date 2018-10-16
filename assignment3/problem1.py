# problem1.py
# Author: Henry Yang (940503-1056)

import numpy as np
from perceptron import Perceptron

def print_network(net):
    for l in net.layers :
        print(l.weights)
        print(l.weights.shape)
        print(l.thresholds)

output_config = {
    'count' : 2,
    'activation': np.tanh
}

config = {
    'input' : 10,
    'output' : output_config,
    'hidden' : []
}

net = Perceptron(config)
test_input = np.random.random(10)
print(test_input)
print_network(net)

output, states = net.feed_forward(test_input)

print(output)