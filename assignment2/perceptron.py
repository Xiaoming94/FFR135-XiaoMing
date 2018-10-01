# perceptron.py
# Author: Henry Yang (940503-1056)

import numpy as np

###############################################################
# Classfile for two layer perceptron.                         #
# Implements Stochastic gradient descent with Backpropagation #
# Slightliy different from lecture slides                     # 
# All instances of this network got two inputs and one output #
############################################################### 

class TLPerceptron :
    def __init__(self,size_m1,size_m2):
        # Initialising Weight matrix input -> Hidden1
        self.weight_jk = np.random.rand(size_m1,2) * 2 - 1

        # Initialising Weight Matrix Hidden1 -> Hidden2
        self.weight_ij = np.random.rand(size_m2,size_m1) * 2 - 1

        # Initialising Weight Matrix Hidden2 -> output
        self.weight_i = np.random.rand(size_m2) * 2 - 1

        # Initialising Threshold1
        self.threshold_j = np.random.rand(size_m1) * 2 - 1
        
        # initialising Threshold2
        self.threshold_i = np.random.rand(size_m2) * 2 - 1

        # initialising Output Threshold
        self.threshold_o = np.random.rand() * 2 - 1

    def feed(self,input):
        states = []
        state_vj = np.tanh(self.weight_jk @ input - self.threshold_j)
        states.append(state_vj)
        state_vi = np.tanh(self.weight_ij @ state_vj - self.threshold_i)
        states.append(state_vi)
        output = np.tanh(self.weight_i @ state_vi - self.threshold_o)
        return (output, states)