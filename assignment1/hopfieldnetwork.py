# hopfieldnetwork.py
# author: Henry Yang (940503-1056)

import numpy as np
import random

###############################################
# Class for the Hopfield Model Neural Network #
# Using McCulloch Pitts Neurons               #
###############################################

class HopField(object):
    def __init__(self,patterns,bitts):
        # Generating Patterns
        pat = []
        w = np.zeros((bitts,bitts))
        for n in range(patterns):
            p = np.random.randint(2,size=(bitts,1)) * 2 - 1
            pat.append(p)

            # Calculating weights Using Hebb's Rule
            w = w + np.multiply((1/bitts),np.multiply(p,p.transpose()))
        
        # Storing The Patterns
        self.patterns = pat

        # Storing the weights
        self.weights = w

    # Asynchronus feed Update
    def feedAsync(self,input,neuronIndex):
        #(bitts,_) = self.weights.shape
        #neuronIndex = np.random.randint(low=0,high=bitts)
        neuronWeights = self.weights[neuronIndex,:]
        z = neuronWeights @ input
        neuronState = np.sign(z)
        if neuronState == 0:
            neuronState = 1
        return neuronState
    
    # Synchronous feed.
    # Implemented as a matrix multiplication
    def feedSync(self,input):
        z_vec = self.weights @ input
        neuronStates = np.sign(z_vec)
        neuronStates[np.where(neuronStates == 0)] = 1
        return neuronStates


