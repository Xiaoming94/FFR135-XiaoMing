# hopfieldnetwork.py
# author: Henry Yang (940503-1056)

import numpy as np
import random

###############################################
# Class for the Hopfield Model Neural Network #
# Using McCulloch Pitts Neurons               #
###############################################

def genRandPatterns(patterns,bitts):
    pat = []
    w = np.zeros((bitts,bitts))
    for n in range(patterns):
        p = np.random.randint(2,size=(bitts,1)) * 2 - 1
        pat.append(p)

        # Calculating weights Using Hebb's Rule
        w = w + np.multiply((1/bitts),np.multiply(p,p.transpose()))
    
    return (pat,w)

def calcWeights(data):
    bitts = data[0].size
    w = np.zeros((bitts,bitts))
    for p in data :
        w = w + np.multiply((1/bitts),np.multiply(p,p.transpose()))

    return w

    
class HopField(object):
    def __init__(self,data=None,patterns=10,bitts=5):
        # Generating Patterns
        if data == None :
            (pat, w) = genRandPatterns(patterns,bitts)
        
        else :
            pat = data
            w = calcWeights(data)

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
