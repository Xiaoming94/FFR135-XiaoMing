# hopfieldnetwork.py
# author: Henry Yang (940503-1056)

import numpy as np
import random

###############################################
# Class for the Hopfield Model Neural Network #
# Using McCulloch Pitts Neurons               #
###############################################

def genRandPatterns(patterns,bitts):
    pat = np.random.randint(2,size=(patterns,bitts)) * 2 - 1
    return pat

def calcWeights(patterns):
    (_,bitts) = patterns.shape 
    w = np.zeros((bitts,bitts))
    for p in patterns :
        vecP = p.reshape(bitts,1)
        w = w + vecP @ np.transpose(vecP)


    return 1/bitts * w


class HopField(object):
    def __init__(self,usedata=False,data=None,patterns=10,bitts=5):
        # Generating Patterns
        if not usedata :
            pat = genRandPatterns(patterns,bitts)

        else :
            pat = data

        # Storing The Patterns
        self.patterns = pat

        # Storing the weights
        self.weights = calcWeights(pat)

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
