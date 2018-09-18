# Author: Henry Yang

import numpy as np
import random
from hopfieldnetwork import HopField

class StochasticHopfield(HopField):
    def __init__(self,usedata=False,data=None,patterns=10,bitts=5,beta=1):
        super(StochasticHopfield, self).__init__(usedata,data,patterns,bitts)
        self.beta = beta
    
    # Definition of gFunction according to lecture slides
    def gFunction(self,b):
        nat_e = np.exp(-2 * self.beta * z)
        return 1/(1 + nat_e)

    # Asynchronous feed using Stochastic dynamics
    def feedAsync(self,input,neuronIndex):
        neuronWeights = self.weights[neuronIndex,:]
        b = neuronWeights @ input
        g = gFunction(b)
        r = random.random()
        return 1 if r < g else -1
    
