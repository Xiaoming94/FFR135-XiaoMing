# problem3.py
# Author: Henry Yang

import numpy as np
from stochastichopfield import StochasticHopfield

########################################
# Script file for problem 3            #
# Using a stochastic Hopfield model    #
# Calculating the orderparameter       #
# With the two given set of parameters #
########################################

## Given parameters
experiments = 100
bigT = 10 ** 5
bigN = 200
beta = 2
patterns = [5,40]

# Performing the experiments
# Calculates the average order parameter using cumulated sums
# Than dividing with numbers of experiments

for p in patterns:

    # Declaraing the cumulative sum
    orderParam = 0
    for _ in range(experiments):
        net = StochasticHopfield(patterns = p, bitts = bigN, beta = beta)
        for i in range(bigN):
            net.weights[i,i] = 0

        # Inner orderparameters
        m = 0

        # Choosing the random neurons
        randIndices = np.random.randint(200,size=bigT)
        
        # Picking the first pattern
        x1 = net.patterns[1]
        states = x1.copy()

        # Calculating the order parameters
        for i in randIndices :
            new_si = net.feedAsync(states,i)
            states[i] = new_si
            m = m + (states @ x1)/bigN

        orderParam += m/bigT

    # Printing the results    
    print("Order parameter for p = %s : %s" % (p,orderParam/experiments))

