# problem1b.py
# Author: Henry Yang (940503-1056)

import numpy as np
from hopfieldnetwork import HopField

#########################################################
# This is the source file for solving the first second  #
# part of problem 1 in Assignment1 of FFR135            #
# The parameters chosen are given by the Assignnment    #
#########################################################

patterns = [12,20,40,60,80,100]
trials = 10 ** 5
bitts = 100
p_errors_list = []

for p in patterns :
    errors = 0
    for i in range(trials):
        net = HopField(patterns=p,bitts=bitts)
          
        # Randomly chooose a pattern and which neuron to observe
        n_i = np.random.randint(bitts)
        p_i = np.random.randint(p)
        inputPattern = net.patterns[p_i]
        
        neuronState = net.feedAsync(inputPattern,n_i)
        if(neuronState != inputPattern[n_i]):
            errors += 1
    
    p_error = errors/trials
    print("p_error : %s , for %s patterns" % (p_error,p))

    p_errors_list.append(p_error)

print(p_errors_list)

        