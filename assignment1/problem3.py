# Author: Henry Yang

import numpy as np
from stochastichopfield import StochasticHopfield

experiments = 100
bigT = 10 ** 5
bigN = 200
beta = 2
patterns = [5,40]


for p in patterns:

    orderParam = 0
    for _ in range(experiments):
        net = StochasticHopfield(patterns = p, bitts = bigN, beta = beta)
        for i in range(bigN):
            net.weights[i,i] = 0

        m = 0
        randIndices = np.random.randint(200,size=bigT)

        x1 = net.patterns[1]
        states = x1.copy()


        for i in randIndices :
            new_si = net.feedAsync(states,i)
            states[i] = new_si
            m = m + (states @ x1)/bigN

        orderParam += m/bigT
    
    print("Order parameter for p = %s : %s" % (p,orderParam/experiments))

