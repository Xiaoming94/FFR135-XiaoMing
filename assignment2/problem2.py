# problem2.py
# Author: Henry Yang (940503-1056)

import numpy as np
import os
from singlelayernetwork import LCNetwork

learning_rate = 0.02

def test_linear_separability(targets, input_data): 
    for i in range(10):
        net = LCNetwork(input_length)
        net.train(input_data,targets,learning_rate)
        outputs = np.array([net.feed(p) for p in input_data])
        error_count = np.sum(np.abs(np.sign(outputs)-targets))
        if error_count == 0:
            return True
    
    return False

input_data = np.genfromtxt(os.path.join('.','input_data_numeric.csv'),delimiter=',')
_,input_length = np.shape(input_data[:,1:])

targets_mat = np.array([[-1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1]
         ,[1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1]
         ,[1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1]
         ,[1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1]
         ,[1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1]
         ,[-1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1]
         ])

separable_results = []

for targets in targets_mat:
    separable = test_linear_separability(targets, input_data[:,1:])
    separable_results.append(separable)


for (i,separable) in zip(list(range(len(separable_results))),separable_results):
    if separable:
        print("function %s is linearly separable" % i)
    else:
        print("function %s is not linearly separable" % i)