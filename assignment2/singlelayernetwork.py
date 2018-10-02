# singlelayernetwork.py
# Author: Henry Yang (940503-1056)

import numpy as np

iterations = 10 ** 5

class LCNetwork:
    def __init__(self,i_n):
        self.inputs = i_n
        self.weights = np.random.uniform(-0.2,0.2,i_n)
        self.threshold = np.random.rand() * 2 - 1
    
    def feed(self,input):
        return np.tanh((1/2) * self.weight @ input - self.threshold)
    
    def train(self,data, targets, learning_rate):
        energy_arr = []
        indices = np.random.randint(2 ** self.inputs,size=iterations)
        for i in iterations:
            energy_arr.append(calc_energy(data,targets))
            sample = data[i]
            output = self.feed(sample)
            target = targets[i]
            dw_i, dt = self.backprop(sample,output,target,)
            self.weights -= learning_rate * dw_i
            self.threshold -= learning_rate * dt
        
        return energy_arr

    def backprop(self,input,output, target):
        delta = -1*(target - output) * (1 - output ** 2)
        dw_i = delta * input
        dt = delta * (-1)
        return dw_i, dt
    
    def calc_energy(self,data,targets):
        outputs = np.array([self.feed(x) for x in data])
        return (1/2) * np.sum((targets - outputs)**2)