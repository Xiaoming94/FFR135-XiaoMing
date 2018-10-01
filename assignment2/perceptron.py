# perceptron.py
# Author: Henry Yang (940503-1056)

import numpy as np

c_error_t = 0.12
###############################################################
# Classfile for two layer perceptron.                         #
# Implements Stochastic gradient descent with Backpropagation #
# Slightliy different from lecture slides                     # 
# All instances of this network got two inputs and one output #
############################################################### 

class TLPerceptron :
    def __init__(self,size_m1,size_m2,learning_rate):
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

        self.learning_rate = learning_rate

    def feed(self,input):
        states = {}
        state_vj = np.tanh(self.weight_jk @ input - self.threshold_j)
        states['vj'] = state_vj
        state_vi = np.tanh(self.weight_ij @ state_vj - self.threshold_i)
        states['vi'] = state_vi
        output = np.tanh(self.weight_i @ state_vi - self.threshold_o)
        return (output, states)
    
    def train(self, training_set, validation_set, batch_size = 1):
        energy_arr = []
        c_errors = []
        c_error = 1
        num_data,_ = np.shape(training_set)
        #valdata_num,_ = np.shape(validation_set)
        iterations = 0
        #while c_error > c_error_t:
        mu_i = np.random.randint(num_data)
        tdata_row = training_set[mu_i,:]
        t_point = tdata_row[0:2]
        t_label = tdata_row[-1]
        output,states = self.feed(t_point)
        energy = 1/2 * (t_label - output)**2
        energy_arr.append(energy)
        update_vals = self.backpropagation(t_point, output, t_label, states )
        print(update_vals)
        print(update_vals['dw_i'].shape)
        print(update_vals['dw_ij'].shape)
        print(update_vals['dw_jk'].shape)

    def backpropagation(self, input , output, target ,states):
        update_vals = {}
        delta = -1 * ( target - output ) * (1 - output ** 2)
        update_vals['dw_i'] = delta * states['vi']
        update_vals['dto'] = delta * (-1)
        delta = delta * self.weight_i * (1 - (states['vi'] ** 2))
        update_vals['dw_ij'] = delta.reshape(np.size(states['vi']),1) @ states['vj'].reshape(1,np.size(states['vj']))
        update_vals['dti'] = delta * (-1)
        delta = (np.transpose(self.weight_ij) @ delta) * (1 - states['vj'] ** 2)
        update_vals['dw_jk'] = delta.reshape(np.size(states['vj']),1) @ input.reshape(1,2)
        update_vals['dtj'] = delta * (-1)

        return update_vals


