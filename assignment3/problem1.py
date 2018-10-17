# problem1.py
# Author: Henry Yang (940503-1056)

import numpy as np
from perceptron import Perceptron
import os
import matplotlib.pyplot as plt
import dataloader

##
# Verification function
#

def print_network(net):
    for l in net.layers :
        print("\n======= NEW LAYER =======\n")
        print(l.weights)
        print(l.weights.shape)
        print(l.thresholds)

def gen_testdata(data_num):
    data_pos = np.random.normal(0,1,size=(data_num,2))
    data_neg = np.random.normal(0,3,size=(data_num,2))

    labelpos = np.ones(data_num)
    labelneg = -1 * np.ones(data_num)

    data = np.concatenate((data_pos,data_neg),axis=0)
    labels = np.concatenate((labelpos,labelneg),axis=None)

    return data, labels

def pn_cerror(output,target):
    num_of_patterns = target.shape[0]
    return (1/(2 * num_of_patterns)) * np.sum(np.abs(target-np.sign(output)))

def bin_cerror(output,target):
    num_of_patterns = target.shape[0]
    h_output = classify(output - 1/2)
    return (1/num_of_patterns) * np.sum(np.abs(target)-h_output)

def classify(output):
    classified = np.sign(output)
    classified[np.where(classified < 0)] = 0
    return classified

output_config = {
    'count' : 1,
    'activation' : np.tanh
}

h1_config = {
    'count' : 128,
    'activation' : np.tanh
}

h2_config = {
    'count' : 32,
    'activation' : np.tanh
}

config = {
    'input' : 2,
    'output' : output_config,
    'hidden' : [h1_config,h2_config]
}

epochs = 40
batchsize = 10

train_data, train_labels = gen_testdata(5000)

val_data, val_labels = gen_testdata(1000)

train_labels = train_labels.reshape(np.size(train_labels),1)
val_labels = val_labels.reshape(np.size(val_labels),1)

net = Perceptron(config)

print_network(net)

val_energy, val_cerror, best_net = net.train(epochs, batchsize,0.01,train_data,train_labels,val_data,val_labels,pn_cerror)

plt.plot(np.arange(epochs),val_energy)
plt.figure()

plt.plot(np.arange(epochs),val_cerror)

tout,_ = best_net.feed_forward(train_data)
vout,_ = best_net.feed_forward(val_data)

print("Accuracy on training data: %s " % (1-pn_cerror(tout,train_labels)))
print("Accuracy on validation data: %s " % (1-pn_cerror(vout,val_labels)))

plt.show()
