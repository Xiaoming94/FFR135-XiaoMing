# problem1.py
# Author: Henry Yang (940503-1056)

import numpy as np
from perceptron import Perceptron
import os
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt

###
# Function that loads the MNIST data-set
###
def load_data(path=os.path.join("."),type="train"):
    filenames = {
        'train' : ('train-images-idx3-ubyte','train-labels-idx1-ubyte'),
        'val'   : ('t10k-images-idx3-ubyte','t10-labels-idx1-ubyte')
    }

    data_file, labels_file = filenames[type]

    data, labels = loadlocal_mnist(
        images_path=os.path.join(path,data_file),
        labels_path=os.path.join(path,labels_file)
    )

    return data,labels

##
# Verification function
#

def print_network(net):
    for l in net.layers :
        print("\n======= NEW LAYER =======\n")
        print(l.weights)
        print(l.weights.shape)
        print(l.thresholds)

output_config = {
    'count' : 1,
    'activation': np.tanh
}

hidden1_config = {
    'count' : 10,
    'activation': np.tanh
}

config = {
    'input' : 2,
    'output' : output_config,
    'hidden' : [hidden1_config]
}

def gen_testdata(data_num):
    data_pos = np.random.normal(0,1,size=(data_num,2))
    max_x = np.max(data_pos[:,0])
    max_y = np.max(data_pos[:,1])

    data_neg = np.random.rand(data_num,2) * 2 - 1
    data_neg = data_neg * np.array([max_x,max_y])

    labelpos = np.ones(data_num)
    labelneg = -1 * np.ones(data_num)

    data = np.concatenate((data_pos,data_neg),axis=0)
    labels = np.concatenate((labelpos,labelneg),axis=None)

    return data, labels


epochs = 40
batchsize = 10

train_data, train_labels = gen_testdata(5000)
val_data, val_labels = gen_testdata(1000)

train_labels = train_labels.reshape(np.size(train_labels),1)

net = Perceptron(config)

print_network(net)

val_energy = net.train(epochs, batchsize,0.01,train_data,train_labels,val_data,val_labels)

plt.plot(np.arange(epochs),val_energy)
plt.show()