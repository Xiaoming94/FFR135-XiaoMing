import numpy as np
import os
from perceptron import TLPerceptron
import matplotlib.pyplot as plt

size_m1 = 8
size_m2 = 4

training_set_path = os.path.join('.','training_set.csv')
validation_set_path = os.path.join('.','validation_set.csv')

training_set = np.genfromtxt(training_set_path,delimiter = ',')
validation_set = np.genfromtxt(validation_set_path,delimiter = ',')

net = TLPerceptron(size_m1,size_m2,0.01)

c_errors, energy_arr = net.train(training_set,validation_set)

plt.plot(np.arange(len(energy_arr)),energy_arr)
plt.show()