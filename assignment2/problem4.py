import numpy as np
import os
from perceptron import TLPerceptron
import matplotlib.pyplot as plt

size_m1 = 10
size_m2 = 21

training_set_path = os.path.join('.','training_set.csv')
validation_set_path = os.path.join('.','validation_set.csv')

training_set = np.genfromtxt(training_set_path,delimiter = ',')
validation_set = np.genfromtxt(validation_set_path,delimiter = ',')

net = TLPerceptron(size_m1,size_m2,0.01)

c_errors, train_energy_arr, val_energy_arr = net.train(training_set,validation_set)

train_line, = plt.plot(np.arange(len(train_energy_arr)),train_energy_arr,'b-')
val_line, = plt.plot(np.arange(len(val_energy_arr)),val_energy_arr,'g-')
plt.legend([train_line, val_line],["Training Energy","Validation Energy"])
plt.grid(b=True)
training_points = training_set[:,0:2]
validation_points = validation_set[:,0:2]

training_targets = training_set[:,-1]
validation_targets = validation_set[:,-1]

train_true = training_points[np.where(training_targets == 1)]
train_false = training_points[np.where(training_targets == -1)]

plt.figure()
plt.scatter(train_true[:,0],train_true[:,1],c='b',marker='o')
plt.scatter(train_false[:,0],train_false[:,1],c='r',marker='o')
plt.show()