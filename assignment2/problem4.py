import numpy as np
import os
from perceptron import TLPerceptron

size_m1 = 8
size_m2 = 4

training_set_path = os.path.join('.','training_set.csv')
validation_set_path = os.path.join('.','validation_set.csv')

training_set = np.genfromtxt(training_set_path,delimiter = ',')
validation_set = np.genfromtxt(validation_set_path,delimiter = ',')

train_data = training_set[:,0:2]
train_targets = training_set[:,-1]

val_data = validation_set[:,0:2]
val_targets = validation_set[:,-1]

