import numpy as np
import os
from perceptron import TLPerceptron
import matplotlib.pyplot as plt

size_m1 = 16
size_m2 = 8

training_set_path = os.path.join('.','training_set.csv')
validation_set_path = os.path.join('.','validation_set.csv')

training_set = np.genfromtxt(training_set_path,delimiter = ',')
validation_set = np.genfromtxt(validation_set_path,delimiter = ',')

training_points = training_set[:,0:2]
validation_points = validation_set[:,0:2]

training_targets = training_set[:,-1]
validation_targets = validation_set[:,-1]

train_true = training_points[np.where(training_targets == 1)]
train_false = training_points[np.where(training_targets == -1)]

val_true = validation_points[np.where(validation_targets == 1)]
val_false = validation_points[np.where(validation_targets == -1)]

#plt.figure()
#pf = plt.scatter(train_false[:,0],train_false[:,1],c='r',marker='o')
#pt = plt.scatter(train_true[:,0],train_true[:,1],c='b',marker='o')
#plt.title("Training data distribution")
#plt.legend([pf,pt],["negative","positive"])
#plt.figure()
#pf = plt.scatter(val_false[:,0],val_false[:,1],c='m',marker='o')
#pt = plt.scatter(val_true[:,0],val_true[:,1],c='g', marker='o')
#plt.title("validation data distribution")
#plt.legend([pf,pt],["negative","positive"])
#plt.show()

net = TLPerceptron(size_m1,size_m2,0.01)

c_errors, train_energy_arr, val_energy_arr = net.train(training_set,validation_set)

print(net.weight_jk.shape)

np.savetxt(os.path.join('.','w1.csv'),net.weight_jk,delimiter=',')
np.savetxt(os.path.join('.','w2.csv'),net.weight_ij,delimiter=',')
np.savetxt(os.path.join('.','w3.csv'),net.weight_i,delimiter=',')

np.savetxt(os.path.join('.','t1.csv'),net.threshold_j,delimiter=',')
np.savetxt(os.path.join('.','t2.csv'),net.threshold_i,delimiter=',')
#np.savetxt(os.path.join('.','t3.csv'),np.array(net.threshold_o),delimiter=',')

plt.figure()
train_line, = plt.plot(np.arange(len(train_energy_arr)),train_energy_arr,'b-')
val_line, = plt.plot(np.arange(len(val_energy_arr)),val_energy_arr,'g-')
plt.legend([train_line, val_line],["Training Energy","Validation Energy"])
plt.grid(b=True)
plt.title("Energy overtime")

measure_count = 40
x = np.linspace(-1,1,measure_count);
y = np.linspace(-1,1,measure_count);

xv,yv = np.meshgrid(x,y)

measure_points = np.transpose(np.array([xv.reshape(-1),yv.reshape(-1)])) 
#print(measure_points)
results = np.array([np.sign(net.just_feed(xp)) for xp in measure_points])
pp = measure_points[np.where(results == 1)]
np = measure_points[np.where(results == -1)]

plt.figure()
plt.scatter(val_false[:,0],val_false[:,1],c="m",marker='o')
plt.scatter(val_true[:,0],val_true[:,1],c="g",marker='o')
plt.scatter(np[:,0],np[:,1],c="black",marker='+',s=10)
plt.scatter(pp[:,0],pp[:,1],c="cyan",marker='+',s=10)
plt.show()

print(net.threshold_o)