import numpy as np
from perceptron import TLPerceptron

net = TLPerceptron(4,5)

print(net.weight_jk)
print(net.weight_ij)
print(net.weight_i)

input = np.random.randint(2,size=2) * 2 - 1
output,states = net.feed(input)

print("output of %s from the network is %s" % (input,output))
print("neuron states:")
print(states)