# Author: Henry Yang

import numpy as np
from stochastichopfield import StochasticHopfield

net = StochasticHopfield(patterns=10,bitts=100)
print(net.weights)



