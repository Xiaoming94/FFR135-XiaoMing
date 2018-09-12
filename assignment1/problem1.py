import numpy as np
from hopfieldnetwork import HopField

patterns = [12,20,40,60,80,100]
trials = 10 ** 5

net = HopField(5,10)
patterns = net.patterns

iPat = np.random.randint(low=0,high=9)
print(iPat)
pattern = patterns[iPat]

print(pattern)

out = net.feedAsync(pattern[:,0],iPat)
print(out)