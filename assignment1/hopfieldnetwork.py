import numpy as np

class HopField(object):
    def __init__(self,patterns,bitts):
        pat = []
        for n in range(patterns):
            p = np.random.rand(bitts,1)
            p[np.where(p >= 0.5)]=1
            p[np.where(p != 1)]=-1
            pat.append(p)
        
        self.patterns = pat
        w = np.zeros((bitts,bitts))
        for p in pat:
            w = w + np.multiply((1/bitts),np.multiply(p,p.transpose()))

        for i in range(bitts):
            w[i,i] = 0

        self.weights = w

