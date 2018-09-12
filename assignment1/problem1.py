import numpy as np
from hopfieldnetwork import HopField

patterns = [12,20,40,60,80,100]
trials = 10 ** 5
bitts = 100
p_errors_list = []

for p in patterns :
    errors = 0
    for i in range(trials):
        net = HopField(p,bitts)
        n_i = np.random.randint(bitts)
        p_i = np.random.randint(p)
        inputPattern = net.patterns[p_i]
        neuronState = net.feedAsync(inputPattern[:,0],n_i)
        if(neuronState != inputPattern[n_i,0]):
            errors += 1
    
    p_error = errors/trials
    print("p_error : %s , for %s patterns" % (p_error,p))

    p_errors_list.append(p_error)


        
