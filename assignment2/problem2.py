# problem2.py
# Author: Henry Yang (940503-1056)

import numpy as np
import os

input_data = np.genfromtxt(os.path.join('.','input_data_numeric.csv'),delimiter=',')
print(input_data)

targets_mat = np.array([[-1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1]
         ,[1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1]
         ,[1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1]
         ,[1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1]
         ,[1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1]
         ,[-1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1]
         ])

for targets in targets_mat:
    print(targets)