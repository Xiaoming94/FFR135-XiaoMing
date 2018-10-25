import numpy as np
from mlxtend.data import loadlocal_mnist
import os

###
# Function that loads the MNIST data-set
###
def load_data(path=os.path.join("."),type="train"):
    filenames = {
        'train' : ('train-images-idx3-ubyte','train-labels-idx1-ubyte'),
        'test'   : ('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte')
    }

    data_file, labels_file = filenames[type]

    data, labels = loadlocal_mnist(
        images_path=os.path.join(path,data_file),
        labels_path=os.path.join(path,labels_file)
    )

    targets = np.zeros((np.size(labels),10))

    for i in range(np.size(labels)):
        targets[i,labels[i]] = 1

    print(targets)

    return data,targets