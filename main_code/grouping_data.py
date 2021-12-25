import os
import numpy as np
import math

if __name__ == "__main__":
    ae_train = np.load("../dataset/ae_train.npy")
    for i in range(270):
        print(math.ceil(i/30))
