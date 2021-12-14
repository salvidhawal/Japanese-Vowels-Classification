import numpy as np
import itertools

filename = ["ae.train", "ae.test"]
savename = ["ae_train", "ae_test"]

for file, savefile in itertools.zip_longest(filename, savename):
    File_data = np.loadtxt(f"dataset/{file}", dtype=float)
    print(File_data.shape)
    np.save(f"dataset/{savefile}", File_data)
