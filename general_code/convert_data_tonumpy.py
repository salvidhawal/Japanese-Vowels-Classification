import numpy as np
import itertools


def save_dataset_in_numpy(X, blockLengthes):
    X_rows, X_col = X.shape
    y = []
    class_counter = 1
    increment_counter = 0
    j = 0
    for i in range(X_rows):
        if X[i][0] == 1.0 and X[i][11] == 1.0:
            y.append(0)
            increment_counter += 1
        else:
            y.append(class_counter)

        if increment_counter == blockLengthes[j]:
            increment_counter = 0
            class_counter += 1
            j += 1

    X = np.delete(X, np.where((X == 1.0))[0], axis=0)
    print(X.shape)

    y = np.array(y)
    y = np.delete(y, np.where((y == 0.0))[0], axis=0)
    print(y.shape)
    return X, y


filename = ["ae.train", "ae.test"]
savename = ["ae_train", "ae_test"]

for file, savefile in itertools.zip_longest(filename, savename):
    File_data = np.loadtxt(f"../dataset/{file}", dtype=float)
    print(File_data.shape)
    np.save(f"../dataset/{savefile}", File_data)

ae_train = np.load("../dataset/ae_train.npy")
blockLengthes = [30, 30, 30, 30, 30, 30, 30, 30, 30]
X_train, y_train = save_dataset_in_numpy(X=ae_train, blockLengthes=blockLengthes)
np.save(f"../dataset/X_train", X_train)
np.save(f"../dataset/y_train", y_train)

ae_test = np.load("../dataset/ae_test.npy")
blockLengthes_test = [31, 35, 88, 44, 29, 24, 40, 50, 29]
X_test, y_test = save_dataset_in_numpy(X=ae_test, blockLengthes=blockLengthes_test)
np.save(f"../dataset/X_test", X_test)
np.save(f"../dataset/y_test", y_test)
