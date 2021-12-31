import numpy as np
import itertools


# The save_dataset_in_numpy function is used to assign labels for classification and to remove separators from
# dataset points.
def save_dataset_in_numpy(X, blockLengthes):
    X_rows, X_col = X.shape
    y = []
    class_counter = 1 # class labels
    block_counter = 0
    j = 0
    for i in range(X_rows):
        # 1.0 marks the block separator for a certain person in the dataset; if the code reaches 1.0, append 0 to y
        # and add 1 to block counter; otherwise, append the class counter to y.
        if X[i][0] == 1.0 and X[i][11] == 1.0:
            y.append(0)
            block_counter += 1
        else:
            y.append(class_counter)

        # If block counter reaches the specified blockLengths, increment the class counter for the new label and
        # reset block counter to 0 for the next block.
        if block_counter == blockLengthes[j]:
            block_counter = 0
            class_counter += 1
            j += 1

    # After you've obtained the labels, remove the sperator from X and y.
    X = np.delete(X, np.where((X == 1.0))[0], axis=0)
    print(f"Numpy Shape of X (datapoints): {X.shape}")

    y = np.array(y)
    y = np.delete(y, np.where((y == 0.0))[0], axis=0)
    print(f"Numpy Shape of y (labels): {y.shape}")
    return X, y


filename = ["ae.train", "ae.test"]
savename = ["ae_train", "ae_test"]

# save the "ae.train", "ae.test" in the form of numpy
for file, savefile in itertools.zip_longest(filename, savename):
    File_data = np.loadtxt(f"../dataset/{file}", dtype=float)
    np.save(f"../dataset/{savefile}", File_data)

# Save the above-converted numpy files to training and testing datasets with labels and without the seperator (1.0
# for datapoints and 0.0 for labels)
ae_train = np.load("../dataset/ae_train.npy")
blockLengthes = [30, 30, 30, 30, 30, 30, 30, 30, 30]
print(f"Assign labels for training dataset")
X_train, y_train = save_dataset_in_numpy(X=ae_train, blockLengthes=blockLengthes)
np.save(f"../dataset/X_train", X_train)
np.save(f"../dataset/y_train", y_train)

print("-------------------------------------------------------------------------")

ae_test = np.load("../dataset/ae_test.npy")
blockLengthes_test = [31, 35, 88, 44, 29, 24, 40, 50, 29]
print(f"Assign labels for testing dataset")
X_test, y_test = save_dataset_in_numpy(X=ae_test, blockLengthes=blockLengthes_test)
np.save(f"../dataset/X_test", X_test)
np.save(f"../dataset/y_test", y_test)
