import numpy as np

ae_train = np.load("dataset/ae_train.npy")

rows, col = ae_train.shape
print(f"rows: {rows}, column: {col}")

count = 0
for i in range(rows):
    if ae_train[i,0] == 1.0:
        count += 1

print(f"total off 1: {count}")