import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    ae_train = np.load("../dataset/ae_train.npy")
    ae_test = np.load("../dataset/ae_test.npy")
    print(ae_train.shape)
    print(ae_test.shape)
    ae_train_rows, ae_train_col = ae_train.shape
    ae_test_rows, ae_test_col = ae_test.shape

    X_train = np.delete(ae_train, np.where((ae_train == 1.0))[0], axis=0)
    print(X_train.shape)
    X_test = np.delete(ae_test, np.where((ae_test == 1.0))[0], axis=0)
    print(X_test.shape)

    y_train = []
    class_counter = 1
    increment_counter = 0
    for i in range(ae_train_rows):
        if ae_train[i][0] == 1.0 and ae_train[i][11] == 1.0:
            y_train.append(0)
            increment_counter += 1
        else:
            y_train.append(class_counter)

        if increment_counter == 30:
            increment_counter = 0
            class_counter += 1

    blockLengthes = [31, 35, 88, 44, 29, 24, 40, 50, 29]
    y_test = []
    class_counter = 1
    increment_counter = 0
    j = 0
    for i in range(ae_test_rows):
        if ae_test[i][0] == 1.0 and ae_test[i][11] == 1.0:
            y_test.append(0)
            increment_counter += 1
        else:
            y_test.append(class_counter)

        if increment_counter == blockLengthes[j]:
            print(blockLengthes[j])
            increment_counter = 0
            class_counter += 1
            j = j+1

    y_train = np.array(y_train)
    y_train = np.delete(y_train, np.where((y_train == 0.0))[0], axis=0)
    print(y_train.shape)

    y_test = np.array(y_test)
    y_test = np.delete(y_test, np.where((y_test == 0.0))[0], axis=0)
    print(y_test.shape)

    X, y = shuffle(X_train, y_train)
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)

    y_pred_val = clf.predict(x_val)
    print(accuracy_score(y_val, y_pred_val))

    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))



