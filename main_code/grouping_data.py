import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, cross_validate, RepeatedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    X_train = np.load(f"../dataset/X_train.npy")
    Y_train = np.load(f"../dataset/y_train.npy")
    X_test = np.load(f"../dataset/X_test.npy")
    Y_test = np.load(f"../dataset/y_test.npy")

    X, Y = shuffle(X_train, Y_train, random_state=0)
    kf = KFold(n_splits=5, shuffle=False)
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
    clf = SVC(kernel='linear', C=1, random_state=2)

    cv_results = cross_validate(clf, X, Y, cv=rkf, return_estimator=True)
    print(f"validation accuracy after kfold: {cv_results['test_score']}")
    max_accuracy_index = np.argmax(cv_results["test_score"])
    rfc_fit = cv_results['estimator']

    y_pred = rfc_fit[max_accuracy_index].predict(X_test)
    print(f"test dataset accuracy with best svm model: {accuracy_score(Y_test, y_pred)}")
