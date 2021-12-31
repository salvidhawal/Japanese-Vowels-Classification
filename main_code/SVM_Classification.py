import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot  as  plt

sns.set(style='darkgrid')

if __name__ == "__main__":
    # load the dataset
    X_train = np.load(f"../dataset/X_train.npy")
    Y_train = np.load(f"../dataset/y_train.npy")
    X_test = np.load(f"../dataset/X_test.npy")
    Y_test = np.load(f"../dataset/y_test.npy")

    # shuffle the training dataset, and set the seed (random_state) for reproducibility
    X, Y = shuffle(X_train, Y_train, random_state=0)

    # set Kfold cross validation split and n_repeats (the number of times the split will be performed). It guarantees
    # that the code will have different folds on each iteration.
    kfold_split = 5
    kfold_repeat = 2
    rkf = RepeatedKFold(n_splits=kfold_split, n_repeats=kfold_repeat, random_state=42)
    clf = SVC(kernel='linear', C=1, random_state=2)

    cv_results = cross_validate(clf, X, Y, cv=rkf, return_train_score=True, return_estimator=True)
    print(f"Training accuracy after kfold: {cv_results['train_score']}")
    print(f"Validation accuracy after kfold: {cv_results['test_score']}")
    rfc_fit = cv_results['estimator']  # returns every kfold models

    # test the prediction with X_test and append the score with every kfold model
    y_ax_test = []
    for svm_fit in rfc_fit:
        y_pred = svm_fit.predict(X_test)
        y_ax_test.append(accuracy_score(Y_test, y_pred))
    y_ax_test = np.array(y_ax_test)

    print(f"Test accuracy after kfold: {y_ax_test}")

    # plot the graph for training, validation and testing scores and save the figure in graphs folder
    x_ax = np.arange(1, 11, dtype=int)
    y_ax_val = cv_results['test_score']
    y_ax_train = cv_results['train_score']

    sns.lineplot(x=x_ax, y=y_ax_train, marker="o")
    sns.lineplot(x=x_ax, y=y_ax_val, marker="o")
    sns.lineplot(x=x_ax, y=y_ax_test, marker="o")
    plt.legend(["Training accuracy", "Validation accuracy", "Test accuracy"])
    plt.xlabel("Kfold Variations")
    plt.ylabel("Accuracies")
    plt.title("SVM Classification with kfold")
    plt.savefig(f'../Graphs/svm_kfold_{kfold_repeat * kfold_split}.png')
    # plt.show()
