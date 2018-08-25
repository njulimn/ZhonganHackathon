from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import RepeatedKFold
import time
import numpy as np


if __name__ == '__main__':

    data = np.genfromtxt('policy_customer_train.csv', delimiter=',')

    X = data[:, :-1]
    Y = data[:, -1]

    print(len(X))
    print(len(Y))

    kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=int(time.time()))
    clf = SVC()

    score = 0
    for train_index, test_index in kf.split(X):
        # print('train_index', train_index, 'test_index', test_index)
        X_train = X[train_index]
        X_validate = X[test_index]
        Y_train = Y[train_index]
        Y_validate = Y[test_index]

        clf = clf.fit(X_train, Y_train)
        score_this_time = clf.score(X_validate, Y_validate)
        print(score_this_time)

        if score_this_time > score:
            score = score_this_time
            print('Save Model...')
            joblib.dump(clf, 'SVM.m')
