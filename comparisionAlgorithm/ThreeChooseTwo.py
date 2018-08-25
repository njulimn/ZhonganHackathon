from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.externals import joblib
from sklearn.model_selection import RepeatedKFold
import time
import numpy as np
from sklearn.metrics import f1_score


if __name__ == '__main__':

    data = np.genfromtxt('10.csv', delimiter=',')

    X = data[:, :-1]
    Y = data[:, -1]

    kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=int(time.time()))
    clf1 = SVC()  # max_features
    clf2 = RandomForestClassifier()
    clf3 = XGBClassifier()

    score = 0
    all_score = 0
    count = 0

    for train_index, test_index in kf.split(X):
        # print('train_index', train_index, 'test_index', test_index)
        X_train = X[train_index]
        X_validate = X[test_index]
        Y_train = Y[train_index]
        Y_validate = Y[test_index]

        clf1.fit(X_train, Y_train)
        clf2.fit(X_train, Y_train)
        clf3.fit(X_train, Y_train)

        Y_guess_1 = clf1.predict(X_validate)
        Y_guess_2 = clf2.predict(X_validate)
        Y_guess_3 = clf3.predict(X_validate)

        result = Y_guess_1+Y_guess_2+Y_guess_3
        result_bool = result>1.5

        result = result_bool.astype('int')

        score_this_time = f1_score(Y_validate, result)

        print(score_this_time)

        count += 1
        all_score += score_this_time
        print('Average score is {}'.format(all_score / count))

        if score_this_time > score:
            score = score_this_time
            print('Save Model...')
            # joblib.dump(clf, 'RandomForest.m')
