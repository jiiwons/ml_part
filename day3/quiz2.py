from random import random

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('https://raw.githubusercontent.com/wikibook/machine-learning/2.0/data/csv/basketball_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/wikibook/machine-learning/2.0/data/csv/basketball_test.csv')
train.info()
print(train)
# 3P, BLK를 x로 Pos를 y로
# svm(svc라이브러리)
# c[0.01, 0.1, 1, 10, 100]랑 감마[0.0001, 0.001, 0.01, 0.1] 값 찾고
# 커널은 rbf로

# X = train[['3P', 'BLK']].to_numpy()
# y = train['Pos'].to_numpy()
#
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
#
# svm_model = SVC()
# svm_model.fit(x_train, y_train)
#
# y_pred = svm_model.predict(x_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")
#
# from sklearn.model_selection import GridSearchCV
#
# params = {'C': [0.01, 0.1, 1, 10, 100],
#           'gamma': [0.0001, 0.001, 0.01, 0.1]}
#
# grid_search = GridSearchCV(estimator=svm_model,
#                            param_grid=params,
#                            n_jobs = -1)
# grid_result = grid_search.fit(x_train, y_train)
#
# best_params = grid_result.best_params_
# best_score = grid_result.best_score_
# print(f"Best: {best_score} using {best_params}" )


### 강사님 코드
x_train = train[['3P', 'BLK']]
y_train = train['Pos']
x_test = test[['3P', 'BLK']]
y_test = test[['Pos']]

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def svc_param_selection(x,y):
    svm_parameters = [
        {'kernel':['rbf'],
         'gamma':[0.00001, 0.0001, 0.001, 0.1, 1],
         'C':[0.01, 0.1, 1, 10, 100]}
    ]

    clf = GridSearchCV(SVC(), svm_parameters, n_jobs=-1)
    clf.fit(x, y.values.ravel())
    print(clf.best_params_)

    return clf

clf = svc_param_selection(x_train,y_train)
y_pred = clf.predict(x_test)
print('accuracy:', accuracy_score(y_test, y_pred))

comparison = pd.DataFrame({'prediction':y_pred, 'truth':y_test.values.ravel()})
print(comparison)