from sklearn.datasets import fetch_openml
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

mnist = fetch_openml('mnist_784', as_frame=False)
print(mnist)
X, y= mnist.data, mnist.target

import matplotlib.pyplot as plt
import numpy as np

# def plot_digit(image_data):
#     image = image_data.reshape(28,28)
#     plt.imshow(image, cmap='binary')
#     plt.axis('off')
#
# # some_digit = X[0]
# # plot_digit(some_digit)
# # plt.show()
#
# plt.figure(figsize=(9,9))
# for idx, image_data in enumerate(X[:100]):
#     plt.subplot(10,10,idx+1)
#     plot_digit(image_data)
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()

# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {'n_neighbors':range(1,5)}
gridsearch = GridSearchCV(KNeighborsClassifier(), params, n_jobs=-1)
gridsearch.fit(x_train,y_train)
print(gridsearch.best_params_)
pred = gridsearch.predict(x_test)
print(pred)
print('accuracy :',gridsearch.score(x_test,y_test)) # 0.9712857142857143


## 강사님 코드
# knn_clf = KNeighborsClassifier()
# params = [{'n_neighbors':range(3,8,1),
#           'weights':['uniform', 'distance']}]
#
# grid_search = GridSearchCV(knn_clf, params, cv=5)
# grid_search.fit(x_train, y_train)
# print(grid_search.best_score_)
# print(grid_search.best_params_)
# print()
#
# best_model = grid_search.best_estimator_
# best_model.fit(x_train, y_train)
# print(best_model.score(x_test, y_test)) # 0.9731428571428572

