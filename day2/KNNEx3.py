import matplotlib.pyplot as plt

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

import numpy as np
# print(np.column_stack(([1,2,3],[4,5,6])) # 옆으로 붙이겠다 horizontal stack과의 차이는 열을 하나씩 조합해서 붙이는 것
fish_data = np.column_stack((bream_length + smelt_length, bream_weight + smelt_weight)) # 데이터 랜덤으로 자르려고 하나로 뭉침
print(fish_data)

fish_target = np.concatenate((np.ones(35), np.zeros(14)))
print(fish_target)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(fish_data, fish_target, test_size=0.25, random_state = 42)
print(x_train.shape, x_test.shape)
print()
print(x_train)
print()

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(x_train, y_train)
print('test data acuracy:', kn.score(x_test, y_test))
predict = kn.predict(x_test)
print(predict)
print(y_test)