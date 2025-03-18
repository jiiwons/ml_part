import matplotlib.pyplot as plt
import numpy as np

perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )

# plt.scatter(perch_length, perch_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(perch_length, perch_weight, random_state=42)
print(x_train.shape, x_test.shape) # (42,) (14,)
# linear model에 입력으로 들어가는 형태는 2차원
# 현재 x데이터는 1차원이므로 [[][][][]] 이런식으로 2차원 데이터로 재구성해야됨
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
print(x_train.shape) # (42, 1)
print(x_train)

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(x_train, y_train)
print('n_neighbors=5, accuracy:', knr.score(x_test, y_test))
print()

knr2 = KNeighborsRegressor(n_neighbors=3)
knr2.fit(x_train, y_train)
print('n_neighbors=3, accuracy:', knr2.score(x_test, y_test))
print()

knr = KNeighborsRegressor()
x = np.arange(5,45).reshape(-1,1)

plt.figure(figsize=(5,12))
for idx, n in enumerate([1,5,10]):
    knr_n_neighbors = n
    knr.fit(x_train, y_train)
    prediction = knr.predict(x)
    plt.subplot(3,1,idx+1)
    plt.scatter(x_train, y_train)
    plt.plot(x, prediction)
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.title(f'n_neighbors = {n}')
plt.tight_layout()
plt.show()