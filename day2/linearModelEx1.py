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

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(perch_length, perch_weight, random_state=42)

x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(x_train, y_train)
print(knr.predict([[50]])) # 2차원 입력 재구성
print()

# distances, indexes = knr.kneighbors([[50]])
# print(distances, indexes)
# plt.scatter(x_train, y_train)
# plt.scatter(x_train[indexes], y_train[indexes], marker='o')
# plt.scatter(50,50,marker='^')
# plt.show()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.predict([[50]])) # [1241.83860323]
print(lr.coef_, lr.intercept_) # [39.01714496] -709.0186449535474

# plt.scatter(x_train, y_train)
# plt.plot([15,50], [15*lr.coef_ + lr.intercept_, 50*lr.coef_ + lr.intercept_])
# plt.scatter(50, 1241, marker='^')
# plt.show()

train_poly = np.column_stack((x_train ** 2, x_train))
test_poly = np.column_stack((x_test ** 2, x_test))
print(train_poly)

lr = LinearRegression() # 모델은 선형 회귀를 사용하지만, 특성이 비선형(제곱 항 포함)이기 때문에 다항 회귀 모델처럼 동작
lr.fit(train_poly, y_train)

print(lr.predict([[50**2, 50]])) # 입력 값 50을 예측할 때, 다항 특성을 사용하므로 입력을 [50**2, 50] 형태로 변환하여 전달
print(lr.coef_, lr.intercept_) # [  1.01433211 -21.55792498] 116.0502107827827 coef를 2개 뽑고 있음 (2차원에서는 x계수가 두개 필요하므로), x^2의 계수랑 x의 계수

point = np.arange(15, 50)
plt.scatter(x_train, y_train)
plt.plot(point, 1.01433211*point**2 -21.55792498*point + 116.0502107827827 )
plt.scatter(50,1574, marker='^')
plt.show()
