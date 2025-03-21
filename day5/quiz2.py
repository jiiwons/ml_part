import pandas as pd

passengers = pd.read_csv('titanic_data.csv')
passengers.info()
print(passengers['Pclass'])
print()

dummies = pd.get_dummies(passengers['Pclass'])
print(dummies)
del passengers['Pclass']
passengers = pd.concat([passengers, dummies], axis=1, join='inner')
passengers.info()
print()

# Cabin은 삭제, Age 결측치 처리 - 평균으로 채우기
del passengers['Cabin']
passengers['Age'] = passengers['Age'].fillna(passengers['Age'].mean()) # passengers['Age'] = 안하고 inplace=True해도 되는데 경고남
passengers.info()
print()

# sex - male은 0, female은 1
passengers['Sex'] = passengers['Sex'].map({'male':0, 'female':1}) # replace써도 되는데 경고남
print(passengers['Sex'].value_counts())
print()

# 1은 FirstClass, 2는 SecondClass, 3은 EtcClass로 변경
passengers.rename(columns = {1:'FirstClass', 2:'SecondClass', 3:'EtcClass'}, inplace=True)
passengers.info()
print()

# sex, age, firstclass, secondclass, etclass를 이용해 survived 예측하는 모델
# Logistic Regression
# train, test = 8:2
# 특성 데이터(x_train, x_test)는 standScaler를 이용해 스케일
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
X = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass', 'EtcClass']]
y = passengers['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

lr = LogisticRegression()
lr.fit(x_train_scaled, y_train)
# print(lr.score(x_test_scaled, y_test))
print('train data accuracy:', lr.score(x_train_scaled, y_train))
print('test data accuracy:', lr.score(x_test_scaled, y_test))
print()

# 학습 완료 후
# kim = np.array([0.0, 20.0, 0.0, 0.0, 1.0])
# hyo = np.array([1.0, 17.0, 1.0, 0.0, 0.0])
# choi = np.array([0.0, 32.0, 0.0, 1.0, 0.0])
# 생존 여부 예측
import numpy as np
kim = np.array([0.0, 20.0, 0.0, 0.0, 1.0])
hyo = np.array([1.0, 17.0, 1.0, 0.0, 0.0])
choi = np.array([0.0, 32.0, 0.0, 1.0, 0.0])

# arr = np.stack(
#     (kim, hyo, choi), axis=0
# )
# print(arr)
# print()
#
# y_pred = lr.predict(arr)
# print(y_pred)

sample_passengers = np.array([kim, hyo, choi])
sample_passengers_scaled = scaler.transform(sample_passengers)
survive_predict = lr.predict(sample_passengers_scaled)
for name, survive in zip(['kim', 'hyo', 'choi'], survive_predict):
    print(f'{name} 성공 예측 : {survive}')