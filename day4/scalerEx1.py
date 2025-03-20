from sklearn.preprocessing import StandardScaler
import numpy as np

features = np.array([[-500.5],
                     [-100.1],
                     [0],
                     [900.9]])
ss_scaler = StandardScaler()
scaled_feature1 = ss_scaler.fit_transform(features)
print(scaled_feature1)
print()

from sklearn.preprocessing import RobustScaler
r_scaler = RobustScaler()
scaled_feature2 = r_scaler.fit_transform(features)
print(scaled_feature2)
print()

from sklearn.preprocessing import MinMaxScaler
m_scaler = MinMaxScaler()
scaled_feature3 = m_scaler.fit_transform(features)
print(scaled_feature3)
print()

# 위 3개의 스케일러들은 1차원 값에 대해서(각각의 값에 대해서)
# normalize는 2차원 데이터에?(normalize는 벡터값에 대해서..? 방향값?)
from sklearn.preprocessing import Normalizer
features = np.array([[0.5, 0.5],
                     [1.1, 3.4],
                     [1.5, 20.2],
                     [1.63, 34.4],
                     [10.9, 3.3]])
n_scaler = Normalizer(norm='l1')
scaled_feature4 = n_scaler.fit_transform(features)
print(scaled_feature4)