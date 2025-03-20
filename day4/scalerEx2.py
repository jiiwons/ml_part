from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

x, _ = make_blobs(n_samples=50, centers=5, random_state=42, cluster_std=2)
x_train, x_test = train_test_split(x, random_state=42, test_size=0.1)

fig, axes = plt.subplots(1,3, figsize=(10,3))
axes[0].scatter(x_train[:,0], x_train[:,1], c='orange', label='train data', s=60)
axes[0].scatter(x_test[:,0], x_test[:,1], c='blue', label='test data', s=60)
axes[0].legend(loc='upper left')
axes[0].set_title('real data')

# train과 test를 모두 같은 스케일러로 작업? 또는 각각의 스케일러?
# 같은 스케일러로
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

axes[1].scatter(x_train_scaled[:,0], x_train_scaled[:,1], c='orange', label='scaled train data', s=60)
axes[1].scatter(x_test_scaled[:,0], x_test_scaled[:,1], c='blue', label='scaled test data', s=60)
axes[1].legend(loc='upper left')
axes[1].set_title('scaled data')

# train은 train대로, test는 test대로 스케일러로(잘못된 방법)
test_scaler = MinMaxScaler()
test_scaler.fit(x_test)
x_test_scaled_badly =test_scaler.transform(x_test)

axes[2].scatter(x_train_scaled[:,0], x_train_scaled[:,1], c='orange', label='scaled train data', s=60)
axes[2].scatter(x_test_scaled_badly[:,0], x_test_scaled_badly[:,1], c='blue', label='scaled test badly data', s=60)
axes[2].legend(loc='upper left')
axes[2].set_title('scaled train & test data') # 값의 패턴을 따르지 않음(잘못된 방법), test데이터만의 평균값을 따르게 되므로

plt.show()
# 스케일은 값의 평균값에 따라 구해지므로 train데이터를 기준으로 맞춰야함. 즉, 독립적으로 말고, 같은 스케일러를 사용해야함