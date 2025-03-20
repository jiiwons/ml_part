import matplotlib.pyplot as plt
import mglearn

# mglearn.plots.plot_agglomerative_algorithm() # 처음엔 모든 하나를 군집 한개씩으로 구성, 군집 간 거리가 짧은 것부터 묶으면서 차례로
# plt.show()

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

iris = load_iris()
features = iris.data

scaler = StandardScaler()
features_std = scaler.fit_transform(features)
cluster = AgglomerativeClustering(n_clusters=5, linkage='complete')
model = cluster.fit(features_std)
print(model.labels_)
print()
print(cluster.fit_predict(features_std))