from scipy.ndimage import label
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# data = load_iris()
# print(data.target)
# print(data.target_names)
#
# x = data.data
# plt.figure(figsize=(9,4))
# plt.subplot(121)
#
# plt.scatter(x[:, 2], x[:, 3], c='k', marker='.')
# plt.xlabel('Patal length', fontsize=14)
# plt.ylabel('Patal width', fontsize=14)
# # plt.show()
# # 이렇게만 봤을 때는 어떻게 군집화해야할지 모름
#
# y = data.target
# plt.subplot(122)
# plt.plot(x[y==0, 2], x[y==0, 3], 'yo', label='iris setosa')
# plt.plot(x[y==1, 2], x[y==1, 3], 'bs', label='iris versicolor')
# plt.plot(x[y==2, 2], x[y==2, 3], 'g*', label='iris viriginica')
# plt.xlabel('Patal length', fontsize=14)
# plt.legend(loc='best')
# plt.tick_params(labelleft=False)
# plt.show()
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "6"

import mglearn
mglearn.plots.plot_kmeans_algorithm()
plt.show()