from matplotlib.image import imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


image = imread('ladybug.png')
print(image.shape) # (533, 800, 3)

x = image.reshape(-1, 3) # 직렬화
print(x.shape) # (426400, 3)
kmeans = KMeans(n_clusters=8, random_state=42).fit(x) # 군집 8 - 8가지의 대표 색상
segmented_img = kmeans.cluster_centers_[kmeans.labels_] # 레이블에 속한 군집의 중앙값을 가져옴(레드계열이면 딱 하나의 레드로 갖고오고, 초록계열(옅은 초록, 초록...)이면 딱 하나의 초록색으로)
segmented_img = segmented_img.reshape(image.shape)

segmented_imgs = []
n_colors = [10,8,6,4,2]
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(x)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))


plt.figure(figsize=(8,4))
plt.subplot(231)
plt.imshow(image)
plt.title('Original image')
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx])
    plt.title(f'{n_clusters} colors')
    plt.axis('off')

plt.show()