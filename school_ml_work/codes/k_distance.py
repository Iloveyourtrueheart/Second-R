import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def loadData(filePath):
    data = []
    img = image.open(filePath)
    m,n = img.size
    for i in range(m):
        for j in range(n):
            x,y,z = img.getpixel((i,j))
            data.append([x,y,z])

    return  data,m,n

imgData,row,col = loadData('test.jpg')
print(imgData)
#eps的选取代码部分
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


data = imgData

# 计算每个样本的 k 距离
k = 6  # 选择一个 k 值，通常是根据问题的特定情况进行选择
nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(data)
distances, indices = nbrs.kneighbors(data)
k_distances = distances[:, -1]

# 对 k 距离进行排序
k_distances_sorted = np.sort(k_distances)

# 绘制 k 距离图
plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(data)), k_distances_sorted, marker='o')
plt.title('k-distance plot')
plt.xlabel('样本索引')
plt.ylabel(f'{k}-距离')
plt.grid(True)
plt.show()