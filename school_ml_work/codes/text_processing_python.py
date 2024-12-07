import pandas as pd
import jieba
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
class Cleaning:
    def split_labels_data(self,data):
        for content in data:
            words = content.split('\t')
            label_list.append(words[0])
            data_list.append(words[1])
        return label_list,data_list

    def stopwordslist(self,filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return stopwords


    def label_encoder(self,label_list):
        enc=preprocessing.LabelEncoder()   #获取一个LabelEncoder
        label_list_encod = enc.fit_transform(label_list)
        return label_list_encod


    def data_cleaning(self,content_list):
        content_seg=[]
        symbols = '-\\n～%≥℃|/【】↓#~_「♂!？\'，、:；。《》()（）·—.…,0123456789abcdefghijklnmopqrstuvwxyz'
        for content in content_list:
            for con in content:
                if con in symbols:
                    content=content.replace(con,' ')
            con_list=jieba.cut(content,cut_all=False)
            result_list=[]
            for con in con_list:
                if con not in stopwords and con!='\n' and con != '\u3000'and con != ' ':
                    result_list.append(con)
            str1=' '.join(result_list)
            content_seg.append(str1)
        return content_seg
'''
class K_means:
    def initialize_centroids(self,data, k):
        # 从数据集中随机选择k个点作为初始质心
        centers = data[np.random.choice(data.shape[0], k, replace=False)]
        return centers


    def get_clusters(self,data, centroids):
        # 计算数据点与质心之间的距离，并将数据点分配给最近的质心
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        cluster_labels = np.argmin(distances, axis=1)
        return cluster_labels


    def update_centroids(self,data, cluster_labels, k):
        # 计算每个簇的新质心，即簇内数据点的均值
        new_centroids = np.array([data[cluster_labels == i].mean(axis=0) for i in range(k)])
        return new_centroids


    def k_means(self,data, k, T, epsilon):
        start = time.time()  # 开始时间，计时
        # 初始化质心
        centroids = self.initialize_centroids(data, k)
        t = 0
        while t <= T:
            # 分配簇
            cluster_labels = self.get_clusters(data, centroids)

            # 更新质心
            new_centroids = self.update_centroids(data, cluster_labels, k)

            # 检查收敛条件
            if np.linalg.norm(new_centroids - centroids) < epsilon:
                break
            centroids = new_centroids
            print("第", t, "次迭代")
            t += 1
        print("用时：{0}".format(time.time() - start))
        return cluster_labels, centroids


    # 计算聚类指标
    def clustering_indicators(self,labels_true, labels_pred):
        #if type(labels_true[0]) != int:
            #labels_true = LabelEncoder().fit_transform(df[columns[len(columns) - 1]])  # 如果数据集的标签为文本类型，把文本标签转换为数字标签
        f_measure = f1_score(labels_true, labels_pred, average='macro')  # F值
        accuracy = accuracy_score(labels_true, labels_pred)  # ACC
        normalized_mutual_information = normalized_mutual_info_score(labels_true, labels_pred)  # NMI
        rand_index = rand_score(labels_true, labels_pred)  # RI
        ARI = adjusted_rand_score(labels_true, labels_pred)
        return f_measure, accuracy, normalized_mutual_information, rand_index, ARI
    #轮廓系数
    def get_sil_score(self,data,labels):
        score = silhouette_score(data,labels)
        return score

    # 绘制聚类结果散点图
    def draw_cluster(self,dataset, centers, labels):
        center_array = array(centers)
        if attributes > 2:
            dataset = PCA(n_components=2).fit_transform(dataset)  # 如果属性数量大于2，降维
            center_array = PCA(n_components=2).fit_transform(center_array)  # 如果属性数量大于2，降维
        else:
            dataset = array(dataset)
        # 做散点图
        label = array(labels)
        plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c='black', s=7)  # 原图
        # plt.show()
        colors = np.array(
            ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#00FFFF", "#FF00FF", "#800000", "#008000", "#000080", "#808000",
            "#800080", "#008080", "#444444", "#FFD700", "#008080"])
        # 循换打印k个簇，每个簇使用不同的颜色
        for i in range(k):
            plt.scatter(dataset[nonzero(label == i), 0], dataset[nonzero(label == i), 1], c=colors[i], s=7, marker='o')
        # plt.scatter(center_array[:, 0], center_array[:, 1], marker='x', color='m', s=30)  # 聚类中心
        plt.show()

'''



class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps  # 邻域半径
        self.min_samples = min_samples  # 最小样本数
        self.labels_ = None  # 簇标签
        self.core_samples_ = None  # 核心点索引

    def fit_predict(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.full(shape=n_samples, fill_value=-1, dtype=int)  # 初始化所有点为噪声点

        core_samples = []
        visited = np.zeros(shape=n_samples, dtype=bool)

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            # 寻找点i的邻域
            neighbors = self._find_neighbors(X, i)

            if len(neighbors) >= self.min_samples:
                # i是核心点
                core_samples.append(i)
                self.labels_[i] = len(core_samples) - 1  # 核心点簇标签从0开始

                # 扩展簇
                self._expand_cluster(X, neighbors, core_samples, visited)

        self.core_samples_ = np.asarray(core_samples)
        return self.labels_

    def _find_neighbors(self, X, i):
        distances = euclidean_distances(X[i].reshape(1, -1), X)
        neighbors = np.where(distances <= self.eps)[1]
        return neighbors

    def _expand_cluster(self, X, neighbors, core_samples, visited):
        queue = list(neighbors)

        while queue:
            j = queue.pop(0)

            if not visited[j]:
                visited[j] = True
                self.labels_[j] = self.labels_[core_samples[-1]]  # 分配与核心点相同的簇标签

                neighbors_j = self._find_neighbors(X, j)

                if len(neighbors_j) >= self.min_samples:
                    queue.extend(neighbors_j)
                    if j not in core_samples:
                        core_samples.append(j)
    def draw_cluster(self,dataset,labels):
            dataset = PCA(n_components=2).fit_transform(dataset)  # 如果属性数量大于2，降维
            # 做散点图
            label = np.array(labels)
            plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c='black', s=7)  # 原图
            # plt.show()
            colors = np.array(
                ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#00FFFF", "#FF00FF", "#800000", "#008000", "#000080", "#808000",
                "#800080", "#008080", "#444444", "#FFD700", "#008080"])
            for i in range(len(set(labels))):
                plt.scatter(dataset[np.nonzero(label == i), 0], dataset[np.nonzero(label == i), 1], c=colors[i], s=7, marker='o')
            plt.show()
    def get_sil_score(self,data,labels):
        score = silhouette_score(data,labels)
        return score
if __name__ == "__main__":
    data1 = pd.read_csv("chinese_news_cutted_train_utf8.csv",encoding='UTF-8')
    words = data1['分类\t分词文章'][0].split('\t')
    label_list = []
    data_list = []
    cleaning = Cleaning()
    label_list,data_lsit = cleaning.split_labels_data(data1['分类\t分词文章'])
    label_encod = cleaning.label_encoder(label_list)
    stopwords = cleaning.stopwordslist("stopwords.txt")
    data_cleaned = cleaning.data_cleaning(data1['分类\t分词文章'])
    data1['data_cleaned'] = data_cleaned
    tv=TfidfVectorizer(max_features=50)
    data_transform=tv.fit_transform(data1['data_cleaned'])
    data_transform=data_transform.toarray()
    dataset = data_transform
    #attributes = len(df.columns) - 1  # 属性数量（数据集维度）
    attributes = 50
    #original_labels = list(df[columns[-1]])  # 原始标签
    original_labels = label_encod


    '''
    #kmeans
    KMEANS = K_means()
    k = 12  # 聚类簇数
    T = 100  # 最大迭代数
    n = len(dataset)  # 样本数
    epsilon = 1e-5
    labels, centers = KMEANS.k_means(np.array(dataset), k, T, epsilon) 
    F_measure, ACC, NMI, RI, ARI = KMEANS.clustering_indicators(original_labels, labels)  # 计算聚类指标
    sil_score = silhouette_score(np.array(dataset),labels)
    print("F_measure:", F_measure, "ACC:", ACC, "NMI", NMI, "RI", RI, "ARI", ARI ,'轮廓系数', sil_score)
    KMEANS.draw_cluster(dataset, centers, labels=labels)
    '''

    #dbscan
    dbscan = DBSCAN(eps=100, min_samples=2)
    labels = dbscan.fit_predict(np.array(dataset))
    dbscan.draw_cluster(np.array(dataset),labels)
    score = dbscan.get_sil_score(np.array(dataset),labels)