#文本数据的预处理
import pandas as pd
import jieba
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def split_labels_data(data):
    label_list,data_list = [],[]
    for content in data:
        words = content.split('\t')
        label_list.append(words[0])
        data_list.append(words[1])
    return label_list,data_list

def data2array(data):
    tv=TfidfVectorizer(max_features=300)
    data_transform=tv.fit_transform(data)
    data_transform=data_transform.toarray()
    return data_transform

def label_encoder(label_list):
    enc=preprocessing.LabelEncoder()   #获取一个LabelEncoder
    label_list_encod = enc.fit_transform(label_list)
    return label_list_encod


#各个标签数量柱形图
def plt_numbers_labels(label_encod):
    counter = Counter(label_encod)
    unique_values = list(counter.keys())
    counts = list(counter.values())
    plt.bar(unique_values, counts, align='center', alpha=0.7)
    plt.xlabel('label')
    plt.ylabel('numbers')
    plt.title('numbers of labels')
    plt.grid(False)
    fig = plt.gcf()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape((h, w, 3))
    plt.clf()
    return  img_array


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def data_cleaning(content_list,stopwords):
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

# 绘制聚类结果散点图
def draw_cluster(dataset,labels):
    dataset = PCA(n_components=2).fit_transform(dataset)  
    label = np.array(labels)
    plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c='black', s=7)  # 原图
    fig1 = plt.gcf()
    fig1.canvas.draw()
    w, h = fig1.canvas.get_width_height()
    img_array1 = np.frombuffer(fig1.canvas.tostring_rgb(), dtype='uint8').reshape((h, w, 3))
    plt.clf()
    colors = np.array(
        ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#00FFFF", "#FF00FF", "#800000", "#008000", "#000080", "#808000",
         "#800080", "#008080", "#444444", "#FFD700", "#008080"])
    # 循换打印k个簇，每个簇使用不同的颜色
    for i in range(12):
        plt.scatter(dataset[np.nonzero(label == i), 0], dataset[np.nonzero(label == i), 1], c=colors[i], s=7, marker='o')
    fig2 = plt.gcf()
    fig2.canvas.draw()
    w, h = fig2.canvas.get_width_height()
    img_array2 = np.frombuffer(fig2.canvas.tostring_rgb(), dtype='uint8').reshape((h, w, 3))
    plt.clf()
    return img_array1,img_array2



def get_Kmeans(data,k):
    kmeans = KMeans(n_clusters=int(k), random_state=6,n_init=12,max_iter=600)
    kmeans.fit(data)
    labels1 = kmeans.labels_
    #centers = kmeans.cluster_centers_
    return labels1
def get_dbscan(data,eps,minPts):
    if float(eps) < 1:
        eps = float(eps)
    else:
        eps = eval(eps)#100
    minPts = int(minPts)#100
    dbscan = DBSCAN(eps=eps, min_samples=minPts)
    dbscan.fit(data)
    labels2 = dbscan.labels_
    return labels2
def get_score(data,labels):
    silhouette_values2 = silhouette_score(data,labels)
    return silhouette_values2
def get_data(filepath):
    data1 = pd.read_csv(filepath,encoding='UTF-8')
    #标签，数据分离
    label,data = split_labels_data(data1['分类\t分词文章'])
    #label encode
    label_encod = label_encoder(label)
    data1['分类\t分词文章'] = data
    stopwords = stopwordslist("stopwords.txt")
    data_cleaned = data_cleaning(data1['分类\t分词文章'],stopwords)
    data_transform = data2array(data_cleaned)
    return data_transform,label_encod
if __name__ == '__main__':
    data_transform = get_data("chinese_news_cutted_train_utf8.csv")
    #kmeans
    labels1 = get_Kmeans(data_transform)
    imag1,imag2 = draw_cluster(data_transform,labels1)
    #dbscan
    labels2 = get_dbscan(data_transform)
    image3,imag4 = draw_cluster(data_transform,labels2)
    # 计算轮廓系数
    score_list = []
    for i in [labels1,labels2]:
        values1 = get_score(data_transform, i)
        score_list.append(values1)

    print(score_list)
