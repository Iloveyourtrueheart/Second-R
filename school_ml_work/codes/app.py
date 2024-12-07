import sys
from PIL import Image
import sys
import random
from ast import literal_eval
import time
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from sklearn.metrics import silhouette_score
import pandas as pd
import text_processing_sklearn
from image_processing_dbscan_python import dbscan_image_process
from sklearn.neighbors import NearestNeighbors


def process_image(image,kVal):
        if image is None or image == '':
            raise gr.Error('请上传图像！')
        if kVal is None:
            raise gr.Error("请选择k值！")
        # 将 ndarray 转换为 RGB 模式的 Image 对象
        if type(image) == type(np.array([[1,1]])):
            im = Image.fromarray(image.astype('uint8'), 'RGB')
        #im = Image.open()
        else:
            im = image
        pix = im.load()
        xVal,yVal = im.size
        pointsArray = []

        #Map pixel values to array
        for y in range(yVal):
            pointsArray.append([])
            for x in range(xVal):
                pointsArray[y].append(list(pix[x,y]))
        ori_array = pointsArray


        start = time.time()
        #Convert input to array and read kVal
        vectors = pointsArray
        kVal = int(kVal)

        #choose k initial random unequal points
        centerPoints = []
        while len(centerPoints) != kVal:
            candidate = random.choice(random.choice(vectors))
            if candidate not in centerPoints:
                centerPoints.append(candidate)

        #clustering
        testPrev = []
        iterationCounter = 0
        while True:
            clusteringTable = []
            for y in range(len(vectors)):
                clusteringTable.append([])
                for x in range(len(vectors[0])):
                    temp = []
                    temp.append(vectors[y][x])
                    for k in range(kVal):
                        #distance from center point
                        distance = 0
                        for l in range(len(centerPoints[0])):
                            distance = distance + abs(centerPoints[k][l]-vectors[y][x][l])
                        temp.append(distance)
                    #assign cluster centroid with min distance
                    temp.append(temp[1:].index(min(temp[1:])))
                    clusteringTable[y].append(temp)
            
            #check if cluster values changed, exit otherwise
            test = []
            for k in range(kVal):
                test.append(0)
                for itemY in clusteringTable:
                    for itemX in itemY:
                        if itemX[-1] == k:
                            test[k] = test[k] + 1
            if testPrev == test:
                break
            testPrev = test
            
            #update centroids
            for k in range(kVal):
                n = 0
                vectorTemps = [0]*len(clusteringTable[0][0][0])
                for itemY in clusteringTable:
                    for itemX in itemY:
                        if itemX[-1] == k:
                            for valCounter in range(len(itemX[0])):
                                vectorTemps[valCounter] = vectorTemps[valCounter] + itemX[0][valCounter]
                            n = n + 1
                #check for 0 division
                for valCounter in range(len(vectorTemps)):
                    if vectorTemps[valCounter] != 0:
                        vectorTemps[valCounter] = vectorTemps[valCounter]/n
                centerPoints[k] = vectorTemps
            iterationCounter = iterationCounter + 1

        #get labels
        labels = []
        for y in range(len(clusteringTable)):
            for x in range(len(clusteringTable[0])):
                labels.append(clusteringTable[y][x][-1])
        
        #Build clustered array
        clusteredVectors = []
        for y in range(len(clusteringTable)):
            clusteredVectors.append([])
            for x in range(len(clusteringTable[0])):
                clusteredVectors[y].append(centerPoints[clusteringTable[y][x][-1]])

        end = time.time()

        #Convert input to array and read outputName parameter
        vectors = clusteredVectors
        score = silhouette_score(np.reshape(ori_array,(xVal*yVal,-1)),np.array(labels))
        modelLen = 3
        #Check supported model and initialize image
        if modelLen == 3:
            image = Image.new('RGB', (len(vectors[0]),len(vectors)))
        elif modelLen == 4:
            image = Image.new('RGBA', (len(vectors[0]),len(vectors)))
        else:
            sys.exit(1)

        #Map array values to image and save
        pix = image.load()
        for y in range(len(vectors)):
            for x in range(len(vectors[0])):
                r = int(round(vectors[y][x][0]))
                g = int(round(vectors[y][x][1]))
                b = int(round(vectors[y][x][2]))
                if modelLen == 3:
                    pix[x,y] = (r,g,b)
                elif modelLen == 4:
                    a = int(round(vectors[y][x][3]))
                    pix[x,y] = (r,g,b,a)
        #image.save(outputName)
        return np.array(image),score
def k_draw(image_data,k):

    
    if type(image_data) == type(np.array([[1,1]])):
        im = Image.fromarray(image_data.astype('uint8'), 'RGB')
    #im = Image.open()
    else:
        im = image_data
    pix = im.load()
    xVal,yVal = im.size
    pointsArray = []

    #Map pixel values to array
    for y in range(yVal):
        pointsArray.append([])
        for x in range(xVal):
            pointsArray[y].append(list(pix[x,y]))
    data = pointsArray
    # 计算每个样本的 k 距离
    k = int(k)  # 选择一个 k 值，通常是根据问题的特定情况进行选择
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
    fig_k = plt.gcf()
    fig_k.canvas.draw()
    w, h = fig_k.canvas.get_width_height()
    img_array_k = np.frombuffer(fig_k.canvas.tostring_rgb(), dtype='uint8').reshape((h, w, 3))
    plt.clf()
    return img_array_k



def read_data(flie):
    data_csv = pd.read_csv(flie,encoding='UTF-8')
    return data_csv


clean_data,label_encod_cle = None,None
def process(filepath,k2):
    global clean_data
    global label_encod_cle
    if clean_data is not None:
        print('Global')
        label = text_processing_sklearn.get_Kmeans(clean_data,k2)
        score = text_processing_sklearn.get_score(clean_data,label)
        fig = text_processing_sklearn.plt_numbers_labels(label_encod_cle)
        fig1,fig2 = text_processing_sklearn.draw_cluster(clean_data,label_encod_cle)
        return score,fig,fig1,fig2
    else:
        data,label_encod = text_processing_sklearn.get_data(filepath)
        clean_data,label_encod_cle = data,label_encod
        label = text_processing_sklearn.get_Kmeans(data,k2)
        score = text_processing_sklearn.get_score(data,label)
        fig = text_processing_sklearn.plt_numbers_labels(label_encod)
        fig1,fig2 = text_processing_sklearn.draw_cluster(data,label_encod)
        return score,fig,fig1,fig2

def process_dbs(filepath,eps,minPts):
    global clean_data
    global label_encod_cle
    if clean_data is not None:
        label = text_processing_sklearn.get_dbscan(clean_data,eps,minPts)
        score = text_processing_sklearn.get_score(clean_data,label)
        fig = text_processing_sklearn.plt_numbers_labels(label_encod_cle)
        fig1,fig2 = text_processing_sklearn.draw_cluster(clean_data,label)
        return score,fig,fig1,fig2
    else:
        data,label_encod = text_processing_sklearn.get_data(filepath)
        clean_data,label_encod_cle = data,label_encod
        label = text_processing_sklearn.get_dbscan(data,eps,minPts)
        score = text_processing_sklearn.get_score(data,label)
        fig = text_processing_sklearn.plt_numbers_labels(label_encod)
        fig1,fig2 = text_processing_sklearn.draw_cluster(data,label)
        return score,fig,fig1,fig2


with gr.Blocks(title='郭子靖的交互界面') as demo:
    with gr.Tab("Kmeans算法处理"):
        gr.Markdown('''
                    
                    # Kmeans的图像分割


                    ''')
        with gr.Row():
                image1 = gr.Image(sources=['upload'],label="上传需要Kmeasn处理的图像")
                out_image1 = gr.Image(type='pil',label="处理时间可能需要一些时间，处理过程请等待")
        score1 = gr.Text(type='text',label="轮廓系数")
        with gr.Row():
                k = gr.Radio(choices=['2','3','4','6','8'],label='k')
                button1 = gr.Button(value='算法演示',link='https://www.naftaliharris.com/blog/visualizing-k-means-clustering/',variant='secondary')
                button2 = gr.Button(value="开始处理",variant='primary')
                gr.Examples(['test.jpg','2.jpg'],inputs=[image1])
                button2.click(process_image,inputs=[image1,k],outputs=[out_image1,score1])
        gr.Markdown('''
                    

                    # kmeans的文本处理
                    
                    
                    ''')
        with gr.Row():
            file1 = gr.File(label='Flie',type='filepath',scale=1)
            data_pd = gr.DataFrame(type='pandas',scale=4,line_breaks=5)
            button4 = gr.Button(value="读取",scale=1,variant='primary')
            button4.click(read_data,inputs=[file1],outputs=[data_pd])
        gr.Examples(['codes/chinese_news_cutted_train_utf8.csv'],inputs=file1,label="数据在这里")
        gr.Markdown('''
                    
                    
                    # DATA CLEARNING AND CLUSTER
                    
                    
                    ''')
        with gr.Row():
            with gr.Column():
                button5 = gr.Button(value="Data clearning and Cluster Start",variant='stop')
                k2 = gr.Radio(choices=['2','4','8','12','16'],label="k")
                image2 = gr.Image(label="真实样本标签情况")
                score2 = gr.Text(type='text',label="轮廓系数")
            image3 = gr.Image(label="未分类的数据可视化")
            image4 = gr.Image(label='聚类结果可视化')
            button5.click(process,inputs=[file1,k2],outputs=[score2,image2,image3,image4])

    with gr.Tab("DBSCAN算法处理"):
        gr.Markdown('''
                    
                    # DBSCAN的图像分割


                    ''')
        with gr.Row():
                image6 = gr.Image(sources=['upload'],label="上传需要DBSCAN处理的图像",scale=4)
                out_image2 = gr.Image(type='pil',label="处理时间可能需要一些时间，处理过程请等待",scale=4)
        with gr.Row():
            score3 = gr.Text(type='text',label="轮廓系数",scale=1,min_width=1)
            #k_d = gr.Radio(choices=['5','10'],label="k",scale=1,min_width=1)
            #k_out_image = gr.Image(type='numpy',label="K-distance",scale=8)
            #button10 = gr.Button(value='生成k-distance',variant='primary',scale=1,min_width=1)
            #button10.click(k_draw,inputs=[image6,k_d],outputs=[k_out_image])
        with gr.Row():
                eps = gr.Radio(choices=['0.02','0.03','100','150','10'],label='eps')
                minPts = gr.Slider(minimum=90,maximum=150,step=10,label='minPts')
                button6 = gr.Button(value='算法演示',link='https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/',variant='secondary')
                button7 = gr.Button(value="开始处理",variant='primary')
                gr.Examples(['test.jpg','2.jpg'],inputs=[image6])
                button7.click(dbscan_image_process,inputs=[image6,eps,minPts],outputs=[out_image2,score3])
        gr.Markdown('''
                    

                    # DBSCAN的文本处理
                    
                    
                    ''')
        with gr.Row():
            file2 = gr.File(label='Flie',type='filepath',scale=1)
            data_pd1 = gr.DataFrame(type='pandas',scale=4,line_breaks=5)
            button8 = gr.Button(value="读取",scale=1,variant='primary')
            button8.click(read_data,inputs=[file2],outputs=[data_pd1])
        gr.Examples(['codes/chinese_news_cutted_train_utf8.csv'],inputs=file2,label="数据在这里")
        gr.Markdown('''
                    
                    
                    # DATA CLEARNING AND CLUSTER
                    
                    
                    ''')
        with gr.Row():
            with gr.Column():
                button9 = gr.Button(value="Data clearning and Cluster Start",variant='stop')
                eps2 = gr.Slider(minimum=0.1,maximum=100,step=0.1,label='eps')
                minPts2 = gr.Slider(minimum='1',maximum='600',step='1',label='minPts')
                image5 = gr.Image(label="真实样本标签情况")
                score4 = gr.Text(type='text',label="轮廓系数")
            image6 = gr.Image(label="未分类的数据可视化")
            image7 = gr.Image(label='聚类结果可视化')
            button9.click(process_dbs,inputs=[file2,eps2,minPts2],outputs=[score4,image5,image6,image7])
demo.launch(share=True)