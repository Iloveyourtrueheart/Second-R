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
            #data.append([x,y,z])
            data.append([x/256.0,y/256.0,z/256.0])

    return  data,m,n
def loadData2(filename):
    im = image.open(filename)
    pix = im.load()
    xVal,yVal = im.size
    pointsArray = []

    #Map pixel values to array
    for y in range(yVal):
        pointsArray.append([])
        for x in range(xVal):
            pointsArray[y].append(list(pix[x,y]))
    return pointsArray
def get_score(data,labels):
    silhouette_values2 = silhouette_score(data,labels)
    return silhouette_values2

def K_means(data,n_c,n_in):
    data_array = np.array(data)
    print(data_array.shape)
    label1 = KMeans(n_clusters=n_c,n_init=n_in).fit_predict(data_array)
    label1.reshape(-1,1)
    score = get_score(data_array,label1)
    print(score)
    return label1
def DBscan(data,eps,minp):
    data_array = np.array(data)
    label2 = DBSCAN(eps=eps,min_samples=minp).fit_predict(data_array)
    label2.reshape(-1,1)
    print(set(label2))
    score = get_score(data_array,label2)
    print(score)
    return label2

def hex_to_rgb(hex_color):
    # 去掉十六进制颜色代码中的 # 符号
    hex_color = hex_color.lstrip('#')
    
    # 解析十六进制颜色代码的各个分量
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    elif len(hex_color) == 8:  # 如果有透明度分量
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = int(hex_color[6:8], 16)
        return (r, g, b, a)
    else:
        raise ValueError("Invalid hex color code")
    
def save_image(label, row, col, model):
    labelo = label.reshape([row, col])
    pic_new = image.new("RGB", (row, col))
    
    # 设置固定颜色，比如黑色 RGB(0, 0, 0)
    noise_color = (0, 0, 0)
    colors = np.array(
                ["#FFFFFF", "#0000FF", "#FCE6C9", "#FFFF00", "#00FFFF", "#FF00FF", "#800000", "#008000", "#000080", "#808000"])
    print(colors)
    for i in range(row):
        for j in range(col):
            if labelo[i][j] == -1:
                pic_new.putpixel((i, j), noise_color)
            else:
                # 如果不是噪声点，可以设置为其他颜色，根据具体的聚类结果设定
                # 这里的 labelo[i][j] + 1 是根据你原来的逻辑
                colors0 = hex_to_rgb(colors[labelo[i][j]])
                pic_new.putpixel((i, j), colors0)

    pic_new.save(model,"JPEG")

if __name__ == '__main__':

    imgData,row,col = loadData('test.jpg')
    label1 = K_means(imgData,2,10)
    model1 = 'kmeans.jpg'
    save_image(label1,row,col,model1)
    #label2 = DBscan(imgData,0.025,90)
    #model2 = 'dbscan.jpg'
    #save_image(label2,row,col,model2)