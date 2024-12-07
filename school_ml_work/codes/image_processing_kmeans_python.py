import sys
from PIL import Image
import sys
import random
from ast import literal_eval
import time
import gradio as gr
import numpy as np
from sklearn.metrics import silhouette_score
#Open and read img
#imageName = sys.argv[1]
class kmeans:
    def __init__(self,imageName):
        imageName = imageName
        im = Image.open(imageName)
        pix = im.load()
        xVal,yVal = im.size
        pointsArray = []

    #Map pixel values to array
        for y in range(yVal):
            pointsArray.append([])
            for x in range(xVal):
                pointsArray[y].append(list(pix[x,y]))


        start = time.time()
        #Convert input to array and read kVal
        vectors = pointsArray
        kVal = 2

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

        #Build clustered array
        clusteredVectors = []
        for y in range(len(clusteringTable)):
            clusteredVectors.append([])
            for x in range(len(clusteringTable[0])):
                clusteredVectors[y].append(centerPoints[clusteringTable[y][x][-1]])

        end = time.time()
        print(end - start)

        #Convert input to array and read outputName parameter
        vectors = clusteredVectors
        outputName = 'codes/jilei_is_cunt222222.jpg'
        modelLen = 3

        #Check supported model and initialize image
        if modelLen == 3:
            image = Image.new('RGB', (len(vectors[0]),len(vectors)))
        elif modelLen == 4:
            image = Image.new('RGBA', (len(vectors[0]),len(vectors)))
        else:
            print("Unsupported model")
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
        image_array = np.array(image)
        image.save(outputName)
def action(imageName):
    act = kmeans(imageName)
action('test.jpg')
