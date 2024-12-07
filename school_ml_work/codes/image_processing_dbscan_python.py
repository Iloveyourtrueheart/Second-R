
from PIL import Image
import numpy as np
import sys
import math
from ast import literal_eval
import time
from sklearn.metrics import silhouette_score
def read_image(filepath):
    imageName = filepath
    im = Image.open(imageName)
    pix = im.load()
    xVal,yVal = im.size

    pointsArray = []

    #Map pixel values to array
    for y in range(yVal):
        pointsArray.append([])
        for x in range(xVal):
            pointsArray[y].append(list(pix[x,y]))
    return pointsArray,xVal,yVal

#Distance functions
def EuclideanDistance(P,Q):
    intermediateValues = []
    for i in range(len(P[2])):
        intermediateValues.append(math.pow(Q[2][i]-P[2][i],2))
    return math.sqrt(sum(intermediateValues))

def MaximumDistance(P,Q):
    intermediateValues = []
    for i in range(len(P[2])):
        intermediateValues.append(abs(Q[2][i]-P[2][i]))
    return max(intermediateValues)

#Finds all neighbor points for a chosen point
def FindNeighbours(Point, Points, distanceFunction, eps):
    tempNeighbours = []
    for y in range(len(Points)):
        for x in range(len(Points[0])):
            if distanceFunction == "e":
                if EuclideanDistance(Point, Points[y][x]) <= eps:
                    tempNeighbours.append(Points[y][x])
            if distanceFunction == "m":
                if MaximumDistance(Point, Points[y][x]) <= eps:
                    tempNeighbours.append(Points[y][x])
    return tempNeighbours


def dbscan_image_process(image_A,eps,minPts):
    if type(image_A) == type(np.array([[1,1]])):
        im = Image.fromarray(image_A.astype('uint8'), 'RGB')
    else:
        im = image_A
    pix = im.load()
    xVal,yVal = im.size
    vectors = []
    #Map pixel values to array
    for y in range(yVal):
        vectors.append([])
        for x in range(xVal):
            vectors[y].append(list(pix[x,y]))
    if float(eps) < 1:
        eps = float(eps)
    else:
        eps = eval(eps)#100
    minPts = int(minPts)#100
    distFunc ='m'

    if distFunc != "e" and distFunc != "m":
        sys.exit(1)

    #prepare array
    pointsArray = []
    for y in range(len(vectors)):
        pointsArray.append([])
        for x in range(len(vectors[0])):
            pointsArray[y].append([y,x,vectors[y][x],"Undefined"])

    #DBSCAN
    clusterCounter = 0
    progress = 0
    for y in range(len(vectors)):
        for x in range(len(vectors[0])):
            if pointsArray[y][x][-1] != "Undefined":
                continue

            Neighbours = FindNeighbours(pointsArray[y][x], pointsArray, distFunc, eps)
            if len(Neighbours) < minPts:
                pointsArray[y][x][-1] = "Noise"
                continue
            
            clusterCounter = clusterCounter + 1
            pointsArray[y][x][-1] = str(clusterCounter)
            if pointsArray[y][x] in Neighbours:
                Neighbours.remove(pointsArray[y][x])
            
            for innerPoint in Neighbours:
                if innerPoint[-1] == "Noise":
                    pointsArray[innerPoint[0]][innerPoint[1]][-1] = str(clusterCounter)
                if innerPoint[-1] != "Undefined":
                    continue
                pointsArray[innerPoint[0]][innerPoint[1]][-1] = str(clusterCounter)
                NeighboursInner = FindNeighbours(innerPoint, pointsArray, distFunc, eps)
                if len(NeighboursInner) >= minPts:
                    Neighbours.append(NeighboursInner)



    #Get labels
    labels = []
    for y in range(len(vectors)):
        for x in range(len(vectors[0])):
            labels.append(pointsArray[y][x][-1])
    score = silhouette_score(np.reshape(vectors,(xVal*yVal,3)),np.array(labels))
    #Get distinct clusters
    clusterNumbers = []
    for y in range(len(vectors)):
        for x in range(len(vectors[0])):
            if pointsArray[y][x][-1] not in clusterNumbers:
                clusterNumbers.append(pointsArray[y][x][-1])

    #Map cluster's averages
    averagesForClusters = []
    for item in clusterNumbers:
        n = 0
        vectorTemps = [0]*len(pointsArray[0][0][2])
        for y in range(len(vectors)):
            for x in range(len(vectors[0])):
                if pointsArray[y][x][-1] == item:
                    for i in range(len(pointsArray[y][x][2])):
                        vectorTemps[i] = vectorTemps[i] + pointsArray[y][x][2][i]
                    n = n + 1
        #Check 0 division
        for i in range(len(vectorTemps)):
            if vectorTemps[i] != 0:
                vectorTemps[i] = vectorTemps[i]/n
        averagesForClusters.append(vectorTemps)

    #Build clustered array and change cluster averages with initial values
    clusteredVectors = []
    for y in range(len(pointsArray)):
        clusteredVectors.append([])
        for x in range(len(pointsArray[0])):
            clusteredVectors[y].append(averagesForClusters[clusterNumbers.index(pointsArray[y][x][-1])])




    #Convert input to array and read outputName parameter
    vectors = clusteredVectors
    outputName = 'jilei_is_cunt.jpg'
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
    return np.array(image),score
if __name__ == '__main__':
    image,score = dbscan_image_process('test.jpg',0.03,100)
    print(score)