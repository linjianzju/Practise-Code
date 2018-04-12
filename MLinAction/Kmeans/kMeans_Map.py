# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:12:50 2018

@author: linjian
"""

from numpy import *
from sklearn.cluster import KMeans

def distEclud(vecA, vecB):    #计算两个向量的距离
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):  #随机在数据的min和max之间生成k个中心点
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))   #k个特征维度n的中心点
    for j in range(n):     #逐个维度生成中心点特征值
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]    #数据量
    clusterAssment = mat(zeros((m,2)))  #每个数据点存储最近中心点和距离
    centroids = createCent(dataSet, k)  #创建中心点
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):         #对每个数据点计算最近的中心点
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])  #计算第j中心点和第i个数据点的距离
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:    #所有数据点都验证正确即认为收敛，中心点不再更新
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2   #存储最近中心点和距离
        for cent in range(k):     #更新中心点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] #minIndex是k的数据点作为群
            centroids[cent,:] = mean(ptsInClust, axis=0)  #群均值作为新中心点
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]    #k=1时的中心点
    centList =[centroid0]            #存储中心点
    for j in range(m):       #逐个数据算距离作为误差衡量
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):   #算到中心点数等于k
        lowestSSE = inf
        for i in range(len(centList)):   #逐个尝试划分每一群
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:] #i分群的数据
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  #对i分群做kmeans
            sseSplit = sum(splitClustAss[:,1])  #计算i分群做kmeans后的误差衡量值
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) #对非i计算误差衡量值
            if (sseSplit + sseNotSplit) < lowestSSE:  #找到进行kmeans二分后最好的分群i
                bestCentToSplit = i    #存储i
                bestNewCents = centroidMat    #存储分群i数据做kmeans后的中心点
                bestClustAss = splitClustAss.copy()   #存储分群i数据做kmeans后的最近中心点和距离
                lowestSSE = sseSplit + sseNotSplit   #存储最小SSE
        #更新群的分配结果,1个中心点编号为新增的len(centList)，1个中心点编号为原有的i
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)    
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit  
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] #最佳划分中心点变成两个中心点 
        centList.append(bestNewCents[1,:].tolist()[0])   #存储中心点列表
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#更新数据的最近中心点和距离
    return mat(centList), clusterAssment

import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print (yahooApi)
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print ("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print ("error fetching")
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def KMeans_Function(datMat,numClust):
    clf=KMeans(n_clusters=numClust)
    clf.fit(datMat)
    clustAssing=mat(zeros((shape(datMat)[0],2)))
    myCentroids=mat(clf.cluster_centers_)
    clustAssing[:,0]=mat(clf.labels_).T
    clustAssing[:,1]=mat(clf.inertia_).T 
    return myCentroids,clustAssing

def clusterClubs(numClust=4):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = KMeans_Function(datMat,numClust)
    #myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

clusterClubs()
    
