# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:12:50 2018

@author: linjian
"""

from numpy import *

def loadDataSet(fileName):      
    dataMat = []               
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) 
        dataMat.append(fltLine)
    return dataMat

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

dataMat=mat(loadDataSet('testSet2.txt'))
centroids, clusterAssment=kMeans(dataMat,3)
dataMat2=mat(loadDataSet('testSet2.txt'))
centroids2, clusterAssment2=biKmeans(dataMat2,3)
print (centroids)
print (centroids2)