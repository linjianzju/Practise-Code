# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:48:04 2018

@author: linjian
"""

from numpy import * 
from sklearn import decomposition   

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]  #取数转化成二维列表
    #datArr = [map(float,line) for line in stringArr]
    datArr = []
    for line in stringArr:
        datArr.append([float(x) for x in line])
    return mat(datArr)

def pcaSelf(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)      #按列求均值，逐个特征求 
    meanRemoved = dataMat - meanVals      #去均值
    covMat = cov(meanRemoved, rowvar=0)   #求协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat))   #求特征值eigVals和特征向量eigVects
    eigValInd = argsort(eigVals)                 #特征值排序并删掉门限下的维度
    eigValInd = eigValInd[:-(topNfeat+1):-1]  
    redEigVects = eigVects[:,eigValInd]          #特征向量取保留维度
    lowDDataMat = meanRemoved * redEigVects      #数据保留新维度
    reconMat = (lowDDataMat * redEigVects.T) + meanVals   #重构数据作为验证
    return lowDDataMat, reconMat

def replaceNanWithMean():    #NaN替换成平均值
    datMat = loadDataSet('secom.data', ' ')   
    numFeat = shape(datMat)[1]     #数据维度
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #每个维度对非NaN取均值，然后替换NaN
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  
    return datMat

#dataMat=loadDataSet('testSet.txt')
#lowDDataMat, reconMat=pca(dataMat,1)
dataMat=replaceNanWithMean()
lowDDataMat, reconMat=pcaSelf(dataMat,20)
print(linalg.norm(dataMat-reconMat)/(shape(dataMat)[0]*shape(dataMat)[1]))
clf=decomposition.PCA(n_components=20)
lowDDataMat1=clf.fit_transform(dataMat)
reconMat1=clf.inverse_transform(lowDDataMat1)
print(linalg.norm(dataMat-reconMat1)/(shape(dataMat)[0]*shape(dataMat)[1]))