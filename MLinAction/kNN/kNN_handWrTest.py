# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:31:13 2018

@author: linjian
"""
from numpy import *
import operator
from os import listdir
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadFromFile(dirName):
    hwLabels=[]
    fileList = listdir(dirName)
    m = len(fileList)
    dataMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = fileList[i]#取第i个文件名
        fileStr = fileNameStr.split('.')[0]     #通过.分成两个，取第0个
        classNumStr = int(fileStr.split('_')[0])#通过_分成两个，取第0个并转为整数
        hwLabels.append(classNumStr)#存储训练集y值
        dataMat[i,:] = img2vector('%s/%s' % (dirName,fileNameStr))#存储训练集x值
    return dataMat,hwLabels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def handwritingClassTest(kRange):
    trainMat,trainLable = loadFromFile('trainingDigits')
    testMat,testLable = loadFromFile('testDigits')
    mTrain = len(trainLable)
    mTest = len(testLable)
    errorCount_k1 = zeros(len(kRange))
    errorCount_k2 = zeros(len(kRange))
    kIndex = 0    
    for k in kRange:
        clf=KNeighborsClassifier(k)
        clf.fit(trainMat,trainLable)
        classifierResult = clf.predict(testMat)
        errorCount_k1[kIndex] = sum(classifierResult!=testLable)        
        for i in range(mTest):
             classifierResult = classify0(testMat[i,:], trainMat, trainLable, k)
             if (classifierResult != testLable[i]): 
                 errorCount_k2[kIndex] += 1.0
        #print ("\nthe %dth total number of errors is: %d" % (kIndex,errorCount_k2[kIndex]))
        kIndex += 1
    plt.plot(kRange,errorCount_k1/float(mTest),'b')
    plt.plot(kRange,errorCount_k2/float(mTest),'g')
    #print ("\nthe total number of errors is: %d" % errorCount)
    #print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))
    
kRange=arange(1,10,2)    
handwritingClassTest(kRange)