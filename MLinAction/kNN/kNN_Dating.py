# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:18:41 2018

@author: linjian
"""

from numpy import *
from numpy import array
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def file2matrix(filename):
    fr = open(filename)#读文件，指针存储到readlines
    readOfLines = fr.readlines()
    numberOfLines = len(readOfLines)
    returnMat = zeros((numberOfLines,3))
    classLableVector = []
    index=0
    for line in readOfLines:
        line = line.strip()#逐行读取文本后转化成
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLableVector.append(listFromLine[-1])
        #classLableVector.append(int(listFromLine[-1]))#转化成整数
        index += 1
    return returnMat,classLableVector

def autoNorm(dataSet):#(X-Xmin)/(Xmax-Xmin)
    minVals = dataSet.min(0)#按列取
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))#shape(dataSet)=(1000,3)
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))#tile:复制N份
    normDataSet = normDataSet/tile(ranges, (m,1))   
    return normDataSet, ranges, minVals

def trainDataTest(datingDataMat,datingLabels):
    fig=plt.figure()
    ax=fig.add_subplot(221)#2行2列第1个
    ax.scatter(datingDataMat[:,0],datingDataMat[:,1],s=15*(array(datingLabels)),c=15*(array(datingLabels)))
    #s大小，c颜色，随y值变化
    ax=fig.add_subplot(222)
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2],s=15*(array(datingLabels)),c=15*(array(datingLabels)))
    ax=fig.add_subplot(223)
    ax.scatter(datingDataMat[:,2],datingDataMat[:,0],s=15*(array(datingLabels)),c=15*(array(datingLabels)))
    plt.show()        

def trainTestDivide(normMat,datingLabels,hoRatio):
    m = normMat.shape[0]
    index = arange(m)
    random.shuffle(index)#下标随机排序
    numTestVecs = int(m*hoRatio)
    normMat_new = zeros((m,3))
    datingLabels_new = []
    for i in range(m):#x,y随机排序
        normMat_new[i,:] = normMat[index[i]]
        datingLabels_new.append(datingLabels[index[i]])
    normMat_Test = normMat_new[0:numTestVecs,:]#test和train分开
    normMat_Train = normMat_new[numTestVecs:m,:]
    datingLabels_Test = datingLabels_new[0:numTestVecs]
    datingLabels_Train = datingLabels_new[numTestVecs:m]
    return normMat_Train,normMat_Test,datingLabels_Train,datingLabels_Test

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    sqDiffMat = (tile(inX,(dataSetSize,1)) - dataSet) ** 2
    distances = sqDiffMat.sum(axis=1) ** 0.5
    sortedDistIndicies = distances.argsort()#排序后的序号值
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]#距离从小到大取y值
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1#更新y值个数
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)#从大到小排y值个数
    return sortedClassCount[0][0]

def datingClassTest(k_Num):
    hoRatio = 0.10      #train:0.9,test:0.1
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)#normalization
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorRate_k1 = zeros((k_Num,1))
    errorRate_k2 = zeros((k_Num,1))
    for k in range(k_Num):#不同k值
        errorCount_t1 = zeros((100,1))
        errorCount_t2 = zeros((100,1))
        for t in range(100):#交叉验证100次
            normMat_Train,normMat_Test,datingLabels_Train,datingLabels_Test=trainTestDivide(normMat,datingLabels,hoRatio)
            #调用sklearn
            clf=KNeighborsClassifier(k+1,weights='distance')
            clf.fit(normMat_Train,datingLabels_Train)
            classifierResult = clf.predict(normMat_Test)
            errorCount_t1[t] = sum(classifierResult!=datingLabels_Test)
            #自己源码
            for i in range(numTestVecs):#前10%做测试，后90%做训练
                classifierResult = classify0(normMat_Test[i,:],normMat_Train,datingLabels_Train,k+1)
                if (classifierResult != datingLabels_Test[i]): 
                    errorCount_t2[t] += 1.0
            #print ("the %dth total error rate is: %f" % (t,errorCount[t]/float(numTestVecs)))
        errorRate_k1[k] = average(errorCount_t1)/float(numTestVecs)
        errorRate_k2[k] = average(errorCount_t2)/float(numTestVecs)
        #print ("the average total error rate of %d is: %f" % (k,errorRate_k[k]))
    plt.plot(range(k_Num),errorRate_k1,'b')
    plt.plot(range(k_Num),errorRate_k2,'g')

def datingClassSKlearn(k_Num):
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)#normalization
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorRate_k = zeros((k_Num,1))
    for k in range(k_Num):#不同k值
        errorCount_t = zeros((100,1))
        for t in range(100):#交叉验证100次
            normMat_Train,normMat_Test,datingLabels_Train,datingLabels_Test=trainTestDivide(normMat,datingLabels,hoRatio)
            clf=KNeighborsClassifier(k+1,weights='distance')
            clf.fit(normMat_Train,datingLabels_Train)
            classifierResult = clf.predict(normMat_Test)
            errorCount_t[t] = sum(classifierResult!=datingLabels_Test)
        errorRate_k[k] = average(errorCount_t)/float(numTestVecs)
        #print ("the average total error rate of %d is: %f" % (k,errorRate_k[k]))
    plt.plot(range(k_Num),errorRate_k)

k_Num=100            
#datingClassSKlearn(k_Num)    
datingClassTest(k_Num)    
