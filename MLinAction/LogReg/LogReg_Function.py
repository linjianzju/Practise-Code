# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:16:46 2018

@author: linjian
"""

from numpy import *
from sklearn.linear_model import LogisticRegression

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #x0/x1/x2
        labelMat.extend((lineArr[2]))  #y
    dataMatrix = array(dataMat)    #改为按行排
    labelMatrix = array(labelMat)    #改为按行排
    return dataMatrix,labelMatrix

def plotBestFit(weights,dataMatrix,labelMatrix):
    import matplotlib.pyplot as plt
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(len(labelMatrix)):
        if int(labelMatrix[i])== 1:
            xcord1.append(dataMatrix[i,1]); ycord1.append(dataMatrix[i,2])
        else:
            xcord2.append(dataMatrix[i,1]); ycord2.append(dataMatrix[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = ((-weights[0]-weights[1]*x)/weights[2])
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def loadDataSet1():
    dataMatTrain = []; labelMatTrain = []
    frTrain = open('horseColicTraining.txt')
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        dataMatTrain.append(lineArr)
        labelMatTrain.append(float(currLine[21]))
    dataMatrixTrain = array(dataMatTrain)    #改为按行排
    labelMatrixTrain = array(labelMatTrain)    #改为按行排
    dataMatTest=[];labelMatTest = []
    frTest = open('horseColicTest.txt')
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        dataMatTest.append(lineArr)
        labelMatTest.append(float(currLine[21]))
    dataMatrixTest = array(dataMatTest)    #改为按行排
    labelMatrixTest = array(labelMatTest)    #改为按行排    
    return dataMatrixTrain,labelMatrixTrain,dataMatrixTest,labelMatrixTest
    
    
dataMatrix,labelMatrix=loadDataSet()
clf=LogisticRegression(max_iter=100)
clf.fit(dataMatrix[:,1:3],labelMatrix)
weights=np.append(clf.intercept_,clf.coef_)
plotBestFit(weights,dataMatrix,labelMatrix)

dataMatrixTrain,labelMatrixTrain,dataMatrixTest,labelMatrixTest=loadDataSet1()
clf1=LogisticRegression(max_iter=100)
clf1.fit(dataMatrixTrain,labelMatrixTrain)
labelMatrixPredict=clf1.predict(dataMatrixTest)
errRate=sum(labelMatrixPredict!=labelMatrixTest)/len(labelMatrixTest)
print ("the error rate of this test is: %f" % errRate)
