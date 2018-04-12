# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:16:46 2018

@author: linjian
"""

from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #x0/x1/x2
        labelMat.append(int(lineArr[2]))  #y
    dataMatrix = mat(dataMat).transpose()    #改为按列排
    labelMatrix = mat(labelMat)    #改为按列排
    return dataMatrix,labelMatrix

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatrix, labelMatrix):
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((m,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(weights.transpose()*dataMatrix)     #h=sig(wTx)
        error = (labelMatrix - h)                          #e=y-h
        weights = weights + (alpha*error*dataMatrix.transpose()).transpose()   #w=w+a*e*x
    return weights

def stocGradAscent(dataMatrix, labelMatrix, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones((m,1))   
    for j in range(numIter):
        dataIndex = list(range(n))
        for i in range(n):
            alpha = 4/(1.0+j+i)+0.0001    #每次减少步长
            randIndex = int(random.uniform(0,len(dataIndex)))#随机选取一个x
            h = sigmoid(weights.transpose()*dataMatrix[:,randIndex])
            error = labelMatrix[:,randIndex] - h
            weights = weights + alpha*dataMatrix[:,randIndex]*error
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights,dataMatrix,labelMatrix):
    import matplotlib.pyplot as plt
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(labelMatrix.shape[1]):
        if int(labelMatrix[0,i])== 1:
            xcord1.append(dataMatrix[1,i]); ycord1.append(dataMatrix[2,i])
        else:
            xcord2.append(dataMatrix[1,i]); ycord2.append(dataMatrix[2,i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = ((-weights[0]-weights[1]*x)/weights[2]).transpose()
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(weights.transpose()*inX)
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent(mat(trainingSet).transpose(),mat(trainingLabels), 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(mat(lineArr).transpose(), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
    
#dataMatrix,labelMatrix=loadDataSet()
#weights=stocGradAscent(dataMatrix,labelMatrix)
#plotBestFit(weights,dataMatrix,labelMatrix)
multiTest()