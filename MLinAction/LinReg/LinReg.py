# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:07:51 2018

@author: linjian
"""

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(testxArr,xArr,yArr):  #正则方程式解法
    testxMat=mat(testxArr); xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    m = shape(testxMat)[0]
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = testxMat[i]*ws
    #fig = plt.figure()
    #ax=fig.add_subplot(111)
    #ax.scatter(xMat[:,1].flatten().A[0],yMat[:,0].T.flatten().A[0])
    #ax.plot(xMat[:,1],yHat,'b')
    #print('corrcoef is ',corrcoef(yHat.T,yMat.T))
    return yHat

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))  #w[j,j]=exp((||x[j]-x[i]||)/(-2k^2))
        #k越大，w越接近I，说明越接近直线回归，过拟合风险越小，欠拟合风险越大
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))  #w=inv(X.TWX)X.TWy
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    #srtInd=xMat[:,1].argsort(0)
    #xSort=xMat[srtInd][:,0,:]
    #fig = plt.figure()
    #ax=fig.add_subplot(111)
    #ax.scatter(xMat[:,1].flatten().A[0],yMat[:,0].T.flatten().A[0])
    #ax.plot(xSort[:,1],yHat[srtInd],'b')
    #print('corrcoef is ',corrcoef(yHat.T,yMat.T))
    return yHat

def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws
    
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #先对X,y进行处理，满足零均值和单位方差
    xMeans = mean(xMat,0)   
    xVar = var(xMat,0)      
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #先对X,y进行处理，满足零均值和单位方差
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   
    inVar = var(inMat,0)      
    inMat = (inMat - inMeans)/inVar    
    m,n=shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):    
        lowestError = inf; 
        for j in range(n):  #逐个特征维度,选正负方向做一小步，看rsse
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = ((yMat.A-yTest.A)**2).sum()
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

def linearTest(dataMat,labelMat):
    yHat01=lwlrTest(dataMat[0:99],dataMat[0:99],labelMat[0:99],0.1)
    yHat1=lwlrTest(dataMat[0:99],dataMat[0:99],labelMat[0:99],1)
    yHat10=lwlrTest(dataMat[0:99],dataMat[0:99],labelMat[0:99],10)
    print ('train mse 0.1 is', ((labelMat[0:99]-yHat01)**2).sum())
    print ('train mse 1 is', ((labelMat[0:99]-yHat1)**2).sum())
    print ('train mse 10 is', ((labelMat[0:99]-yHat10)**2).sum())
    yHat01=lwlrTest(dataMat[100:199],dataMat[0:99],labelMat[0:99],0.1)
    yHat1=lwlrTest(dataMat[100:199],dataMat[0:99],labelMat[0:99],1)
    yHat10=lwlrTest(dataMat[100:199],dataMat[0:99],labelMat[0:99],10)
    print ('test mse 0.1 is', ((labelMat[100:199]-yHat01)**2).sum())
    print ('test mse 1 is', ((labelMat[100:199]-yHat1)**2).sum())
    print ('test mse 10 is', ((labelMat[100:199]-yHat10)**2).sum())
    yHatSR=standRegres(dataMat[100:199],dataMat[0:99],labelMat[0:99])
    print ('testSR mse 1 is', ((labelMat[100:199]-yHatSR)**2).sum())

#dataMat,labelMat = loadDataSet('ex0.txt')
dataMat,labelMat = loadDataSet('abalone.txt')
returnMat=stageWise(dataMat,labelMat)
