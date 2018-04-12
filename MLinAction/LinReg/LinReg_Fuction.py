# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:07:51 2018

@author: linjian
"""

from numpy import *
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,LassoLars
from sklearn.linear_model import RidgeCV,LassoCV,LassoLarsCV,ElasticNetCV

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

def OnceTest(dataMat,labelMat):
    clf1=LinearRegression()
    clf1.fit(dataMat[0:99],labelMat[0:99])
    labelTest1=clf1.predict(dataMat[100:199])
    print ('default LinearRegression' ,((labelTest1-labelMat[100:199])**2).sum())
    clf2=Ridge(alpha=1,max_iter=100,tol=0.001)
    clf2.fit(dataMat[0:99],labelMat[0:99])
    labelTest2=clf2.predict(dataMat[100:199])
    print ('Ridge alhpa=1 max_iter=100 tol=0.001' ,((labelTest2-labelMat[100:199])**2).sum())
    clf3=Lasso(alpha=1,max_iter=100,tol=0.001)
    clf3.fit(dataMat[0:99],labelMat[0:99])
    labelTest3=clf3.predict(dataMat[100:199])
    print ('Lasso alhpa=1 max_iter=100 tol=0.001' ,((labelTest3-labelMat[100:199])**2).sum())
    clf4=ElasticNet(alpha=1,l1_ratio=0.5,max_iter=100,tol=1e-4)
    clf4.fit(dataMat[0:99],labelMat[0:99])
    labelTest4=clf4.predict(dataMat[100:199])
    print ('ElasticNet alhpa=1 max_iter=100 tol=0.001' ,((labelTest4-labelMat[100:199])**2).sum())
    clf5=LassoLars(alpha=1,max_iter=100)
    clf5.fit(dataMat[0:99],labelMat[0:99])
    labelTest5=clf4.predict(dataMat[100:199])
    print ('LassoLars alhpa=1 max_iter=100' ,((labelTest5-labelMat[100:199])**2).sum())

def RidgeTest(dataMat,labelMat):
    clf1=Ridge(alpha=1,max_iter=100,tol=0.001)
    clf1.fit(dataMat[0:99],labelMat[0:99])
    labelTest1=clf1.predict(dataMat[100:199])
    print ('Ridge ' ,((labelTest1-labelMat[100:199])**2).sum())
    clf2=RidgeCV(alphas=[0.1,1,10])
    clf2.fit(dataMat[0:99],labelMat[0:99])
    labelTest2=clf2.predict(dataMat[100:199])
    print ('RidgeCV' ,((labelTest2-labelMat[100:199])**2).sum())
    
def LassoTest(dataMat,labelMat):
    clf1=Lasso(alpha=1,max_iter=100,tol=0.001)
    clf1.fit(dataMat[0:99],labelMat[0:99])
    labelTest1=clf1.predict(dataMat[100:199])
    print ('Lasso ' ,((labelTest1-labelMat[100:199])**2).sum())
    clf2=LassoCV(alphas=[0.1,1,10],max_iter=100,tol=0.001)
    clf2.fit(dataMat[0:99],labelMat[0:99])
    labelTest2=clf2.predict(dataMat[100:199])
    print ('LassoCV' ,((labelTest2-labelMat[100:199])**2).sum())

def LassoLarsTest(dataMat,labelMat):
    clf1=LassoLars(alpha=1,max_iter=100)
    clf1.fit(dataMat[0:99],labelMat[0:99])
    labelTest1=clf1.predict(dataMat[100:199])
    print ('LassoLars ' ,((labelTest1-labelMat[100:199])**2).sum())
    clf2=LassoLarsCV(max_n_alphas=10,max_iter=100)
    clf2.fit(dataMat[0:99],labelMat[0:99])
    labelTest2=clf2.predict(dataMat[100:199])
    print ('LassoLarsCV' ,((labelTest2-labelMat[100:199])**2).sum())
    
dataMat,labelMat = loadDataSet('abalone.txt')

