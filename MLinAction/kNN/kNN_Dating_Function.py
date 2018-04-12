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
from sklearn import preprocessing
from sklearn import model_selection

def file2matrix(filename):
    fr = open(filename)#读文件，指针存储到readlines
    readOfLines = fr.readlines()
    returnMat = zeros((len(readOfLines),3))
    classLableVector = []
    index=0
    for line in readOfLines:
        listFromLine = line.strip().split('\t')  #逐行读取文本后删除回车
        returnMat[index,:] = listFromLine[0:3]
        classLableVector.append(listFromLine[-1])
        index += 1
    return returnMat,classLableVector

def datingClassSKlearn(k_Range):
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')       #load data setfrom file
    min_max_scaler=preprocessing.MinMaxScaler()    #归一化
    normMat=min_max_scaler.fit_transform(datingDataMat)
    knn_score_k = zeros((len(k_Range),1))
    k_index = 0
    for k in k_Range:#不同k值
        clf=KNeighborsClassifier(n_neighbors=k)#knn
        knn_score = model_selection.cross_val_score(clf,normMat,datingLabels,cv=100)#交叉验证
        knn_score_k[k_index] = average(knn_score)
        k_index += 1
    plt.plot(k_Range,knn_score_k)
    
k_Range=arange(1,100,5)            
datingClassSKlearn(k_Range)    
