# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:18:16 2018

@author: linjian

选能最佳划分的特征维度，作为决策父节点进行建树，将父节点所在的特征维度值分类后剔除再进行迭代建树
"""

from math import log
import operator
from pickle import dump,load

def loadFromFile(filename):
    fr = open(filename,'r')
    lense = [inst.strip().split('\t') for inst in fr.readlines()]
    lenseLabel = ['age','prescript','astigmatic','tearRate']
    return lense,lenseLabel

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {} #字典
    for featVec in dataSet: #为可能分类创建字典
        currentLabel = featVec[-1] #取分类标识
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  #分类数加1
    shannonEnt = 0.0   
    for key in labelCounts:   #shannon=sum(-p(x)*log2(p(x))
        prob = float(labelCounts[key])/numEntries   
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

def splitDataSet(dataSet, axis, value):#待划分数据集、划分特征、特征期望值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:  #特征值跟特征期望值相同
            reducedFeatVec = featVec[:axis]     #将特征值剔除
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)  
    return retDataSet

def chooseBestFeatureToSplit(dataSet):#选最好的划分方式
    numFeatures = len(dataSet[0]) - 1      #特征数
    baseEntropy = calcShannonEnt(dataSet)  #未划分前的熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]  #取第i列特征维度
        uniqueVals = set(featList)       #特征向量有的特征值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) #不同的划分数据集方式
            prob = len(subDataSet)/float(len(dataSet))  
            newEntropy += prob * calcShannonEnt(subDataSet)   #划分后的熵  
        infoGain = baseEntropy - newEntropy     #熵增益
        if (infoGain > bestInfoGain):       #比较后取最好的熵增益
            bestInfoGain = infoGain         
            bestFeature = i
    return bestFeature              #返回划分最好的特征维度        

def majorityCnt(classList):  #叶子节点进行特征判决
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): 
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): #标签全是同个值
        return classList[0]  
    if len(dataSet[0]) == 1: #只有1列标签，返回判决结果
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  #选最佳划分维度
    bestFeatLabel = labels[bestFeat]  #最佳划分维度对应的label
    myTree = {bestFeatLabel:{}}   #将要决策的label划入树
    del(labels[bestFeat])    #在待选label中删除划入树的label
    featValues = [example[bestFeat] for example in dataSet]  #判决的特征维度向量
    uniqueVals = set(featValues)  #判决的特征维度值
    for value in uniqueVals:    #将判决剩余的特征维度、label进行建树迭代
        subLabels = labels[:]       
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree         
"""
原始是[[1, 1, 'yes'],[1, 1, 'yes'],[1, 0, 'no'],[0, 1, 'no'],[0, 1, 'no']]
第一次迭代最佳划分是第1个维度，分割后进行第二次迭代
value=0时第二次迭代输入是[[1, 'no'],[1, 'no'],标签全是同个值不需要划分，直接判决为‘no’
value=1时第二次迭代输入是[[1, 'yes'],[1, 'yes'],[0, 'no']]，最佳划分是第2个维度，分割后进行第三次迭代
       value=0时第三次迭代输入是[['no']]，直接判决为'no'
       value=1时第三次迭代输入是[[1, 'yes'],[1, 'yes']]，直接判决为'yes'
"""

def classify(inputTree,featLabels,testVec):  #输入是决策树结构，判决标签，测试向量
    firstStr = tuple(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: 
        classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    fw = open(filename,'wb')
    dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    fr = open(filename,'rb')
    return load(fr)
    

dataSet,labelsTrain = loadFromFile('lenses.txt')
labelsTest=labelsTrain.copy()
myTree = createTree(dataSet,labelsTrain)
#storeTree(myTree,'classifierStorage.pkl')
#myTree = grabTree('classifierStorage.pkl')
#classLabel=[]
#classify(myTree,labelsTest,testData[0:4])
