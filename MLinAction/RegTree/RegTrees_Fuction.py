# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:54:43 2018

@author: linjian
"""

from numpy import *
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map表示对指定序列做运算，本例为映射成float
        #fltLine = float(curLine)
        dataMat.append(fltLine)
    return dataMat


"""回归树"""
def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

"""模型树"""
def linearSolve(dataSet):   #X和y做线性回归
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#X0=1，将X和y开两个矩阵
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y
def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

"""选择最佳维度并进行划分为两个矩阵"""
def binSplitDataSet(dataSet, feature, value):  #数据集、待划分的特征维度、判断特征值门限
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]  #nonzero返回dataSet某个特征大于门限的下标
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1
def chooseBestSplit(dataSet, leafType, errType, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #所有y值都是同一个，说明是叶子节点
        return None, leafType(dataSet)   #返回y均值
    m,n = shape(dataSet)  #m是训练集数量，n-1是特征维度，最后是y
    S = errType(dataSet)  #y的方差
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):  #逐个特征维度判断
        for splitVal in set(array(dataSet[:,featIndex]).T[0]):  #在每个特征维度上用逐个数据进行二分，用哪个最佳 
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)  #splitVal是用于二分的数据，mat0是大值，mat1是小值
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue    #切分出来有一边太小则不算
            newS = errType(mat0) + errType(mat1)    #用方差之和作为数据集切分的衡量，类似决策树的熵
            if newS < bestS:     #存储最佳切分维度、最佳切分值、新的方差和
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:      #误差减少不大则退出
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #切分出来有一边太小则退出
        return None, leafType(dataSet)
    return bestIndex,bestValue  #返回最佳切分维度和切分值

"""迭代进行创建树"""
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#leafType是y的均值，errType是y的方差
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)     #选择最佳切分方式的特征维度和特征值门限
    if feat == None: return val     #如果没有维度说明不可再分，返回树的模型即均值
    retTree = {}  #'spInd'是切分维度，'spVal'是切分值，'left'是左树，'right'是右树。如果叶子节点则为树的模型即均值
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)  #根据特征值进行左右树分叉
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree    

"""树剪枝"""
def isTree(obj):  #测试变量是否树，是则返回True，否则返回False
    return (type(obj).__name__=='dict')
def getMean(tree):  #树递归，如果两边都是叶子节点则返回左右的均值
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
def prune(tree, testData):  #如果合并叶子节点后误差减小则进行剪枝，有利于ops选取不准确导致的过拟合情况
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])): #如果左或右有树，则对测试数据进行切分
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)  #如果左是树，则递归左树和切分后的左数据
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)  #如果右是树，则递归右树和切分后的右数据
    #如果左右都是叶子节点，则对测试数据切分后计算切分数据的误差值errorNoMerge,和不切分数据的误差值errorMerge
    if not isTree(tree['left']) and not isTree(tree['right']):   
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:     #不切分误差小于切分误差则聚合不切分
            print ("merging")
            return treeMean
        else: return tree
    else: return tree

"""对testdata进行预测"""
def regTreeEval(model, inDat):    #回归树预测
    return float(model)
def modelTreeEval(model, inDat):  #模型树预测
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)
def treeForeCast(tree, inData, modelEval=regTreeEval):  #预测
    if not isTree(tree): return modelEval(tree, inData)    #树只有1个叶子节点则直接预测
    if inData[tree['spInd']] > tree['spVal']:              #根据树存储的特征维度进行判决预测
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = zeros((m,1))
    for i in range(m):
        yHat[i] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

    
dataMatTrain=loadDataSet('bikeSpeedVsIq_train.txt')
dataMatTest=loadDataSet('bikeSpeedVsIq_test.txt')
myTreeReg=createTree(mat(dataMatTrain), leafType=regLeaf, errType=regErr, ops=(1,20))
yHatReg=createForeCast(myTreeReg, mat(dataMatTest)[:,0], modelEval=regTreeEval)
print ('corrcoefReg is' , corrcoef(mat(yHatReg),mat(dataMatTest)[:,1],rowvar=0)[0,1])
clf1=DecisionTreeRegressor(min_impurity_decrease=1,min_samples_split=20)
clf1.fit(mat(dataMatTrain)[:,0],mat(dataMatTrain)[:,1])
yHatRegFun=clf1.predict(mat(dataMatTest)[:,0])
print ('corrcoefRegFun is' , corrcoef(yHatRegFun,mat(dataMatTest)[:,1],rowvar=0)[0,1]) 

myTreeModel=createTree(mat(dataMatTrain), leafType=modelLeaf, errType=modelErr, ops=(1,20))
yHatModel=createForeCast(myTreeModel, mat(dataMatTest)[:,0], modelEval=modelTreeEval)
print ('corrcoefModel is' , corrcoef(mat(yHatModel),mat(dataMatTest)[:,1],rowvar=0)[0,1])

#Linear Regression
wsLin,XLin,yLin=linearSolve(mat(dataMatTrain))
yHatLin=zeros((shape(mat(dataMatTest))[0],1))
for i in range(shape(mat(dataMatTest))[0]):
    yHatLin[i]=mat(dataMatTest)[i,0]*wsLin[1,0]+wsLin[0,0]
print ('corrcoefLin is' , corrcoef(mat(yHatLin),mat(dataMatTest)[:,1],rowvar=0)[0,1]) 
clf3=LinearRegression()
clf3.fit(mat(dataMatTrain)[:,0],mat(dataMatTrain)[:,1])
yHatLinFun=clf3.predict(mat(dataMatTest)[:,0])
print ('corrcoefLinFun is' , corrcoef(yHatLinFun,mat(dataMatTest)[:,1],rowvar=0)[0,1]) 