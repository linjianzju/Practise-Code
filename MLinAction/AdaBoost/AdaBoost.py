# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:43:24 2018

@author: linjian
"""
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#多次切片，需要注意+-1和大小的对应关系
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
  
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)  #m是数据个数，n是特征维度
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf     #初始化为无限大
    for i in range(n):  #单维度切，逐个维度循环做
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();  #单维度最大最小
        stepSize = (rangeMax-rangeMin)/numSteps  #最大到最小之间切片移动numSteps次
        for j in range(-1,int(numSteps)+1):  #逐次做判断分类
            for inequal in ['lt', 'gt']:  #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)  #当前切片位置
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal) #对第i个维度进行分类，根据切片位置和大小进行分类
                errArr = mat(ones((m,1)))  #错误个数
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #权重向量D，sum(D[i]*e[i])
                if weightedError < minError:   #更新最小误差，分类结果，得到最小误差的维度、切片位置和+-1关系,
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D) #返回单次最优解
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))  #a=0.5*ln((1-err)/err)
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  #最佳分类的参数
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)  #正确分类数多，expon=-a，错误分类数多，expon=a
        D = multiply(D,exp(expon))/D.sum()                      #D=D*exp(expon)/sum(D)
        #D = D/D.sum()                                           
        #进行分类，当分类完全正确是退出循环
        aggClassEst += alpha*classEst                  
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))#算错误率
        errorRate = aggErrors.sum()/m
        #print ("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst  #返回最佳分类的参数和最佳分类结果

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass) #do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):  #每个维度单独分类
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print (aggClassEst)
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()         #从小到大排序返回序号
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #从置信度负最大的开始判断，如果实际分类为1，则cur在Y轴倒退一步，如果实际分类为-1，则cur在X轴倒退一步。
    #如果分类完全正确，cur会先在X轴回退到0,然后再开始Y轴回退。
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]   #出现X轴回退时累加光标的Y轴距离作为小矩形面积
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print ("the Area Under the Curve is: ",ySum*xStep)

#datMat,classLabels=loadSimpData()
#weakClassArr,aggClassEst=adaBoostTrainDS(datMat,classLabels)
#aggClassEstSign=adaClassify([1,2],weakClassArr)
dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
weakClassArr,aggClassEst = adaBoostTrainDS(dataArr,labelArr,5000)
testArr,testlabelArr = loadDataSet('horseColicTest2.txt')
prediction10 = adaClassify(testArr,weakClassArr)
errArr = mat(ones((67,1)))
errRate=sum(errArr[prediction10!=mat(testlabelArr).T])/67
plotROC(aggClassEst.T, labelArr)