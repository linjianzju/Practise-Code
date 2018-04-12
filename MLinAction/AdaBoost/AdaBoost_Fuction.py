# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:43:24 2018

@author: linjian
"""
from numpy import *
from sklearn.ensemble import AdaBoostClassifier

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
    for index in sortedIndicies.tolist():
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


dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
testArr,testlabelArr = loadDataSet('horseColicTest2.txt')
clf=AdaBoostClassifier(n_estimators=300,learning_rate=0.7)  #n_estimators、learning_rate越大越容易overfit，
clf.fit(dataArr,labelArr)
print('score is ',clf.score(dataArr,labelArr))
prediction=clf.predict(testArr)
print('test errRate is ',sum(prediction!=array(testlabelArr))/len(testlabelArr))
prediction_proba=clf.predict_proba(testArr)
plotROC(prediction_proba[:,1]-0.5, testlabelArr)