#encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import operator
import time
def createTrainDataSet():#训练样本,第一个1为阈值对应的w，下同
    trainData=[   [1, 1, 4],
                    [1, 2, 3], 
                    [1, -2, 3], 
                    [1, -2, 2], 
                    [1, 0, 1], 
                    [1, 1, 2]]
    label=[1, 1, 1, -1, -1,  -1]
    return trainData, label
def createTestDataSet():#数据样本
    testData = [   [1, 1, 1],
                   [1, 2, 0], 
                   [1, 2, 4], 
                   [1, 1, 3]]
    return testData
def pla(traindataIn,trainlabelIn):#PLA计算
    traindata=mat(traindataIn)
    trainlabel=mat(trainlabelIn).transpose()
    m,n=shape(traindata)
    w=ones((n,1))
    stop_num=27
    while True:#找到错误则更新，无错误则输出w
        iscompleted=True
        stop_num-=1
        for i in range(m):
            if (sign(dot(traindata[i],w))==trainlabel[i]):
                continue
            else:
                iscompleted=False
                w_temp=w
                w+=(trainlabel[i]*traindata[i]).transpose()
                y_temp=classifyall(traindata,w_temp)
                y=classifyall(traindata,w)
                if dot(y_temp,mat(label).T) > dot(y,mat(label).T):
                    w=w_temp
        if iscompleted or stop_num==0:
            break
    return w
def classifyall(datatest,w):#通过学习到的w计算data_out
    predict=[]
    for data in datatest:
        result=sign(sum(w*data))
        predict.append(result)
    return predict

def plotBestFit(w):#打印图
    traindata,label=createTrainDataSet()
    dataArr = array(traindata)
    n = shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(label[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax= fig.add_subplot(111)
    ax.scatter(xcord1, ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2, ycord2,s=30,c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-w[0]-w[1] * x)/w[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def main():
    trainData,label=createTrainDataSet()#生成训练数据
    testdata=createTestDataSet()#生成data数据
    w=pla(trainData,label)#开始训练，训练出w
    result=classifyall(testdata,w)#根据训练的w跟data算出data_out
    plotBestFit(w)
    print ('w=',w)
    print ('result=',result)
    return trainData,label
if __name__=='__main__':
    start = time.clock()
    trainData,label=main()
    end = time.clock()
    print('finish all in %s' % str(end - start))