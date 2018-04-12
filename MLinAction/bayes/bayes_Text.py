# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 20:09:30 2018

@author: linjian
"""

from numpy import *

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString) #利用正则式一串字符切成多个单词
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
                 
def createVocabList(dataSet):  #输出词汇表
    vocabSet = set([])  #set返回不重复的数组，相当于集合
    for document in dataSet:
        vocabSet = vocabSet | set(document) #做并集
    return list(vocabSet)

def bagOfWords2VecMN(vocabList, inputSet): #词汇表中出现文档字符串每次叠加1
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)  #有多少个向量
    numWords = len(trainMatrix[0])  #向量有多少个维度
    pAbusive = sum(trainCategory)/float(numTrainDocs)  #向量标签为1的比例
    p0Num = ones(numWords); p1Num = ones(numWords)      #避免0乘积 
    p0Denom = 2.0; p1Denom = 2.0                       #避免分母是0 
    for i in range(numTrainDocs):   #逐个向量计算
        if trainCategory[i] == 1:   
            p1Num += trainMatrix[i]  #标签为1的向量，计算各维度的和
            p1Denom += sum(trainMatrix[i])  #计算整个类别的和
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #避免最后乘积太小，被截断为0
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #相关性乘以标签比例
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
def spamTest():
    import random
    docList=[]; classList = []; fullText =[]
    docListTrain=[];docListTest=[];classListTrain=[];classListTest=[]
    testIndex = random.sample(range(1,26),5)
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i,errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        if i in testIndex:
            docListTest.append(wordList)
            classListTest.append(1)
        else:
            docListTrain.append(wordList)
            classListTrain.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
        if i in testIndex:
            docListTest.append(wordList)
            classListTest.append(0)
        else:
            docListTrain.append(wordList)
            classListTrain.append(0)
    vocabList = createVocabList(docList) #doclist的词汇表
    trainMat=[]
    for trainInDoc in docListTrain:
        trainMat.append(bagOfWords2VecMN(vocabList,trainInDoc))
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(classListTrain)) 
    errorCount = 0
    testCount = 0
    for testInDoc in docListTest:
        wordVector = bagOfWords2VecMN(vocabList, testInDoc)
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classListTest[testCount]:
            errorCount += 1
            print ("classification error",testInDoc)
        testCount += 1
    print ('the error rate is: ',float(errorCount)/len(docListTest))
    return vocabList,docList

vocabList,docList=spamTest()