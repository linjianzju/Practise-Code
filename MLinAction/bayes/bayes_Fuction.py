# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 20:09:30 2018

@author: linjian
"""

from numpy import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import  TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split

def spamTest():
    import random
    docList=[]; classList = []; fullText =[]
    docListTrain=[];docListTest=[];classListTrain=[];classListTest=[]
    for i in range(1,26):
        wordList = open('email/spam/%d.txt' % i,errors='ignore').read()
        docList.append(wordList)
        classList.append(1)
        wordList = open('email/ham/%d.txt' % i,errors='ignore').read()
        docList.append(wordList)
        classList.append(0)
    vectorizer = TfidfVectorizer()
    docListVec=vectorizer.fit_transform(docList)
    trainMat, testMat, classListTrain, classListTest = train_test_split(docListVec, classList, test_size = 0.2)
    clf = MultinomialNB().fit(trainMat, classListTrain)
    classListPredicted = clf.predict(testMat)
    errorCount = sum(classListPredicted!=classListTest)
    print ('the error rate is: ',float(errorCount)/len(classListTest))
    return docListVec

docListVec=spamTest()