# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:57:24 2018

@author: linjian
"""

from numpy import * 
import matplotlib.pyplot as plt

def loadData(filename):
    #data = loadtxt(filename)
    #x = data[:,0:2]
    #y = data[:,2:3]
    x = array([[ -1.,  3.,],
               [ 4.,   7.,],
               [ 6.,   5.,],
               [-6.,  -4.,],
               [-3.,  -5.,],
               [-3.,  -6.,]])
    y = array([[ 1.],
               [ 1.],
               [ 1.],
               [-1.],
               [-1.],
               [-1.]])
    return x,y


#the sigmoid function
def sigmoid(x):
    return 1.0 / (1 + exp(-x))


#the cost function
def costfunction(y,h):
    y = array(y)
    h = array(h)
    J = sum(y*log(h))+sum((1-y)*log(1-h))
    return J


# the batch gradient descent algrithm
def gradescent(x,y):
    m,n = shape(x)     #m: number of training example; n: number of features
    x = c_[ones(m),x]     #add x0
    x = mat(x)      # to matrix
    y = mat(y)
    a = 0.002       # learning rate
    maxcycle = 2000
    theta = ones((n+1,1))    #initial theta

    J = []
    for i in range(maxcycle):
        h = sigmoid(x*theta)
        theta = theta + a * x.transpose()*(y-h)
        cost = costfunction(y,h)
        J.append(cost)

    plt.plot(J)
    plt.show()
    return theta,cost


#the stochastic gradient descent (m should be large,if you want the result is good)
def stocGraddescent(x,y):
    m,n = shape(x)     #m: number of training example; n: number of features
    x = c_[ones(m),x]     #add x0
    x = mat(x)      # to matrix
    y = mat(y)
    a = 0.01       # learning rate
    theta = ones((n+1,1))    #initial theta
    
    J = []
    for i in range(m):
        h = sigmoid(x[i]*theta)
        theta = theta + a * x[i].transpose()*(y[i]-h)
        cost = costfunction(y,h)
        J.append(cost)
        plt.plot(J)
        plt.show()
    return theta,cost


#plot the decision boundary
def plotbestfit(x,y,theta):
    plt.plot(x[:,0:1][where(y==1)],x[:,1:2][where(y==1)],'ro')
    plt.plot(x[:,0:1][where(y!=1)],x[:,1:2][where(y!=1)],'bx')
    x1= arange(-4,4,0.1)
    x2 =(-float(theta[0])-float(theta[1])*x1) /float(theta[2])
    
    plt.plot(x1,x2)
    plt.xlabel('x1')
    plt.ylabel(('x2'))
    plt.show()
    

def classifyVector(inX,theta):
    prob = sigmoid(sum(inX*theta))
    print ('the probobility is:',prob)
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

if __name__=='__main__':
    x,y = loadData("testset.txt")
    theta,cost = gradescent(x,y)
    print ('theta:\n',theta)
    print ('J:',cost)

    X = [1,2,9]
    print ('the new input:',X)
    h = classifyVector(X,theta)
    print ('the predict y:',h)
    plotbestfit(x,y,theta)
