# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:23:59 2018

@author: linjian
"""

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue     #节点名
        self.count = numOccur     #节点名计次
        self.nodeLink = None
        self.parent = parentNode      #父节点
        self.children = {}           #子节点
    
    def inc(self, numOccur):
        self.count += numOccur
        
    def disp(self, ind=1):            #显示类信息
        print ('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

def createTree(dataSet, minSup=1): #create FP-tree from dataset but don't mine
    headerTable = {}       #字典存储item的频次
    for trans in dataSet:  #C1的频次
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):          #L1的频次
        if headerTable[k] < minSup: 
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())       #取字典的key值
    if len(freqItemSet) == 0: return None, None #没有字典则返回None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] #reformat headerTable to use Node link 
    retTree = treeNode('Null Set', 1, None)     #初始化FP树
    for tranSet, count in dataSet.items():      #重新遍历数据集
        localD = {}
        for item in tranSet:                    #项集里逐个字符遍历
            if item in freqItemSet:             #如果字符是频繁集字符，则加入localD字典
                localD[item] = headerTable[item][0]
        if len(localD) > 0:                     #如果localD有值，则取item
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)  
    return retTree, headerTable #return tree and header table

def updateTree(items, inTree, headerTable, count):   #
    if items[0] in inTree.children:               #如果最远的节点在树内则计数值加1
        inTree.children[items[0]].inc(count)
    else:                                         #如果最远的节点不在树内则加个节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None: #update header table 
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:                      #去掉列表第一个元素后，迭代更新树
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):   #往下链接节点
    while (nodeToTest.nodeLink != None):    
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def ascendTree(leafNode, prefixPath):            #从叶子节点上溯到根
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
    
def findPrefixPath(basePat, treeNode):           #对treeNode进行溯根并增加计数值
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)           
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1][0])]  #根据计数排序取项集
    for basePat in bigL:                                 #从计数最少的频繁项集开始做
        newFreqSet = preFix.copy()                       #集合用于存储频繁项集和
        newFreqSet.add(basePat)                          #
        print ('finalFrequent Item: ',newFreqSet)        #append to set
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])   #返回前缀路径
        #print ('condPattBases :',basePat, condPattBases)
        #2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)             #创建条件树
        #print ('head from conditional tree: ', myHead)
        if myHead != None: #3. mine cond. FP-tree
            print ('conditional tree for: ',newFreqSet)
            myCondTree.disp(1)            
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
        
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

simpDat = loadSimpDat()
retDict = createInitSet(simpDat)
retTree, headerTable = createTree(retDict,3)    #建立FP树
freqItems=[]
mineTree(retTree, headerTable,3,set([]),freqItems)

#minSup = 3
#simpDat = loadSimpDat()
#initSet = createInitSet(simpDat)
#myFPtree, myHeaderTab = createTree(initSet, minSup)
#myFPtree.disp()
#myFreqList = []
#mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)