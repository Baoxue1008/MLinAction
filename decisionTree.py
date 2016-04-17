import math
import matplotlib.pyplot as plt
def createTestData():
    dataSet = [[1,1,'yes'],
	    [1,1,'yes'],
	    [1,0,'no'],
	    [0,1,'no'],
	    [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels



def calcShannonEnt(dataset):
    labelList = [t[-1] for t in dataset]
    unequalLabel = set(labelList)
    labelcounts = [labelList.count(i) for i in unequalLabel]
    prob = [ float(i)/len(labelList) for i in labelcounts]
    entropy = sum( [-i * math.log(i,2) for i in prob ])
    return entropy
	

def splitDataSet(dataSet,axis,value):
    treatedSet = filter(lambda x:x[axis] == value,dataSet)
    retSet = [d[:axis] + d[axis+1:] for d in treatedSet]
    return retSet


def chooseBestFeatureToSplit(dataSet):
    featureNum = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestFeature = -1
    for i in range(featureNum):
	newEntropy = 0.0
	difValue = set([val[i] for val in dataSet])
	for val in difValue:
	    subData = splitDataSet(dataSet,i,val)
	    prob = len(subData) * 1.0/len(dataSet)
	    newEntropy += prob * calcShannonEnt(subData)
	
	if newEntropy < baseEntropy:
	    baseEntropy = newEntropy
	    bestFeature = i
    return bestFeature


def majorityCnt(classList):
    difClass = set(classList)
    Count =[ classList.count(i) for i in difClass]
    retClass = difClass[Count.index(max(Count))]
    return retClass

def createTree(dataSet,labels):
    allLabels = [d[-1] for d in dataSet]
    if allLabels.count(allLabels[0]) == len(allLabels):
	    return allLabels[0]
    if len(dataSet[0]) == 1:
	    return majorityCnt(dataSet)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestLabel = labels[bestFeature]
    retTree = {bestLabel:{}}
    uniqueVals = set([ d[bestFeature] for d in dataSet])
    for v in uniqueVals:
	    subSet =  splitDataSet(dataSet,bestFeature,v)
	    subLabels = labels[:bestFeature] + labels[bestFeature+1:]
	    retTree[bestLabel][v] = createTree(subSet,subLabels)
    return retTree



	
