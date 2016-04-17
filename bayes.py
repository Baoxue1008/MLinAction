from numpy import *


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


def createVocabList(dataSet):
	'''
	return all the words appeared in dataset
	'''
	vocabSet = set()
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)


def setOfWords2Vec(vocabSet,inputSet):
	res = [1 if word in inputSet else 0 for word in vocabSet]
	return res
	'''
	res = [0]*len(vocabSet)
	for i in range(len(vocabSet)):
		res[i] = 1 if vocabSet[i] in inputSet else 0 
	return res
	'''

def trainNB0(trainMatrix,trainCategory):
	c0 = sum(trainCategory)*1.0/len(trainCategory)
	docNum = len(trainMatrix)
	wordNumOf0 = zeros(len(trainMatrix[0]))
	wordNumOf1 = zeros(len(trainMatrix[0]))
	tot0 = 0.0
	tot1 = 0.0
	for i in range(docNum):
		if trainCategory[i] == 0:
			wordNumOf0 += trainMatrix[i]
			tot0 += sum(trainMatrix[i])

		else :
			wordNumOf1 += trainMatrix[i]
			tot1 += sum(trainMatrix[i])
	p0 = wordNumOf0/tot0
	p1 = wordNumOf1/tot1
	
	return p0,p1,c0


