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


def trainNB0(trainMatrix,trainCategory):
	posi_prob = sum(trainCategory)/float(len(trainMatrix))
	words_num = len(trainMatrix[0])
	p0 = zeros(len(words_num)); p1 = zeros(len(words_num))
	samples_num = len(trainCategory)

	sample0 = [ trainMatrix[i] for i in range(samples_num) if trainCategory[i] == 0]
	sample1 = [ trainMatrix[i] for i in range(samples_num) if trainCategory[i] == 1]

    tot0 = sum(sample0)
    tot1 = sum(sample1)



    sample0.append(ones(words_num))
    sample1.append(ones(words_num))


	prob0 = log(sum(sample0,0) * 1.0 / tot0)
	prob1 = log(sum(sample1,0) * 1.0 / tot1)
    print prob0


if __name__ == 'main':
    a,b = loadDataSet()
    vocabSet = createVocabList(a)
    trainNB0(setOfWords2Vec(a[0]),b[0])



