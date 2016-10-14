# coding:utf-8
__author__ = 'Baoxue1008'
from numpy import *
import re
import random
import feedparser


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    '''
	返回数据集中出现过的所有单词
	'''
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabSet, inputSet):
    '''
	返回vocabSet中各个单词在inputSet中是否出现，不能体现出现次数
	'''
    res = [1 if word in inputSet else 0 for word in vocabSet]
    return res


def bagOfWords2VecMN(vocabList, inputSet):
    '''
	返回vocabSet中各个单词在inputSet中的出现次数
	'''
    res = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            res[vocabList.index(word)] += 1
    return res


def trainNB0(trainMatrix, trainCategory):
    c1 = sum(trainCategory) * 1.0 / len(trainCategory)
    docNum = len(trainMatrix)
    wordNumOf0 = ones(len(trainMatrix[0]))
    wordNumOf1 = ones(len(trainMatrix[0]))
    tot0 = 2.0
    tot1 = 2.0
    for i in range(docNum):
        if trainCategory[i] == 0:
            wordNumOf0 += trainMatrix[i]
            tot0 += sum(trainMatrix[i])

        else:
            wordNumOf1 += trainMatrix[i]
            tot1 += sum(trainMatrix[i])
    p0 = log(wordNumOf0 / tot0)
    p1 = log(wordNumOf1 / tot1)
    return p0, p1, c1


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    return 1 if p1 > p0 else 0


def textParse(bigString):
    words = re.split(r'\W*', bigString)
    return [word.lower() for word in words if len(word) > 2]


def spamTest():
    dir = r'F:\machinelearninginaction\Ch04\email'
    docList = []
    classList = []
    for i in range(1, 26):
        doc = open(dir + '\\spam\\%d.txt' % i).read()
        docList.append(textParse(doc))
        classList.append(1)

        doc = open(dir + '\\ham\\%d.txt' % i).read()
        docList.append(textParse(doc))
        classList.append(0)

    vocabList = createVocabList(docList)
    trainSet = range(50);
    testSet = []
    for i in range(10):
        randIndex = random.randint(0, len(trainSet) - 1)
        testSet.append(trainSet[randIndex])
        del (trainSet[randIndex])
    trainMat = [bagOfWords2VecMN(vocabList, docList[i]) for i in trainSet]
    trainClass = [classList[i] for i in trainSet]
    p0, p1, c1 = trainNB0(trainMat, trainClass)
    errnum = 0
    for i in testSet:
        if classifyNB(bagOfWords2VecMN(vocabList, docList[i]), p0, p1, c1) != classList[i]:
            print i, docList[i]
            errnum += 1
    return float(errnum) / len(testSet)


def calcMostFreq(vocabList, fullText):
    num = {}
    for word in vocabList:
        num[word] = fullText.count(word)
    sortedNum = sorted(num.iteritems(), key=lambda x: x[1], reverse=True)
    return sortedNum[:30]


def localWords(feed0, feed1):
    docList = []
    docClass = []
    allWords = []
    print len(feed0['entries']), len(feed1['entries'])
    minLen = min(len(feed0['entries']), len(feed1['entries']))
    for i in range(minLen):
        curDoc = textParse(feed0['entries'][i]['summary'])
        docList.append(curDoc)
        docClass.append(0)
        allWords.extend(curDoc)
        curDoc = textParse(feed1['entries'][i]['summary'])
        docList.append(curDoc)
        docClass.append(1)
        allWords.extend(curDoc)
    vocabList = createVocabList(docList)
    mostFreq = calcMostFreq(vocabList, allWords)
    trainSet = range(minLen * 2)
    vocabList = [word for word in vocabList if word not in mostFreq]
    testSet = []
    for i in range(20):
        if len(trainSet) == 0:
            break
        randomIndex = random.randint(0, len(trainSet) - 1)
        testSet.append(randomIndex)
        del (trainSet[randomIndex])

    trainMat = []
    trainClass = []
    for i in trainSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[i]))
        trainClass.append(docClass[i])

    p0, p1, c1 = trainNB0(trainMat, trainClass)
    errnum = 0
    for i in testSet:
        if classifyNB(setOfWords2Vec(vocabList, docList[i]), p0, p1, c1) != docClass[i]:
            errnum += 1

    print float(errnum) / len(testSet)
    return vocabList, p0, p1


def getTopWords(ny, sf):
    vocabList, p0, p1 = localWords(ny, sf)
    topNY = [];
    topSF = []
    for i in range(len(p0)):
        if p0[i] > -6.0: topNY.append((vocabList[i], p0[i]))
        if p1[i] > -6.0: topSF.append((vocabList[i], p1[i]))

    topNY = sorted(topNY, key=lambda x: x[1], reverse=True)
    topSF = sorted(topSF, key=lambda x: x[1], reverse=True)

    print 'most words in NY:'
    for i in topNY:
        print i[0],

    print '\nmost words in SF:'
    for i in topSF:
        print i[0],


