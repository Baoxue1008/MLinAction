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
    samples_num = len(trainCategory)

    sample0 = [ trainMatrix[i] for i in range(samples_num) if trainCategory[i] == 0]
    sample1 = [ trainMatrix[i] for i in range(samples_num) if trainCategory[i] == 1]

    tot0 = sum(sample0)
    tot1 = sum(sample1)



    #prob0 = sum(sample0,0) * 1.0 / tot0
    #prob1 = sum(sample1,0) * 1.0 / tot1

    sample0.append(ones(words_num))
    sample1.append(ones(words_num))
    tot0 += 2
    tot1 += 2

    prob0 = log(sum(sample0,0) * 1.0 / tot0)
    prob1 = log(sum(sample1,0) * 1.0 / tot1)

    return prob0,prob1,posi_prob

def classifyNB(vec2classify,prob0,prob1,posi_prob):
    p0 = sum(vec2classify * prob0) + log(posi_prob)
    p1 = sum(vec2classify * prob1) + log(1-posi_prob)
    return 0 if p0 > p1 else 1

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)




if __name__ == '__main__':
    # a,b = loadDataSet()
    # vocabSet = createVocabList(a)
    # trainMat = []
    # for i in a:
    #     trainMat.append(setOfWords2Vec(vocabSet,i))
    # a,b,c = trainNB0(trainMat,b)
    # print a,b,c
    testingNB()


