from numpy import *


def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmoid(n):
    return 1.0/(1 + exp(-n))

def gradAscent(dataMat,dataLabel):
    dataMat = mat(dataMat)
    dataLabel = mat(dataLabel).transpose()
    maxCycles = 500
    alpha = 0.001
    m,n = shape(dataMat)
    weight = ones((n,1))

    for i in range(maxCycles):
        h = sigmoid(dataMat * weight)
        error = dataLabel - h
        weight = weight + alpha *


