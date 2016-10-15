from numpy import *
import random

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
        weight = weight + alpha * dataMat.transpose() * error

    return weight

def stocGradAscent0(dataMat,dataLabel):
    dataMat = array(dataMat)
    m,n = shape(dataMat)
    weight = ones(n)
    alpha = 0.01
    for i in range(m):
        h = sigmoid(sum(dataMat[i] * weight))
        error = dataLabel[i] - h
        weight = weight + alpha *  dataMat[i]
    return weight

def stocGradAscent1(dataMat,dataLabel,numIter=150):
    dataMat = array(dataMat)
    m,n = shape(dataMat)
    weights = ones(n)

    for i in range(numIter):
        indexs = range(m)
        for j in range(m):
            alpha = 0.01 + 4.0/(i + j + 1)
            cur = int(random.uniform(0,len(indexs)))
            h = sigmoid(sum(dataMat[cur]*weights))
            error = dataLabel[cur] - h
            weights = weights + alpha * error * dataMat[cur]
            del(indexs[cur])

    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

if __name__ == '__main__':
    a,b = loadDataSet()
    x,y = shape(a)
    weights = stocGradAscent1(a,b)

    plotBestFit(weights)


