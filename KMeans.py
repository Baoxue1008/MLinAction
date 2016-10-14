#coding: utf-8 -*-
__author__ = 'Baoxue1008'
import numpy as np
import random
kernelsNum = 4


real_label = []
real_num = {}

def closestIdx(sample,kernels):
    dis = [np.sum(np.square(np.array(sample) - k)) for k in kernels]
    return np.argmin(dis)


#读入数据
def loadData():
    global real_label
    dir = r'C:\Users\Baoxue1008\Downloads\MLdata\user.data'
    data = open(dir).readlines()
    real_label = [sample.split()[-1] for sample in data]
    data = [sample.split()[0:-1] for sample in data]

    data = np.array(data)
    data =np.float64(data)
    return data

def kmeans():
    data = loadData()
    #随机生成初始中心
    dataNum = len(data)
    kernels = random.sample(data,kernelsNum)


    #kmeans
    changed = True
    while changed == True:
        changed = False
        closests = [closestIdx(s,kernels) for s in data]
        for i in range(kernelsNum):
            belongs = [data[t] for t in range(dataNum) if closests[t] == i]
            if len(belongs) == 0: continue
            newKernel = np.sum(belongs,axis = 0)/len(belongs)
            if abs(np.sum(newKernel - kernels[i])) > 1e-8:
                changed = True
                kernels[i] = newKernel


    closests = [closestIdx(s,kernels) for s in data]
    belongs = [0]*kernelsNum
    for i in range(kernelsNum):
        belongs[i] = [t for t in range(dataNum) if closests[t] == i]
    return kernels,belongs



if __name__ == '__main__':
    kernels,belongs = kmeans()
    for l in real_label:
        if l not in real_num:
            real_num[l] = real_label.count(l)

    a = [[] for i in range(4)]
    for i in range(len(real_label)):
        a[real_num.keys().index(real_label[i])].append(i)


    for i in range(4):
        for j in range(4):
            print i,j,len(set(belongs[i]) & set(a[j]))
    for i in range(len(kernels)):
        print 'Kernel %d: '%(i+1),kernels[i]
        print 'Number of samples belonging to this kernel is:',len(belongs[i])
        print 'These samples are:',belongs[i]
        print
