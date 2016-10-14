#coding:utf-8
import numpy as np
from numpy.random import rand
__author__ = 'Baoxue1008'


Preference = 13
real_num = {}
real_label = []
def ap_cluster(S) :

    n = S.shape[0]
    realmax = np.finfo(float).max
    realmin = np.finfo(float).tiny
    eps = np.finfo(float).eps
    S = S + (eps*S+realmin*100)*rand(n,n)
    A = np.zeros((n,n))
    R = np.zeros((n,n))
    lam = 0.5

    for itr in range(500) :

        # 计算 responsibilities
        Rold = R
        AS = A + S
        Y, I = AS.max(1), AS.argmax(1)
        for i in range(n) :
            AS[i,I[i]] = -realmax
        Y2, I2 = AS.max(1), AS.argmax(1)
        R = S - np.kron(np.ones((n,1)),Y).T
        for i in range(n) :
            R[i,I[i]] = S[i,I[i]]-Y2[i]
        # Dampen responsibilities
        R = (1-lam)*R + lam*Rold

        # 计算 availabilities
        Aold = A
        Rp = np.maximum(R,0)
        for i in range(n) :
            Rp[i,i] = R[i,i]
        A = np.kron(np.ones((n,1)),Rp.sum(0)) - Rp
        dA = np.diag(A)
        A = np.minimum(A,0)
        for i in range(n) :
            A[i,i] = dA[i]
        A = (1-lam)*A + lam*Aold

    E = R.T + A
    end_I = np.where(np.diag(E)>0)[0]
    K = end_I.size
    tmp, c = S[:,end_I].max(1), S[:,end_I].argmax(1)
    c[end_I] = np.arange(K)
    #return E, R, A, end_I, c, idx
    belongs = []
    for k in range(K):
        t = [i for i in range(len(c)) if c[i] == k]
        belongs.append(t)
    return end_I,belongs


def getS(data):
    N = np.shape(data)[0]
    S = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            S[i,j] = - np.sum(np.square(data[i] - data[j]))
    avg = np.sum(np.sum(S))/(N*N)
    for i in range(N):
        S[i,i] = avg*Preference
    return S

def loadData():
    global real_label
    dr = r'C:\Users\Baoxue1008\Downloads\MLdata\user.data'
    data = open(dr).readlines()
    real_label = [sample.split()[-1] for sample in data]
    data = [sample.split()[:-1] for sample in data]
    data = np.array(data)
    data = np.float64(data)
    return data


if __name__ == '__main__':

    data = loadData()
    kernels,belongs = ap_cluster(getS(data))

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
        print 'Kernel %d: '%(i+1),data[kernels[i]]
        print 'Number of samples belonging to this kernel is:',len(belongs[i])
        print 'These samples are:',belongs[i]
        print

