# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 13:12:06 2021

@author: ikerc
"""
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import stats
from RF_cython import DensityRF
import time

def syntheticData(d, N):
    data = np.zeros((d,N))
    for i in range(d):
        x1 = np.random.uniform(low = 0.7, high = 1.0, size = int(0.3*N))
        x2 = np.random.uniform(low = 0.0, high = 0.4, size = int(0.7*N))
        x = np.concatenate((x1,x2))
        np.random.shuffle(x)
        data[i,:] = x
    return data

def fMarginal(x):
    if x >= 0 and x <= 0.4:
        return 7/4
    elif x >= 0.7 and x <= 1:
        return 1
    else:
        return 0
    
def f1(data):
    d, n = data.shape
    density = np.ones(n)
    for i in range(n):
        for j in range(d):
            density[i] *= fMarginal(data[j, i])
    return density


nrExp = 1
sizeSample = 500
#sizeSampleTest = 100
d = 5

nTreesList = [25, 50, 100, 200]
minSizeList = [5, 10, 25, 50]
pList = [0.1*i for i in range(11)]

# nTreesList = [25, 50]
# minSizeList = [5, 50]
# pList = [0.1, 0.5, 0.9]

# paramGrid = {'nTrees': [25, 50, 100, 200],
#              'minSize': [5, 10, 25, 50],
#              'p': [0.1*i for i in range(1,11)]}

# t0 = time.time()

# kde = stats.gaussian_kde(train, bw_method='silverman')
# estimates = kde(test)
# realest = f1(test)
# metrics = np.zeros((nrExp,3))
# metrics[i,0] = np.mean((estimates - realest)**2)       #MSE
# metrics[i,1] = np.mean(np.abs(estimates - realest))    #MAE
# metrics[i,2] = np.mean(-np.log(estimates))             #ANLL

#Generate data
data = syntheticData(d, sizeSample)
#test = syntheticData(d, sizeSampleTest)

errorMSE = np.zeros( (len(minSizeList), len(pList), len(nTreesList)))
errorMAE = np.zeros( (len(minSizeList), len(pList), len(nTreesList)))
errorANLL = np.zeros( (len(minSizeList), len(pList), len(nTreesList)))

t0 = time.time()
for _ in range(nrExp):
    #IID SAMPLE
    kf = KFold(n_splits = 10, random_state = 123, shuffle = True)
    for train_index, test_index in kf.split(range(sizeSample)):
        train = data[:,train_index]
        test = data[:,test_index]
        densityTest = f1(test)
        
        i = -1
        for minSize in minSizeList:
            i += 1
            j = -1
            for p in pList:
                j += 1
                k = -1
                for nTrees in nTreesList:
                    k += 1
                    
                    myForest = DensityRF(nTrees = nTrees, minSize = minSize, p = p)
                    myForest.fit(train)
                    preds = myForest.estimate(test)
                    
                    errorMSE[i,j,k] += np.mean((preds - densityTest)**2)
                    errorMAE[i,j,k] += np.mean(np.abs(preds - densityTest))
                    errorANLL[i,j,k] += np.mean(-np.log(preds))         
                    print(0)
                    
#Divide by numper of experiments and the number of folds (10)
errorMSE = errorMSE/(10*nrExp)
errorMAE = errorMAE/(10*nrExp)
errorANLL = errorANLL/(10*nrExp)
    
t1 = time.time()
print(t1-t0)

#ERROR KDE
data = syntheticData(10, sizeSample)
mseKDE = 0
maeKDE = 0
anllKDE = 0

for _ in range(nrExp):
    #IID SAMPLE
    kf = KFold(n_splits = 10, random_state = 123, shuffle = True)
    for train_index, test_index in kf.split(range(sizeSample)):
        train = data[:,train_index]
        test = data[:,test_index]
        densityTest = f1(test)
        
        kde = stats.gaussian_kde(train)
        estimates = kde(test)
        mseKDE += np.mean((estimates - densityTest)**2)/(10*nrExp)
        maeKDE += np.mean(np.abs(estimates - densityTest))/(10*nrExp)
        anllKDE += np.mean(-np.log(estimates))/(10*nrExp)
print(mseKDE)
print(maeKDE)
print(anllKDE)

# #PLOT ERROR
# mse = (preds - densityTest)**2
# mae = np.abs(preds - densityTest)
# anll = -np.log(preds)

# import matplotlib.pyplot as plt
# plt.plot(range(50), mse, label = "MSE")
# plt.plot(range(50), mae*20, label = "MAE*20")
# plt.plot(range(50), anll*20, label = "ANLL*20")
# plt.legend()
# plt.show()
    
# Check float division by zero
# t0 = time.time()
# myForest = DensityRF(nTrees = 100, minSize = 25, p = 0.5)
# myForest.fit(train)
# example = np.array([[-1,2,1,0.3,0.2], [-1,2,1,3,0.1]])
# preds = myForest.estimate(example.T)
# t1 = time.time()
# print(t1-t0)

    