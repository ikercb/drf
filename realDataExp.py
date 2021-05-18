# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 13:12:06 2021

@author: ikerc
"""
import numpy as np
from sklearn.model_selection import KFold
from RF_cython import DensityRF
import time

nrExp = 1
nTreesList = [25, 50, 100, 200]
minSizeList = [5, 10, 25, 50]
pList = [0.2*i for i in range(6)]

#Read dataset
data = np.genfromtxt('ionosphere.csv',delimiter=',').T
#data = np.genfromtxt('whitewine.csv',delimiter=',').T
sizeSample = data.shape[1]

errorANLL = np.zeros((len(minSizeList), len(pList), len(nTreesList)))

t0 = time.time()
for _ in range(nrExp):
    #IID SAMPLE
    kf = KFold(n_splits = 10, random_state = 123, shuffle = True)
    for train_index, test_index in kf.split(range(sizeSample)):
        train = data[:,train_index]
        test = data[:,test_index]
    
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
                    
                    errorANLL[i,j,k] += np.mean(-np.log(preds))         
                    print(0)
                    
#Divide by numper of experiments and the number of folds (10)
errorANLL = errorANLL/(10*nrExp)
    
t1 = time.time()
print(t1-t0)


#KDE
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
        anllKDE += np.mean(-np.log(estimates))/(10*nrExp)
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
    
# # Check float division by zero
# t0 = time.time()
# myForest = DensityRF(nTrees = 200, minSize = 5, p = 0.5)
# myForest.fit(train)
# preds = myForest.estimate(test)
# t1 = time.time()
# print(t1-t0)
    