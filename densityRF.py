# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:39:31 2021

@author: ikerc
"""
import numpy as np

class DensityRF:
    
    def __init__(self, nTrees = 20, minSize = 2, p = 0.1, bootstrapping = True):
        self.nTrees = nTrees
        self.minSize = minSize
        self.p = p
        self.bootstrapping = bootstrapping
        self.rf = None
        self.N = None
        self.nFeatures = None
        
    def fit(self, data):
        #Data.shape = (nFeatures, N)
        self.nFeatures, self.N = data.shape
        forest = []
        for _ in range(self.nTrees):
            if self.bootstrapping == True:
                bData = self.bootstrapSample(data)
                forest.append( self.buildTree(bData) )
            else:
                forest.append( self.buildTree(data) )
        self.rf = forest
        
    def estimate(self, data):
        _, n = data.shape
        preds = np.zeros(n)
        for i in range(self.nTrees):
            for j in range(n):
                preds[j] += self.densityEstimation(self.rf[i], data[:,j])
        return preds/self.nTrees
        
    class Test:
        def __init__(self, feature, value):
            self.feature = feature
            self.value = value
    
        def match(self, example):
            val = example[self.feature]
            return val > self.value

    class Node:
        def __init__(self, test, left, right):
            self.test = test
            self.left = left
            self.right = right
        
    class Leaf:
        def __init__(self, data, volume, N):
            self.volume = volume
            self.N = N
            self.density = self.leafDensity(data)
        
        def leafDensity(self, data):
            return data.shape[1]/(self.volume*self.N)
            
    def buildTree(self, data, domain = None):
        if domain is None:
            domain = self.initialDomain(data)

        if np.shape(data)[1] < self.minSize:
            return self.Leaf(data, self.volume(domain), self.N)
        
        if np.random.uniform(0,1) < self.p:
            # Random split
            validSplit = False
            while not validSplit:
                test = self.randomSplit(data, domain)
                left, right = self.partition(data, test)
                if np.shape(left)[1] != 0 and np.shape(right)[1] != 0:
                    validSplit = True
        else:
            # Impurity criterion
            test, errorReduction = self.bestSplit(data, domain)
            if errorReduction == 0:
                return self.Leaf(data, self.volume(domain), self.N)
            else:
                left, right = self.partition(data, test)
                    
        # Recursively build the left branch.
        domainLeft = np.copy(domain)
        domainLeft[test.feature, 0] = test.value
        leftBranch = self.buildTree(left, domainLeft)
    
        # Recursively build the right branch.
        domainRight = np.copy(domain)
        domainRight[test.feature, 1] = test.value
        rightBranch = self.buildTree(right, domainRight)
    
        return self.Node(test, leftBranch, rightBranch)
      
    def randomSplit(self, data, domain):
        feature = np.random.randint(0, self.nFeatures)
        value = np.random.uniform(domain[feature,0], domain[feature,1])
        return self.Test(feature, value)
    
    def bestSplit(self, data, domain):
        bigReduction = 0
        bestTest = None
        currentError = self.error(data, domain)
        nFeatures, nData = data.shape
        for col in range(nFeatures):
            values = data[col,:]
            for val in values:
                #Do not consider extremes of the interval
                if val == domain[col,0] or val == domain[col,1]:
                    continue
                test = self.Test(col, val)
                left, right = self.partition(data, test)
                nLeft = left.shape[1]
                nRight = right.shape[1]
                #if nLeft < 2 or nRight < 2:
                #    continue
                domainLeft, domainRight = self.newDomains(domain, col, val)
                reduction = currentError - self.error(left, domainLeft) - self.error(right, domainRight)
                if reduction >= bigReduction:
                    bigReduction, bestTest = reduction, test
        return bestTest, bigReduction
    
    def error(self, data, domain):
        vol = self.volume(domain)
        return -data.shape[1]**2/(vol*self.N**2)
    
    def volume(self, domain):
        vol = 1
        for i in range(domain.shape[0]):
            vol *= domain[i,1] - domain[i,0]
        return vol
    
    def initialDomain(self, data):
        domain = np.zeros((self.nFeatures,2))
        domain[:,0] = np.amin(data, axis = 1)
        domain[:,1] = np.amax(data, axis = 1)
        return domain
    
    def newDomains(self, domain, feature, value):
        domainLeft = np.copy(domain)
        domainLeft[feature, 0] = value
        domainRight = np.copy(domain)
        domainRight[feature, 1] = value
        return domainLeft, domainRight
    
    def partition(self, data, test):
        cond = data[test.feature,:] > test.value
        left = data[:,cond]
        right = data[:,~cond]
        return left, right
    
    def densityEstimation(self, node, example):
        if isinstance(node, self.Leaf):
            return node.density
        if node.test.match(example):
            return self.densityEstimation(node.left, example)
        else:
            return self.densityEstimation(node.right, example)
        
    def bootstrapSample(self, data):
        idx = np.random.choice(self.N, self.N, replace = True)
        return data[:,idx]
        
# =============================================================================
# #TEST
# v = np.random.normal(0,1,1000)
# data = np.zeros((1,1000))
# data[0,:] = v
# 
# myForest = densityRF(nTrees = 100, minSize = 50)
# myForest.fit(data)
# xg = np.linspace(-2,2,100)
# x = np.zeros((1,100))
# x[0,:] = xg
# preds = myForest.estimate(x)
# plt.plot(xg, 1/(np.sqrt(2 * np.pi)) * np.exp(-xg**2 /2), linewidth=3, color='r')
# plt.step(xg, preds)
# plt.show()        
# =============================================================================
