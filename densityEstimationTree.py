# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:11:35 2021

@author: ikerc
"""
import numpy as np

class Test:
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def match(self, example):
        val = example[self.feature]
        return val >= self.value
        
class Node:
    def __init__(self, test, left, right):
        self.test = test
        self.left = left
        self.right = right
        
class Leaf:
    def __init__(self, data, domain, N):
        self.density = densityEstimation(data, domain, N)
        
def densityEstimation(data, domain, N):
    vol = volume(domain)
    return len(data)/(vol*N)

def volume(domain):
    vol = 1
    for i in range(len(domain)):
        vol *= domain[i][1] - domain[i][0]
    return vol

def partition(data, test):
    left, right = [], []
    for row in data:
        if test.match(row):
            left.append(row)
        else:
            right.append(row)
    return left, right

def bestSplit(data, minSize, domain, N):
    bigReduction = 0
    bestTest = None
    currentError = error(data, domain, N)
    nFeatures = len(data[0])
    
    for col in range(nFeatures):
        values = set([row[col] for row in data])
        for val in values:
            test = Test(col, val)
            left, right = partition(data, test)
            if len(left) < minSize or len(right) < minSize:
                continue
            domainLeft, domainRight = newDomains(domain, col, val)
            reduction = currentError - error(left, domainLeft, N) - error(right, domainRight, N)
            if reduction >= bigReduction:
                bigReduction, bestTest = reduction, test
    return bigReduction, bestTest

def newDomains(domain, feature, value):
    domainLeft = np.copy(domain)
    domainLeft[feature, 0] = value
    domainRight = np.copy(domain)
    domainRight[feature, 1] = value
    return domainLeft, domainRight
    
def error(data, domain, N):
    vol = volume(domain)
    return -len(data)**2/(vol*N**2)
    
def initialDomain(data):
    nFeatures = len(data[0])
    domain = np.zeros((nFeatures,2))
    for feat in range(nFeatures):
        domain[feat, 0] = min(x[feat] for x in data)
        domain[feat, 1] = max(x[feat] for x in data)
    return domain
    
def buildTree(data, minSize, domain = None, N = None):
    if domain is None:
        domain = initialDomain(data)
    if N is None:
        N = len(data)
        
    errorReduction, test = bestSplit(data, minSize, domain, N)
    if errorReduction == 0:
        return Leaf(data, domain, N)
    else:
        left, right = partition(data, test)
        domainLeft, domainRight = newDomains(domain, test.feature, test.value)
    
        leftBranch = buildTree(left, minSize, domainLeft, N)
        rightBranch = buildTree(right, minSize, domainRight, N)
    
        return Node(test, leftBranch, rightBranch)
    
def estimate(node, example):
    if isinstance(node, Leaf):
        return node.density

    if node.test.match(example):
        return estimate(node.left, example)
    else:
        return estimate(node.right, example)

v = np.random.normal(0,1,1000)
data = []
for i in range(1000):
    data.append( [v[i]] )

minSize = 20
myTree = buildTree(data, minSize)  
#Plots
import matplotlib.pyplot as plt
n = 1000
x = np.linspace(-2,2,n)
preds = np.zeros(n)
for i in range(n):
    preds[i] = estimate(myTree, [x[i]])
    
plt.plot(x, 1/(np.sqrt(2 * np.pi)) * np.exp(-x**2 /2), linewidth=3, color='r')
plt.step(x, preds)
plt.show()

    





    