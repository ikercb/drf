# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 12:02:27 2021

@author: begoc
"""
import numpy as np
from RF_similarity import DensityRF
from sklearn.cluster import SpectralClustering

# GENERATE DATA: 2 BIVARIATE GAUSSIANS

mean1 = [0, 0]
cov1 = [[1, 0], [0, 2]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 100).T

mean2 = [12, 5]
cov2 = [[2, 0], [0, 1]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 100).T

x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))
data = np.vstack((x,y))

# plt.plot(x, y, 'x')
# plt.axis('equal')
# plt.show()

# FIT RF MODEL
myForest = DensityRF(nTrees = 50, minSize = 10, p = 0.5)
myForest.fit(data)
A = myForest.similarity(data)

# SPECTRAL CLUSTERING
clustering = SpectralClustering(n_clusters = 2 , affinity = 'precomputed').fit(A)
clustering.labels_