# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:21:32 2021

@author: begoc
"""
import numpy as np
from RF_similarity import DensityRF
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score

# LOAD DATASET
my_data = np.genfromtxt('image_segmentation_uci.csv', delimiter=',')
data = my_data[:,:-1].T
labels_true = my_data[:,-1]

# FIT RF MODEL
myForest = DensityRF(nTrees = 50, minSize = 10, p = 0.5)
myForest.fit(data)
A = myForest.similarity(data)

# SPECTRAL CLUSTERING
clustering = SpectralClustering(n_clusters = 7 , affinity = 'precomputed').fit(A)
labels_pred = clustering.labels_
adjusted_rand_score(labels_pred, labels_true)

# K-MEANS COMPARISON
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7, random_state=0).fit(data.T)
adjusted_rand_score(kmeans.labels_, labels_true)
