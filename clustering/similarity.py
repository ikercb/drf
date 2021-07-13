# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 18:16:03 2021

@author: begoc
"""

import numpy as np

tree_paths = np.array([[0, 1, 3],
                       [0, 2, -10],
                       [0, 1, 4],
                       [0, 2, -20],
                       [0, 1, 3]])
path_lengths = np.array([[3, 2, 3, 2, 3]]).T

P = np.kron(tree_paths, np.ones((5,1)))
Q = np.tile(tree_paths, (5,1))

P_len = np.kron(path_lengths, np.ones((5,1)))
Q_len = np.tile(path_lengths, (5,1))
lens = np.vstack((P_len.T, Q_len.T))

max_len = np.amax(lens, axis = 0) - 1

shared_paths_idx = P == Q;
shared_paths = np.sum(shared_paths_idx, 1) - 1;

similarity = shared_paths/max_len
sim_matrix = np.reshape(similarity, (5,5))
# Solve problem in the diagonal with leafs that does not reach maximum depth
np.fill_diagonal(sim_matrix, 1)

