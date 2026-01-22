# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)

import numpy as np

def get_hop_distance(num_node, edge, max_hop=1):
    '''
    Compute the hop distance matrix for a given graph.
    '''
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node), dtype=np.float32) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    '''
    Normalize the adjacency matrix of a directed graph.
    This prevents numerical values from exploding during nn operations.
    '''
    Dl = np.sum(A, axis=0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node), dtype=np.float32)
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return A @ Dn

def normalize_undigraph(A):
    '''
    Normalize the adjacency matrix of an undirected graph.
    This scale balances the influence between nodes, 
    considering both those who send and those who receive information
    '''
    Dl = np.sum(A, axis=0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node), dtype=np.float32)
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    return Dn @ A @ Dn