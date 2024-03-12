#! /usr/bin/env python3
import numpy as np
from scipy.special import loggamma
from itertools import chain
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import f1_score as F1
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment

def match_distributions(set1, set2, distance_func):
    """
    Match elements from two sets of probability distributions using the Hungarian algorithm.
    Inputs: set1, set2: Sets of probability distributions; distance_func: Function to compute the distance between two distributions
    Output: List of tuples where each tuple contains indices of matched elements (from set1 to set2)
    """
    # Calculate the cost matrix based on the distance function
    cost_matrix = np.array([[distance_func(d1, d2) for d2 in set2] for d1 in set1])
    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Return the matching pairs
    return list(zip(row_ind, col_ind))

## Computes logarithm of the multivariate beta function
def logB(vec):
    out = np.sum([loggamma(v) for v in vec])
    out -= loggamma(np.sum(vec))
    return out

## Adjusted Rand Index for t
def ari_t(t_true, t_est):
	return ari(t_true, t_est)

## Adjusted Rand Index for s
def ari_s(s_true, s_est, crosstab_out=False):
	chain_true = list(chain.from_iterable(s_true.values()))
	chain_est = list(chain.from_iterable(s_est.values()))
	if crosstab_out:
		return ari(chain_true, chain_est), pd.crosstab(np.array(chain_true), np.array(chain_est))
	else:
		return ari(chain_true, chain_est)

## F1 scores for z
def F1_Z(z_true, z_est):
	chain_true = []; chain_est = []
	for d in z_est:
		chain_true += list(chain.from_iterable(z_true[d].values()))
		chain_est += list(chain.from_iterable(z_est[d].values()))
	# Crosstab
	ct = pd.crosstab(np.array(chain_true), np.array(chain_est))
	return F1(chain_true, chain_est), ct

## Estimate communities using hierarchical clustering on the posterior similarity matrix
def estimate_t(q,m,K,linkage='average'):
    import numpy as np
    ## Scaled posterior similarity matrix
    psm = np.zeros((m.D,m.D))
    for i in range(q['t'].shape[0]):
        psm += np.equal.outer(q['t'][i],q['t'][i])
    ## Posterior similarity matrix (estimate)
    psm /= q['t'].shape[0]
    ## Clustering based on posterior similarity matrix (hierarchical clustering)
    cluster_model = AgglomerativeClustering(n_clusters=K, metric='precomputed', linkage=linkage) 
    clust = cluster_model.fit_predict(1-psm)
    return clust

## Estimate communities using hierarchical clustering on the posterior similarity matrix
def estimate_s(q,m,K,linkage='average'):
    import numpy as np
    ## Scaled posterior similarity matrix
    for i in range(m.D):
        try:
            v = np.hstack((v, q['s'][i]))
        except:
            v = q['s'][i]
    psm = np.zeros((v.shape[1], v.shape[1]))
    for i in range(v.shape[0]):
        psm += np.equal.outer(v[i], v[i])
    ## Posterior similarity matrix (estimate)
    psm /= v.shape[0]
    ## Clustering based on posterior similarity matrix (hierarchical clustering)
    cluster_model = AgglomerativeClustering(n_clusters=K, metric='precomputed', linkage=linkage) 
    clust = cluster_model.fit_predict(1-psm)
    nn = 0
    clust_dict = {}
    for i in range(m.D):
        clust_dict[i] = clust[nn:(nn+m.N[i])]
        nn += m.N[i]
    return clust_dict