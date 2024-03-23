#! /usr/bin/env python3
import numpy as np
from scipy.special import loggamma
from itertools import chain
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import f1_score as F1
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment

## Match distributions using the Hungarian algorithm
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
    return list(zip(row_ind, col_ind)), cost_matrix

## Computes the Kullback-Leibler divergence between two probability distributions
def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler divergence between two probability distributions.
    Inputs: p, q: Probability distributions
    Output: Kullback-Leibler divergence
    """
    return np.sum(p * np.log(p / q))

## Computes the Jensen-Shannon divergence between two probability distributions
def js_divergence(p, q):
    """
    Compute the Jensen-Shannon divergence between two probability distributions.
    Inputs: p, q: Probability distributions
    Output: Jensen-Shannon divergence
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

## Obtain permutation matrix from matching pairs of indices
def get_permutation_matrix(matching_pairs, n):
    """
    Obtain a permutation matrix from matching pairs of indices.
    Inputs: matching_pairs: List of tuples where each tuple contains indices of matched elements; n: Number of elements
    Output: Permutation matrix
    """
    perm_matrix = np.zeros((n, n))
    for pair in matching_pairs:
        perm_matrix[pair[0], pair[1]] = 1
    return perm_matrix

## Computers the Jensen-Shannon distance between multiple probability distributions
def js_divergence_multiple(prob_dists):
    """
    Compute the Jensen-Shannon distance between multiple probability distributions.
    Input: prob_dists: List of probability distributions
    Output: Jensen-Shannon distance
    """
    m = np.mean(prob_dists, axis=0)
    return np.sqrt(0.5 * np.sum([kl_divergence(p, m) for p in prob_dists]))

## Computes the Hellinger distance between two probability distributions
def hellinger_distance(p, q):
    """
    Compute the Hellinger distance between two probability distributions.
    Inputs: p, q: Probability distributions
    Output: Hellinger distance
    """
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))

## Computes the Bhattacharyya distance between two probability distributions
def bhattacharyya_distance(p, q):
    """
    Compute the Bhattacharyya distance between two probability distributions.
    Inputs: p, q: Probability distributions
    Output: Bhattacharyya distance
    """
    return -np.log(np.sum(np.sqrt(p * q)))

## Computes the total variation distance between two probability distributions
def total_variation_distance(p, q):
    """
    Compute the total variation distance between two probability distributions.
    Inputs: p, q: Probability distributions
    Output: Total variation distance
    """
    return 0.5 * np.sum(np.abs(p - q))

## Normalise a matrix by row or column
def normalise_matrix(matrix, axis=0):
    """
    Normalise a matrix by row or column.
    Inputs: matrix: Input matrix; axis: Axis along which to normalise the matrix (0 for column, 1 for row)
    Output: Normalised matrix
    """
    return matrix / np.sum(matrix, axis=axis, keepdims=True)

## Calculate the entropy of a probability distribution
def entropy(prob_dist):
    """
    Calculate the entropy of a probability distribution.
    Input: prob_dist: Probability distribution
    Output: Entropy
    """
    return -np.sum(prob_dist * np.log(prob_dist))

## Obtain permutation matrix from matching pairs of indices
def get_permutation_matrix(matching_pairs, n):
    """
    Obtain a permutation matrix from matching pairs of indices.
    Inputs: matching_pairs: List of tuples where each tuple contains indices of matched elements; n: Number of elements
    Output: Permutation matrix
    """
    perm_matrix = np.zeros((n, n))
    for pair in matching_pairs:
        perm_matrix[pair[0], pair[1]] = 1
    return perm_matrix

## Computes the Jaccard similarity between two lists
def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

## Computes the Sorensen similarity between two lists
def sorensen(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    return float(2 * intersection) / (len(list(set(list1))) + len(list(set(list2))))

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