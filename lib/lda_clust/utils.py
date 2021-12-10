#! /usr/bin/env python3
import numpy as np
from scipy.special import loggamma
from itertools import chain
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import f1_score as F1
import pandas as pd

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