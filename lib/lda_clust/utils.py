#! /usr/bin/env python3
import numpy as np
from scipy.special import loggamma

## Computes logarithm of the multivariate beta function
def logB(vec):
    out = 0
    for v in vec:
        out += loggamma(v)
    out -= loggamma(np.sum(vec))
    return out
