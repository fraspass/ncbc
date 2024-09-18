#! /usr/bin/env python3
import numpy as np
from collections import Counter
from scipy.special import logsumexp, loggamma
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from .utils import logB
from IPython.display import display, clear_output

class parent_child_model:
    
    # The class can be used to fit the Anchored Nested Bayesian Clustering Model discussed in:
    # Sanna Passino, F., Mantziou, A., Ghani, D., Thiede, P., Bevington, R. and Heard, N.A.
    # "NESTED DIRICHLET MODELS FOR UNSUPERVISED ATTACK PATTERN DETECTION IN HONEYPOT DATA"
    # Required input: W_a - dictionary of dictionaries containing the parent words (as consecutive integers starting at 0)
    #                 W_c - dictionary of dictionaries containing the child words (as consecutive integers starting at 0)
    def __init__(self, W_a, W_c, K, H, V=0, gamma=1.0, chi=1.0, tau=1.0, eta=1.0, numpyfy=False):
        # Documents & sentences (sessions & commands) in python dictionary form
        self.w_a = W_a
        if not numpyfy:
            self.w_c = W_c
        else:
            self.w_c = {}
            for d in W_c:
                self.w_c[d] = {}
                for s in W_c[d]:
                    self.w_c[d][s] = np.array(W_c[d][s])
        # Number of documents
        if len(W_a) != len(W_c):
            raise ValueError('The number of documents in W_a and W_c must be the same.')
        else:
            self.D = len(W_a)
        # Length of each document
        N_a = np.array([len(self.w_a[d]) for d in self.w_a])
        N_c = np.array([len(self.w_c[d]) for d in self.w_c])
        if not np.array_equal(N_a, N_c):
            raise ValueError('The number of sentences in W_a and W_c must be the same for each document.')
        else:
            self.N = np.copy(N_a)
            self.N_cumsum = np.cumsum(self.N)
            self.N_cumsum0 = np.append(0,self.N_cumsum)
        # Calculate the number of child words for each document
        self.M = {}
        for d in self.w_c:
            self.M[d] = [len(self.w_c[d][command]) for command in self.w_c[d]]
        # Prior parameters
        if isinstance(gamma, float) or isinstance(gamma, int):
            self.gamma = gamma
            if not self.gamma > 0:
                raise ValueError('The prior parameters gamma must be positive.')
        else: 
            raise TypeError('The prior parameter gamma must be a float or integer.')
        if isinstance(eta, float) or isinstance(eta, int):
            self.eta = eta
            if not self.eta > 0:
                raise ValueError('The prior parameters eta must be positive.')
        else: 
            raise TypeError('The prior parameter eta must be a float or integer.')
        if isinstance(tau, float) or isinstance(tau, int):
            self.tau = tau
            if not self.tau > 0:
                raise ValueError('The prior parameters tau must be positive.')
        else: 
            raise TypeError('The prior parameter tau must be a float or integer.')
        if isinstance(chi, float) or isinstance(chi, int):
            self.chi = chi
            if not self.chi > 0:
                raise ValueError('The prior parameters chi must be positive.')
        else: 
            raise TypeError('The prior parameter chi must be a float or integer.')
        # Calculate observed vocabulary size if necessary
        if V > 0:
            self.V = V
        else:
            VV = Counter()
            for d in W_c:
                for j in W_c[d]:
                    VV += Counter(W_c[d][j])
            for d in W_a:
                VV += Counter(W_a[d])
            self.V = len(VV)
        # Check if the provided value for K is appropriate
        if not isinstance(K, int) or K < 2:
            raise ValueError('K must be an integer value larger or equal to 2.') 
        self.K = K
        # Check if the provided value for H is appropriate
        if not isinstance(H, int) or H < 2:
            raise ValueError('H must be an integer value larger or equal to 2.') 
        self.H = H
        # Initialise dictionaries
        self.t = np.zeros(self.D, dtype=int)
        self.u = np.zeros(self.V, dtype=int)

    ## Calculate marginal posterior
    def marginal_loglikelihood(self):
        ll = 0
        ll += logB(self.gamma + self.T) - logB(self.gamma * np.ones(self.K))
        ll += logB(self.chi + self.U) - logB(self.chi * np.ones(self.H))
        ll += np.sum([logB(self.tau + self.W_a[k]) for k in range(self.K)]) - self.K * logB(self.tau * np.ones(self.V))
        ll += np.sum([logB(self.eta + self.W_c[h]) for h in range(self.H)]) - self.H * logB(self.eta * np.ones(self.V))
        return ll

    ## Initialise counts given initial values of t, s and z
    def init_counts(self):
        # Session-level topics
        self.T = np.zeros(self.K, dtype=int)
        # Chain-level topics
        self.U = np.zeros(self.H, dtype=int)
        # Obtain W_a and W_c
        self.W_a = np.zeros(shape=(self.K, self.V), dtype=int)   
        self.W_c = np.zeros(shape=(self.H, self.V), dtype=int)          
        # Initialise quantities 
        Q_t = Counter(self.t)
        Q_u = Counter(self.u)
        for topic in Q_t:
            self.T[topic] += Q_t[topic]
        for topic in Q_u:
            self.U[topic] += Q_u[topic]
        # Obtain W_a and W_c
        for doc in self.w_a:
            td = self.t[doc]
            for j in range(self.N[doc]):
                parent = self.w_a[doc][j]
                self.W_a[td, parent] += 1
                for w in self.w_c[doc][j]:
                    self.W_c[self.u[parent], w] += 1

    ## Initialise from other topic model object
    def init_from_other(self, other):
        ## Initialise dictionary from previous model
        self.V = other.V
        ## Incorporate information within prior for t
        if other.T.shape != (self.K,):
            raise ValueError('The dimension of T in the other model is not compatible with the current model.')
        else:
            self.T = other.gamma + other.T
        ## Incorporate information within prior for u
        if other.U.shape != (self.H,):
            raise ValueError('The dimension of U in the other model is not compatible with the current model.')
        else:
            self.U = other.chi + other.U
        ## Incorporate information within prior for W_a
        if other.W_a.shape != (self.K, self.V):
            raise ValueError('The dimension of W_a in the other model is not compatible with the current model.')
        else:
            self.W_a = other.tau + other.W_a
        ## Incorporate information within prior for W_c
        if other.W_c.shape != (self.H, self.V):
            raise ValueError('The dimension of W_c in the other model is not compatible with the current model.')
        else:
            self.W_c = other.eta + other.W_c

    ## Initializes chain at given values of t, s and z
    def custom_init(self, t, u):
        # Initialise t from custom initial value
        if isinstance(t, list) or isinstance(t, np.ndarray):
            if len(t) != self.D:
                raise TypeError('The initial value for t should be a D-dimensional list or np.ndarray.')
            self.t = np.array(t, dtype=int)
        else:
            raise TypeError('The initial value for t should be a K-dimensional list or np.ndarray.')
        # Initialise u from custom initial value
        if isinstance(u, list) or isinstance(u, np.ndarray):
            if len(u) != self.V:
                raise TypeError('The initial value for u should be a V-dimensional list or np.ndarray.')
            self.u = np.array(u, dtype=int)
        # Initialise counts
        self.init_counts()     

    ## Initializes uniformly at random
    def random_init(self, K_init=None, H_init=None):
        if K_init is not None:
            if not isinstance(K_init, int) or K_init < 1:
                raise ValueError('K_init must be an integer value larger or equal to 1.') 
        else:
            K_init = int(np.copy(self.K))
        if H_init is not None:
            if not isinstance(H_init, int) or H_init < 1:
                raise ValueError('H_init must be an integer value larger or equal to 1.')
        # Random initialisation
        self.t = np.random.choice(K_init, size=self.D)
        self.u = np.random.choice(H_init, size=self.V)
        ## Initialise counts
        self.init_counts()

    ## Initializes chain using spectral clustering  
    def spectral_init(self, K_init=None, H_init=None, random_state=0):
        # Check if the provided value for K is appropriate
        if K_init is not None:
            if not isinstance(K_init, int) or K_init < 1:
                raise ValueError('K_init must be an integer value larger or equal to 1.') 
        else:
            K_init = int(np.copy(self.K))
        # Check if the provided value for H is appropriate
        if H_init is not None:
            if not isinstance(H_init, int) or H_init < 1:
                raise ValueError('H_init must be an integer value larger or equal to 1.') 
        else:
            H_init = int(np.copy(self.H))
        # Build co-occurrence matrix for parent words
        cooccurrence_matrix = {}
        for d in self.w_a:
            cooccurrence_matrix[d] = Counter(self.w_a[d])
        # Obtain matrix
        vals = []; rows = []; cols = []
        for key in cooccurrence_matrix:
            vals += list(cooccurrence_matrix[key].values())
            rows += [key] * len(cooccurrence_matrix[key])
            cols += list(cooccurrence_matrix[key].keys())
        # Co-occurrence matrix
        cooccurrence_matrix = coo_matrix((vals, (rows, cols)), shape=(self.D, self.V))
		# Spectral decomposition of A
        U, S, _ = svds(cooccurrence_matrix.asfptype(), k=K_init)
        kmod = KMeans(n_clusters=K_init, random_state=random_state).fit(U[:,::-1] * (S[::-1] ** .5))
        self.t = kmod.labels_
        # Build co-occurrence matrix for child words
        cooccurrence_matrix = {}
        for v in range(self.V):
            cooccurrence_matrix[v] = Counter()
        for d in self.w_c:
            for j in range(self.N[d]):
                cooccurrence_matrix[self.w_a[d][j]] += Counter(self.w_c[d][j])
        # Obtain matrix
        vals = []; rows = []; cols = []
        for key in cooccurrence_matrix:
            vals += list(cooccurrence_matrix[key].values())
            rows += [key] * len(cooccurrence_matrix[key])
            cols += list(cooccurrence_matrix[key].keys())
        # Co-occurrence matrix
        cooccurrence_matrix = coo_matrix((vals, (rows, cols)), shape=(self.V, self.V))
        ## Spectral decomposition
        U, S, _ = svds(cooccurrence_matrix.asfptype(), k=H_init)
        kmod = KMeans(n_clusters=K_init, random_state=random_state).fit(U[:,::-1] * (S[::-1] ** .5))
        self.u = kmod.labels_     
        # Initialise counts
        self.init_counts()     

   ## Resample parent-level topics
    def resample_parent_topics(self, size=1, indices=None):
        # Optional input: subset - list of integers in {0,1,...,D-1}
        if indices is None:
            indices = np.random.choice(self.D, size=size)
        # Resample each document
        for d in indices:
            td_old = self.t[d]
            # Remove counts
            self.T[td_old] -= 1
            Wd = Counter(self.w_a[d])
            for v in Wd:
                self.W_a[td_old,v] -= Wd[v]
            # Calculate allocation probabilities
            probs = np.log(self.gamma + self.T)
            for v in Wd:
                probs += np.sum(np.log(np.add.outer(self.tau + self.W_a[:,v], np.arange(Wd[v]))), axis=1)
            probs -= np.sum(np.log(np.add.outer(np.sum(self.tau + self.W_a, axis=1), np.arange(np.sum(list(Wd.values()))))), axis=1)
            # Transform the probabilities
            probs = np.exp(probs - logsumexp(probs))
            # Resample session-level topic
            td_new = np.random.choice(len(probs), p=probs)
            self.t[d] = td_new
            # Update counts
            self.T[td_new] += 1
            for v in Wd:
                self.W_a[td_new,v] += Wd[v]

    ## Resample child-level topics
    def resample_child_topics(self, size=1, indices=None):
        if indices is None:
            indices = np.random.choice(self.V, size=size)
        for v in indices:
            uv_old = self.u[v]
            # Remove counts
            self.U[uv_old] -= 1
            Wv = Counter()
            for d in self.w_c:
                for j in range(self.N[d]):
                    if self.w_a[d][j] == v:
                        Wv += Counter(self.w_c[d][j])
            for vv in Wv:
                self.W_c[uv_old,vv] -= Wv[vv]
            # Calculate allocation probabilities
            probs = np.log(self.chi + self.U[uv_old])
            for vv in Wv:
                probs += np.sum(np.log(np.add.outer(self.eta + self.W_c[:,vv], np.arange(Wv[vv]))), axis=1)
            probs -= np.sum(np.log(np.add.outer(np.sum(self.eta + self.W_c, axis=1), np.arange(np.sum(list(Wv.values()))))), axis=1)
            # Transform the probabilities
            probs = np.exp(probs - logsumexp(probs))
            # Resample command-level topic
            u_new = np.random.choice(len(probs), p=probs)
            self.u[v] = u_new
            # Update counts
            self.U[u_new] += 1
            for vv in Wv:
                self.W_c[u_new,vv] += Wv[vv]

    ## Runs MCMC chain
    def MCMC(self, iterations, burnin=0, size=1, verbose=True, calculate_ll=False, jupy_out=False, return_t=True, return_u=True, thinning=1):
        # Moves
        moves = ['t','u']
        moves_probs = [1,1]
        moves_probs /= np.sum(moves_probs)
        ## Marginal posterior
        if calculate_ll:
            ll = []
        ## Return output
        Q = int(iterations // thinning)
        if return_t:
            t_out = np.zeros((Q,self.D),dtype=int)
        if return_u:
            u_out = np.zeros((Q,self.V),dtype=int)
        ## Run MCMC
        for it in range(iterations+burnin):
            # Sample move
            move = np.random.choice(moves, p=moves_probs)
            # Do move
            if move == 't':
                self.resample_parent_topics(size=size)
            else:
                self.resample_child_topics(size=size)
            if calculate_ll:
                ll += [self.marginal_loglikelihood()]
            # Print progression
            if verbose:
                if it < burnin:
                    if jupy_out:
                        clear_output(wait=True)
                        display('Burnin: ' + str(it+1) + ' / ' + str(burnin)) 
                    else:
                        print('\rBurnin: ', str(it+1), ' / ', str(burnin), sep='', end=' ', flush=True)
                elif it == burnin and burnin > 0:
                    if jupy_out:
                        clear_output(wait=True)
                        display('Progression: ' + str(it-burnin+1) + ' / ' + str(iterations)) 
                    else:
                        print('\nProgression: ', str(it-burnin+1), ' / ', str(iterations), sep='', end=' ', flush=True)
                else:
                    if jupy_out:
                        clear_output(wait=True)
                        display('Progression: ' + str(it-burnin+1) + ' / ' + str(iterations))         
                    else:
                        print('\rProgression: ', str(it-burnin+1), ' / ', str(iterations), sep='', end=' ', flush=True)
            ## Store output
            if it >= burnin and (it - burnin) % thinning == 0:
                q = (it - burnin) // thinning
                if return_t:
                    t_out[q] = np.copy(self.t)
                if return_u:
                    u_out[q] = np.copy(self.u)
        ## Output
        out = {}
        if calculate_ll:
            out['loglik'] = ll
        if return_t:
            out['t'] = t_out
        if return_u:
            out['u'] = u_out
        ## Return output
        return out