#! /usr/bin/env python3
import numpy as np
from scipy.special import gammaln
from collections import Counter
from scipy.special import logsumexp

class topic_model:
    # The class can be used to fit one of the topic models discussed in:
    # _authornames_
    # "Topic modelling of command lines for attack pattern detection in cyber-security"
    # Required input: W - dictionary of dictionaries containing the words (as consecutive integers starting at 0)

    def __init__(self, W, K, fixed_K = True, H=0, fixed_H = True, V=0, fixed_V = True, 
                    secondary_topic = True, command_level_topics = True,
                    gamma=1.0, eta=1.0, alpha=1.0, alpha0=1.0, tau=1.0):
        # Documents & sentences (sessions & commands) in python dictionary form
        self.w = W
        # Number of documents
        self.D = len(W)
        # Length of each document
        self.N = np.array([len(self.w[d]) for d in self.w])
        self.M = {}
        for d in self.w:
            self.M[d] = [len(command) for command in self.w[d]]
        # Determine V is fixed or unbounded - if fixed, determine if it is given as input
        if not isinstance(fixed_V, bool):
            raise TypeError('fixed_V must be True or False.')
        self.fixed_V = fixed_V
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
        # Calculate observed vocabulary size if necessary
        if V > 0:
            self.V = V
        else:
            self.V = 0
            for d in W:
                for i in range(self.N[d]):
                    v = self.w[d][i]
                    if v > self.V:
                        self.V = v
            self.V += 1
        # Secondary topics
        if not isinstance(secondary_topic, bool):
            raise TypeError('secondary_topic must be True or False.')
        else:
            if secondary_topic:
                self.secondary_topic = True
            else:
                self.secondary_topic = False
            if isinstance(alpha, float) or isinstance(alpha, int):
                self.alpha = alpha
                if not self.alpha > 0:
                    raise ValueError('The prior parameters alpha must be positive.')
                if isinstance(alpha0, float) or isinstance(alpha0, int):
                    self.alpha0 = alpha0
                    if not self.alpha0 > 0:
                        raise ValueError('The prior parameters alpha0 must be positive.')
                else:
                    raise TypeError('The prior parameter alpha0 must be a float or integer.')
            else: 
                raise TypeError('The prior parameter alpha must be a float or integer.')
        # Command-level topics
        if not isinstance(command_level_topics, bool):
            raise ValueError('command_level_topics must be True or False.')
        else:
            if command_level_topics:
                self.command_level_topics = True
            else:
                self.command_level_topics = False
            if isinstance(tau, float) or isinstance(tau, int):
                self.tau = tau
                if not self.tau > 0:
                    raise ValueError('The prior parameters tau must be positive.')
            else:
                raise TypeError('The prior parameter tau must be a float or integer.')
        # Check if the provided value for K is appropriate
        if not isinstance(K, int) or K < 2:
            raise ValueError('K must be an integer value larger or equal to 2.') 
        self.K = K
        # Fixed or unbounded K
        if not isinstance(fixed_K, bool):
            return TypeError('fixed_K must be True or False.')
        self.fixed_K = fixed_K
        # Check if the provided value for H is appropriate
        if not isinstance(H, int) or (self.command_level_topics and H < 2):
            raise ValueError('H must be an integer value larger or equal to 2 if command-level topics are used.') 
        # If the number of command-level topics is initialiised but the model does not use command-level topics, return an error
        if H > 0 and not self.command_level_topics:
            raise ValueError('H can only be specified when command-level topics are used. Proposed solution: initialise H=0.')
        self.H = H
        # Fixed or unbounded H
        if not isinstance(fixed_H, bool):
            return TypeError('fixed_H must be True or False.')
        self.fixed_H = fixed_H
        # Session-level topics
        self.t = np.zeros(self.D, dtype=int)
        self.T = np.zeros(self.K, dtype=int)
        # Command-level topics
        if self.command_level_topics:
            self.s = {}
            for d in self.w:
                self.s[d] = np.zeros(len(self.w[d]), dtype=int)
            self.S = np.zeros(self.K, self.H, dtype=int)
            self.W = np.zeros(shape=(self.H + (1 if self.secondary_topics else 0), self.V), dtype=int)
        else:
            self.W = np.zeros(shape=(self.K + (1 if self.secondary_topics else 0), self.V), dtype=int)             
        # Primary-secondary topic indicators
        if self.secondary_topic:
            self.z = {}
            for d in self.w:
                for j in self.w[d]:
                    self.z[j,d] = np.zeros(len(self.w[d][j]), dtype=int)
            if self.command_level_topics:
                self.M_star = np.zeros(shape=self.H, dtype=int)
                self.Z = np.zeros(shape=self.H, dtype=int)
            else:
                self.M_star = np.zeros(shape=self.K, dtype=int)
                self.Z = np.zeros(shape=self.K, dtype=int)

   ## Resample session-level topics
    def resample_session_topics(self, size=1, indices=None):
        ## Optional input: subset - list of integers d
        if indices is None:
            indices = np.random.choice(self.D, size=size)
        # Resample each document
        for d in indices:
            td_old = self.t[d]  
            # Remove counts
            self.T[td_old] -= 1
            if self.command_level_topics:
                Sd = Counter(self.s[d])
                for h in Sd:
                    self.S[td_old,h] -= Sd[h]
            else:
                Wd = Counter()
                if self.secondary_topic:
                    Zd = 0
                    self.M_star[td_old] -= np.sum(self.M[d])
                for j in self.w[d]:
                    if self.secondary_topic:
                        Zdj = self.z[d][j]
                        Wd += Counter(self.w[d][j][Zdj == 1])
                        Z_partial = np.sum(Zdj)
                        self.Z[td_old] -= Z_partial
                        Zd += Z_partial
                    else:
                        Wd += Counter(self.w[d][j])
                for v in Wd:
                    self.W[td_old + (1 if self.secondary_topic else 0),v] -= Wd[v]
            # Calculate allocation probabilities
            probs = np.log(self.gamma + self.T)
            if self.command_level_topics:
                if self.secondary_topic:
                    for h in Sd:
                        probs += np.sum(np.log(np.add.outer(self.eta + self.S[1:,h], np.arange(Sd[h]))), axis=1)
                    probs -= np.sum(np.log(np.add.outer(np.sum(self.eta + self.S[1:], axis=1), np.arange(np.sum(Sd.values())))), axis=1)
                else:
                    for h in Sd:
                        probs += np.sum(np.log(np.add.outer(self.eta + self.S[:,h], np.arange(Sd[h]))), axis=1)
                    probs -= np.sum(np.log(np.add.outer(np.sum(self.eta + self.S, axis=1), np.arange(np.sum(Sd.values())))), axis=1)               
            else:
                if self.secondary_topic:
                    ## w | t,z components
                    for v in Wd:
                        probs += np.sum(np.log(np.add.outer(self.tau + self.W[1:,v], np.arange(Wd[v]))), axis=1)
                    probs -= np.sum(np.log(np.add.outer(np.sum(self.tau + self.W[1:], axis=1), np.arange(np.sum(Wd.values())))), axis=1)
                    ## z | t components
                    probs += np.sum(np.log(self.alpha + self.Z, np.arange(Zd)), axis=1)
                    probs += np.sum(np.log(self.alpha0 + self.M_star - self.Z, np.arange(np.sum(self.M[d]) - Zd)), axis=1)
                    probs -= np.sum(np.log(self.alpha0 + self.alpha + self.M_star, np.arange(np.sum(self.M[d]))), axis=1)
                else:
                    for v in Wd:
                        probs += np.sum(np.log(np.add.outer(self.tau + self.W[:,v], np.arange(Wd[v]))), axis=1)
                    probs -= np.sum(np.log(np.add.outer(np.sum(self.tau + self.W, axis=1), np.arange(np.sum(Wd.values())))), axis=1)
            # Transform the probabilities
            probs = np.exp(probs - logsumexp(probs))
            # Resample session-level topic
            td_new = np.random.choice(self.K, p=probs)
            # Update counts
            self.T[td_new] += 1
            if self.command_level_topics:
                for h in Sd:
                    self.S[td_new,h] += Sd[h]
            else:
                for v in Wd:
                    self.W[td_new + (1 if self.secondary_topic else 0),v] += Wd[v]
                if self.secondary_topic:
                    self.M_star[td_new] += np.sum(self.M[d])
                    self.Z[td_new] += Zd

   ## Resample session-level topics
    def resample_command_topics(self, size=1, indices=None):
        if not self.command_level_topics:
            raise TypeError('Command-level topics cannot be resampled if command_level_topics is not used.')
        if indices is None:
            indices_d = np.random.choice(self.D, size=size)
            indices_j = []
            for d in indices_d:
                indices_j += [int(np.random.choice(self.N[d]))]
            indices = np.vstack((indices_j,indices_d)).T
        for j, d in indices:
            td = self.t[d]
            s_old = int(self.s[d][j])
            self.S[td,s_old] -= 1
            if self.secondary_topic:
                self.M_star[s_old] -= self.M[d][j]
                Zdj = self.z[d][j]
                Wd = Counter(self.w[d][j][Zdj == 1])
                Zdj = int(np.sum(Zdj))
                self.Z[s_old] -= Zdj
            else:
                Wd = Counter(self.w[d][j])
            for v in Wd:
                self.W[s_old + (1 if self.secondary_topic else 0),v] -= Wd[v]
            # Calculate allocation probabilities
            probs = np.log(self.eta + self.S[td])
            if self.secondary_topic:
                ## w | s,z components
                for v in Wd:
                    probs += np.sum(np.log(np.add.outer(self.tau + self.W[1:,v], np.arange(Wd[v]))), axis=1)
                probs -= np.sum(np.log(np.add.outer(np.sum(self.tau + self.W[1:], axis=1), np.arange(np.sum(Wd.values())))), axis=1)
                ## z | s components
                probs += np.sum(np.log(self.alpha + self.Z, np.arange(Zdj)), axis=1)
                probs += np.sum(np.log(self.alpha0 + self.M_star - self.Z, np.arange(self.M[d][j] - Zdj)), axis=1)
                probs -= np.sum(np.log(self.alpha0 + self.alpha + self.M_star, np.arange(self.M[d][j])), axis=1)
            else:
                for v in Wd:
                    probs += np.sum(np.log(np.add.outer(self.tau + self.W[:,v], np.arange(Wd[v]))), axis=1)
                probs -= np.sum(np.log(np.add.outer(np.sum(self.tau + self.W, axis=1), np.arange(np.sum(Wd.values())))), axis=1)
        # Transform the probabilities
        probs = np.exp(probs - logsumexp(probs))
        # Resample command-level topic
        s_new = np.random.choice(self.H, p=probs)
        self.S[td,s_new] += 1
        for v in Wd:
            self.W[s_new + (1 if self.secondary_topic else 0),v] -= Wd[v]
        if self.secondary_topic:
            self.M_star[s_new] += self.M[d][j]
            self.Z[s_new] += Zdj
                        
    ## Resample indicator for primary-secondary topic
    def resample_indicators(self, size=1, indices=None):
        if not self.secondary_topic:
            raise TypeError('Indicators cannot be resampled if secondary topic are not used.')
        if indices is None:
            indices_d = np.random.choice(self.D, size=size)
            indices_j = []; indices_i = []
            for d in indices_d:
                indj = int(np.random.choice(self.N[d]))
                indices_j += [indj]
                indices_i += [int(np.random.choice(self.M[d][indj]))]
            indices = np.vstack((indices_i,indices_j,indices_d)).T
        # Resample the primary-secondary topic indicators
        for i, j, d in indices:
            v = int(self.w[d][j][i])
            z_old = int(self.z[d][j][i])
            if self.command_level_topics:
                topic = self.s[d,j]
            else:
                topic = self.t[d]
            self.Z[topic] -= z_old
            self.W[(topic+1)*z_old,v] -= 1
            # Calculate allocation probabilities
            probs = np.zeros(2)
            probs[0] = np.log(self.alpha + self.Z[topic]) + np.log(self.tau + self.W[topic+1,v]) - np.log(np.sum(self.tau + self.W[topic+1]))
            probs[1] = np.log(self.alpha0 + self.M_star[topic] - 1 - self.Z[topic]) + np.log(self.tau + self.W[0,v]) - np.log(np.sum(self.tau + self.W[0]))
            probs = np.exp(probs - logsumexp(probs))
            # Resample z
            z_new = np.random.choice(range(2), p=probs)
            # Update counts
            self.Z[topic] += z_new
            self.W[(topic+1)*z_new,v] += 1