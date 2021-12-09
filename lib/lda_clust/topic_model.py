#! /usr/bin/env python3
import numpy as np
from collections import Counter
from scipy.special import logsumexp, loggamma
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans

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
        self.N_cumsum = np.cumsum(self.N)
        self.M = {}
        for d in self.w:
            self.M[d] = [len(self.w[d][command]) for command in self.w[d]]
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
            self.S = np.zeros((self.K, self.H), dtype=int)
            self.W = np.zeros(shape=(self.H + (1 if self.secondary_topic else 0), self.V), dtype=int)
        else:
            self.W = np.zeros(shape=(self.K + (1 if self.secondary_topic else 0), self.V), dtype=int)             
        # Primary-secondary topic indicators
        if self.secondary_topic:
            self.z = {}
            for d in self.w:
                self.z[d] = {} 
                for j in self.w[d]:
                    self.z[d][j] = np.zeros(len(self.w[d][j]), dtype=int)
            if self.command_level_topics:
                self.M_star = np.zeros(shape=self.H, dtype=int)
                self.Z = np.zeros(shape=self.H, dtype=int)
            else:
                self.M_star = np.zeros(shape=self.K, dtype=int)
                self.Z = np.zeros(shape=self.K, dtype=int)

    ## Initialise counts given initial values of t, s and z
    def init_counts(self):
        # Initialise quantities 
        Q = Counter(self.t)
        for topic in Q:
            self.T[topic] += Q[topic]
        for doc in self.w:
            td = self.t[doc]
            if self.command_level_topics:
                for j in self.w[doc]:
                    sjd = self.s[doc][j]
                    self.S[td,sjd] += 1
                    if self.secondary_topic:
                        self.M_star[sjd] += self.M[doc][j] 
                        self.Z[sjd] += np.sum(self.z[doc][j])
                        # Primary topics
                        Wjd = Counter(self.w[doc][j][self.z[doc][j] == 1])
                        for v in Wjd:
                            self.W[sjd + 1, v] += Wjd[v]
                        # Secondary topics
                        Wjd = Counter(self.w[doc][j][self.z[doc][j] == 0])
                        for v in Wjd:
                            self.W[0, v] += Wjd[v]
                    else:
                        Wjd = Counter(self.w[doc][j])
                        for v in Wjd:
                            self.W[sjd, v] += Wjd[v]
            else:
                for j in self.w[doc]:
                    if self.secondary_topic:
                        self.M_star[td] += self.M[doc][j] 
                        self.Z[td] += np.sum(self.z[doc][j])
                        # Primary topics
                        Wjd = Counter(self.w[doc][j][self.z[doc][j] == 1])
                        for v in Wjd:
                            self.W[td + 1, v] += Wjd[v]
                        # Secondary topics
                        Wjd = Counter(self.w[doc][j][self.z[doc][j] == 0])
                        for v in Wjd:
                            self.W[0, v] += Wjd[v]
                    else:
                        Wjd = Counter(self.w[doc][j])
                        for v in Wjd:
                            self.W[td, v] += Wjd[v]
    
    ## Initializes chain at given values of t, s and z
    def custom_init(self, t, s=None, z=None):
        if isinstance(t, list) or isinstance(t, np.ndarray):
            if len(t) != self.K:
                raise TypeError('The initial value for t should be a K-dimensional list or np.ndarray.')
            if isinstance(t, list):
                self.t = np.array(t)
            else:
                self.t = t
        else:
            raise TypeError('The initial value for t should be a K-dimensional list or np.ndarray.')
        if s is not None:
            if not self.command_level_topics:
                raise TypeError('Command-level topics cannot be initialised if command_level_topics is not used.')
            elif not isinstance(s, dict):
                raise TypeError('The initial value for s should be a dictionary.')
            else:
                self.s = s
        if z is not None: 
            if not self.secondary_topic:
                raise TypeError('Secondary topics cannot be initialised if secondary_topic is not used.')
            elif not isinstance(s, dict):
                raise TypeError('The initial value for z should be a dictionary.')
            else:
                self.z = z
        ## Initialise counts
        self.init_counts()

    ## Initializes uniformly at random
    def random_init(self):
        ## Random initialisation
        for d in range(self.D):
            self.t[d] = np.random.choice(self.K)
            if self.command_level_topics:
                for j in self.w[d]:
                    self.s[d][j] = np.random.choice(self.H)
            if self.secondary_topic:
                for j in self.w[d]:
                    for i in range(self.M[d][j]):
                        self.z[d][j][i] = np.random.choice(2)
        ## Initialise counts
        self.init_counts()

    ## Initializes chain using gensim   
    def gensim_init(self, chunksize = 2000, passes = 100, iterations = 1000, eval_every= None):
        # Convert words into strings (gensim requirement)
        docs = []
        for d in self.w:
            if not self.command_level_topics:
                docs.append([])
            for j in self.w[d]:
                if self.command_level_topics:
                    docs.append([])
                for v in self.w[d][j]:
                    docs[-1].append(str(v))
        # Create dictionary
        dictionary = Dictionary(docs); temp = dictionary[0]
        # Create corpus
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        # Set number of topics
        num_topics = (self.K if not self.command_level_topics else self.H) + (1 if self.secondary_topic else 0)
        # Model setup
        id2word = dictionary.id2token
        model = LdaModel(corpus = corpus, id2word=id2word, chunksize=chunksize, alpha='auto', eta='auto',
                    iterations=iterations, num_topics=num_topics, passes=passes, eval_every=eval_every,)
        # Obtain topics from LDA
        topic_allocation = {}
        self.t = np.zeros(self.D, dtype=int)
        if self.command_level_topics:
            self.s = {}
        if self.secondary_topic:
            all_counter = Counter()
        topic_term = model.get_topics()
        for d in self.w:
            if not self.command_level_topics:
                topic_allocation[d] = []
            else:
                self.s[d] = np.zeros(self.N[d], dtype=int)
            for j in self.w[d]:
                if self.command_level_topics:
                    topic_allocation[d,j] = []
                for v in self.w[d][j]:
                    if self.command_level_topics:
                        topic_allocation[d,j] = np.append(topic_allocation[d,j],np.argmax(topic_term[:,v]))
                    else:
                        topic_allocation[d] = np.append(topic_allocation[d],np.argmax(topic_term[:,v]))
                if self.command_level_topics:
                    if self.secondary_topic:
                        all_counter += Counter(topic_allocation[d,j])
                    else:
                        self.s[d][j] = int(Counter(topic_allocation[d,j]).most_common(1)[0][0])
            if not self.command_level_topics:
                if self.secondary_topic:
                    all_counter += Counter(topic_allocation[d])
                else:
                    self.t[d] = int(Counter(topic_allocation[d]).most_common(1)[0][0])
        # If secondary topics are used, find the most common topic
        if self.secondary_topic:
            self.z = {}
            secondary_t = int(all_counter.most_common(1)[0][0])
            for d in self.w:
                self.z[d] = {}
                if not self.command_level_topics:
                    if np.sum(topic_allocation[d] != secondary_t) > 0:
                        primary_t = int(Counter(topic_allocation[d][topic_allocation[d] != secondary_t]).most_common(1)[0][0])
                        self.t[d] = primary_t - (1 if primary_t > secondary_t else 0)
                    else:
                        self.t[d] = np.random.choice(self.K)
                for j in self.w[d]:
                    if self.command_level_topics:
                        if np.sum(topic_allocation[d,j] != secondary_t) > 0:
                            primary_t = int(Counter(topic_allocation[d,j][topic_allocation[d,j] != secondary_t]).most_common(1)[0][0])
                            self.s[d][j] = primary_t - (1 if primary_t > secondary_t else 0)
                        else:
                            self.s[d][j] = np.random.choice(self.H)     
                    self.z[d][j] = np.array([int(np.argmax([topic_term[secondary_t,v], topic_term[primary_t,v]])) for v in self.w[d][j]])
        # If command-level topics are used, repeat gensim
        if self.command_level_topics:
            # Convert words (command-level topics) into strings (gensim requirement)
            docs = []
            for d in self.w:
                docs.append([])
                for s in self.s[d]:
                    docs[-1].append(str(s))
            # Dictionary and corpus
            dictionary = Dictionary(docs); temp = dictionary[0]
            corpus = [dictionary.doc2bow(doc) for doc in docs]
            # Model setup
            id2word = dictionary.id2token
            model = LdaModel(corpus = corpus, id2word=id2word, chunksize=chunksize, alpha='auto', eta='auto',
                    iterations=iterations, num_topics=self.K, passes=passes, eval_every=eval_every,)
            topic_term = model.get_topics()
            # Estimate topics
            for d in self.w:
                topic_allocation = []
                for s in self.s[d]:
                    topic_allocation = np.append(topic_allocation,np.argmax(topic_term[:,s]))
                self.t[d] = int(Counter(topic_allocation).most_common(1)[0][0])
        # Initialise counts
        self.init_counts()

    ## Initializes chain using spectral clustering  
    def spectral_init(self):
        if self.secondary_topic:
            raise TypeError('Spectral clustering is only implemented for initialisation when secondary topics are not used.')
        # Build co-occurrence matrix
        cooccurrence_matrix = {}
        for d in self.w:
            if self.command_level_topics:
                index = (self.N_cumsum[d-1] if self. d > 0 else 0)
                for j in self.w[d]:
                    cooccurrence_matrix[index+j] = Counter(self.w[d][j])
            else:
                cooccurrence_matrix[d] = Counter()
                for j in self.w[d]:
                    cooccurrence_matrix[d] += Counter(self.w[d][j])
        # Obtain matrix
        vals = []; rows = []; cols = []
        for key in cooccurrence_matrix:
            vals += [list(cooccurrence_matrix[key].values())]
            rows += [key] * len(cooccurrence_matrix[key])
            cols += [list(cooccurrence_matrix[key].keys())]
        # Co-occurrence matrix
        cooccurrence_matrix = coo_matrix((vals, (rows, cols)), shape=(self.N_cumsum[-1] if self.command_level_topics else self.D, self.V))
		## Spectral decomposition of A
        U, S, _ = svds(cooccurrence_matrix, k=self.H if self.command_level_topics else self.K)
        kmod = KMeans(n_clusters=self.H if self.command_level_topics else self.K, random_state=0).fit(U[:,::-1] * (S[::-1] ** .5))
        if not self.command_level_topics:
            self.t = kmod.labels_
        else:
            cooccurrence_matrix = {}
            self.s[0] = kmod.labels_[:self.N_cumsum[0]]
            cooccurrence_matrix[0] = Counter(self.s[0])
            for d in range(1,self.D):
                self.s[d] = kmod.labels_[self.N_cumsum[d-1]:self.N_cumsum[d]]
                cooccurrence_matrix[d] = Counter(self.s[d])
            # Obtain matrix
            vals = []; rows = []; cols = []
            for key in cooccurrence_matrix:
                vals += [list(cooccurrence_matrix[key].values())]
                rows += [key] * len(cooccurrence_matrix[key])
                cols += [list(cooccurrence_matrix[key].keys())]
            # Co-occurrence matrix
            cooccurrence_matrix = coo_matrix((vals, (rows, cols)), shape=(self.D, self.H))
            ## Spectral decomposition
            U, S, _ = svds(cooccurrence_matrix, k=self.K)
            kmod = KMeans(n_clusters=self.K, random_state=0).fit(U[:,::-1] * (S[::-1] ** .5))
            self.t = kmod.labels_
        # Initialise counts
        self.init_counts()     

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
        # Update counts
        self.S[td,s_new] += 1
        for v in Wd:
            self.W[s_new + (1 if self.secondary_topic else 0),v] += Wd[v]
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

    ## Split-merge move for session-level topics
    def split_merge_session(self, random_allocation=False):
        # Randomly choose two documents
        d, d_prime = np.random.choice(self.D, size=2, replace=False)
        # Propose a split or merge move according to the sampled values
        boundary = False
        if self.t[d] == self.t[d_prime]:
            if np.sum(self.T == 0) == 0:
                boundary = True
            else:
                split = True
                t = self.t[d]
                t_ast = np.min(np.where(self.T == 0)[0])
        else:
            if np.sum(self.T == 0) < self.K:
                split = False
                t = np.min([self.t[d],self.t[d_prime]])
                t_ast = np.max([self.t[d],self.t[d_prime]])
            else:
                boundary = True
        # Check if the proposed move is not at the boundary
        if not boundary:
            # Preprocessing for split / merge move
            if split:
                # Split move
                indices = np.where(self.t == t)[0]
                indices = indices[indices != d and indices != d_prime]
                if random_allocation:
                    allocation = np.random.choice(2,size=len(indices))
                    T_prop = np.zeros(2); T_prop[0] += np.sum(allocation); T_prop[1] = np.sum(1-allocation)
                    if self.command_level_topics:
                        S_prop = np.zeros((2,self.H))
                        S_prop[0] = Counter()
                        for doc in indices[allocation]:
                            Q = Counter(self.s[doc])
                            for h in Q:
                                S_prop[0] += Q[h]
                        S_prop[t_ast] = Counter()
                        for doc in indices[1-allocation]:
                            Q = Counter(self.s[doc])
                            for h in Q:
                                S_prop[t_ast] += Q[h]
                    else:
                        W_prop = np.zeros((2,self.V))
                        if self.secondary_topic:
                            M_ast_prop = np.zeros(2)
                            Z_prop = np.zeros(2)
                        for doc in indices[allocation]:
                            for j in self.w[d]:
                                Q = Counter(self.w[doc][j] if not self.secondary_topic else self.w[doc][j][self.z[doc][j] == 1])
                                for v in Q:
                                    W_prop[0,v] += Q[v]
                                if self.secondary_topic:
                                    M_ast_prop[0] += self.M[doc][j]
                                    Z_prop[0] += np.sum(self.z[doc][j])
                        for doc in indices[1-allocation]:
                            for j in self.w[doc]:
                                Q = Counter(self.w[doc][j] if not self.secondary_topic else self.w[doc][j][self.z[doc][j] == 1])
                                for v in Q:
                                    W_prop[1,v] += Q[v]
                                if self.secondary_topic:
                                    M_ast_prop[1] += self.M[doc][j]
                                    Z_prop[1] += np.sum(self.z[doc][j])                    
                else:
                    T_prop = np.ones(2)
                    if self.command_level_topics:
                        S_prop = np.zeros((2,self.H))
                        Q = Counter(self.s[d])
                        for h in Q:
                            S_prop[0,h] = Q[h]
                        Q = Counter(self.s[d_prime]) 
                        for h in Q:
                            S_prop[1,h] = Q[h]
                    else:
                        W_prop = np.zeros((2,self.V))
                        if self.secondary_topic:
                            M_ast_prop = np.zeros(2)
                            Z_prop = np.zeros(2)
                        for j in self.w[d]:
                            Q = Counter(self.w[d][j] if not self.secondary_topic else self.w[d][j][self.z[d][j] == 1])
                            for v in Q:
                                W_prop[0,v] += Q[v]
                            if self.secondary_topic:
                                M_ast_prop[0] += self.M[d][j]
                                Z_prop[0] += np.sum(self.z[d][j])
                        for j in self.w[d_prime]:
                            Q = Counter(self.w[d_prime][j] if not self.secondary_topic else self.w[d_prime][j][self.z[d_prime][j] == 1])
                            for v in Q:
                                W_prop[1,v] += Q[v]
                            if self.secondary_topic:
                                M_ast_prop[1] += self.M[d_prime][j]
                                Z_prop[1] += np.sum(self.z[d_prime][j])
            else:
                # Merge move
                indices = np.where(self.t == t or self.t == t_ast)
                indices = indices[indices != d and indices != d_prime]
                T_prop = np.array([self.T[t] + self.T[t_ast],0])
                T_temp = np.ones(2)
                if self.command_level_topics:
                    S_prop = np.zeros((2,self.H)); S_prop[0] = self.S[t] + self.S[t_ast]
                    S_temp = np.zeros((2,self.H))
                    Q = Counter(self.s[d])
                    for h in Q:
                        S_temp[0,h] = Q[h]
                    Q = Counter(self.s[d_prime])
                    for h in Q:
                        S_temp[1,h] = Q[h]
                else:
                    W_prop = np.zeros((2,self.V)); W_prop[0] = self.W[t + (1 if self.secondary_topic else 0)] + self.W[t_ast + (1 if self.secondary_topic else 0)]
                    W_temp = np.zeros((2,self.V))
                    for j in self.w[d]:
                        Q = Counter(self.w[d][j] if not self.secondary_topic else self.w[d][j][self.z[d][j] == 1])
                        for v in Q:
                            W_temp[0,v] += Q[v] 
                    for j in self.w[d_prime]:    
                        Q = Counter(self.w[d_prime][j] if not self.secondary_topic else self.w[d_prime][j][self.z[d_prime][j] == 1])
                        for v in Q:
                            W_temp[1,v] += Q[v] 
                    if self.secondary_topic:
                        M_ast_prop = np.zeros(2); M_ast_prop[0] = self.M_star[t] + self.M_star[t_ast]
                        Z_prop = np.zeros(2); Z_prop[0] = self.Z[t] + self.Z[t_ast]
                        M_ast_temp = np.zeros(2); Z_temp = np.zeros(2)
                        for j in self.w[d]:
                            M_ast_temp[0] += self.M[d][j]
                            Z_temp[0] += np.sum(self.w[d][j])
                        for j in self.w[d_prime]:
                            M_ast_temp[1] += self.M[d_prime][j]
                            Z_temp[1] += np.sum(self.w[d_prime][j])
            # Caclulate proposal probability
            if not random_allocation:
                probs_proposal = 0
                for doc in np.random.choice(indices,size=len(indices),replace=False):
                    if self.command_level_topics:
                        Sd = Counter(self.s[doc])
                    else:
                        Wd = Counter()
                        if self.secondary_topic:
                            Zd = 0
                        for j in self.w[doc]:
                            if self.secondary_topic:
                                Zdj = self.z[doc][j]
                                Wd += Counter(self.w[doc][j][Zdj == 1])
                                Z_partial = np.sum(Zdj)
                                Zd += Z_partial
                            else:
                                Wd += Counter(self.w[doc][j])
                    # Calculate allocation probabilities
                    if split:
                        probs = np.log(self.gamma + T_prop)
                        if self.command_level_topics:
                            for h in Sd:
                                probs += np.sum(np.log(np.add.outer(self.eta + S_prop[:,h], np.arange(Sd[h]))), axis=1)
                            probs -= np.sum(np.log(np.add.outer(np.sum(self.eta + S_prop, axis=1), np.arange(np.sum(Sd.values())))), axis=1)               
                        else:
                            ## w | t,z components
                            for v in Wd:
                                probs += np.sum(np.log(np.add.outer(self.tau + W_prop[:,v], np.arange(Wd[v]))), axis=1)
                            probs -= np.sum(np.log(np.add.outer(np.sum(self.tau + W_prop, axis=1), np.arange(np.sum(Wd.values())))), axis=1)
                            if self.secondary_topic:
                                ## z | t components
                                probs += np.sum(np.log(self.alpha + Z_prop, np.arange(Zd)), axis=1)
                                probs += np.sum(np.log(self.alpha0 + M_ast_prop - Z_prop, np.arange(np.sum(self.M[doc]) - Zd)), axis=1)
                                probs -= np.sum(np.log(self.alpha0 + self.alpha + M_ast_prop, np.arange(np.sum(self.M[doc]))), axis=1)
                    else:
                        probs = np.log(self.gamma + T_temp)
                        if self.command_level_topics:
                            for h in Sd:
                                probs += np.sum(np.log(np.add.outer(self.eta + S_temp[:,h], np.arange(Sd[h]))), axis=1)
                            probs -= np.sum(np.log(np.add.outer(np.sum(self.eta + S_temp, axis=1), np.arange(np.sum(Sd.values())))), axis=1)               
                        else:
                            ## w | t,z components
                            for v in Wd:
                                probs += np.sum(np.log(np.add.outer(self.tau + W_temp[:,v], np.arange(Wd[v]))), axis=1)
                            probs -= np.sum(np.log(np.add.outer(np.sum(self.tau + W_temp, axis=1), np.arange(np.sum(Wd.values())))), axis=1)
                            if self.secondary_topic:
                                ## z | t components
                                probs += np.sum(np.log(self.alpha + Z_temp, np.arange(Zd)), axis=1)
                                probs += np.sum(np.log(self.alpha0 + M_ast_temp - Z_temp, np.arange(np.sum(self.M[doc]) - Zd)), axis=1)
                                probs -= np.sum(np.log(self.alpha0 + self.alpha + M_ast_temp, np.arange(np.sum(self.M[doc]))), axis=1)
                    # Transform the probabilities
                    probs = np.exp(probs - logsumexp(probs))
                    # Resample
                    td_new = np.random.choice(2, p=probs)
                    # Calculate Q's for the MH ratio
                    probs_proposal += np.log(probs[td_new])
                    if split:
                        # Update counts
                        T_prop[td_new] += 1
                        if self.command_level_topics:
                            for h in Sd:
                                self.S[td_new,h] += Sd[h]
                        else:
                            for v in Wd:
                                W_prop[td_new,v] += Wd[v]
                            if self.secondary_topic:
                                M_ast_prop[td_new] += np.sum(self.M[doc])
                                Z_prop[td_new] += Zd
                    else:
                        T_temp[td_new] += 1
                        if self.command_level_topics:
                            for h in Sd:
                                self.S[td_new,h] += Sd[h]
                        else:
                            for v in Wd:
                                W_temp[td_new,v] += Wd[v]
                            if self.secondary_topic:
                                M_ast_temp[td_new] += np.sum(self.M[doc])
                                Z_temp[td_new] += Zd
            else:
                probs_proposal = len(indices) * np.log(2)
            # Calculate the Metropolis-Hastings acceptance ratio
            t_indices = np.array([t,t_ast])
            acceptance_ratio = np.sum(loggamma(self.gamma + T_prop)) - np.sum(loggamma(self.gamma + self.T[t_indices]))
            if self.command_level_topics:
                acceptance_ratio += np.sum(loggamma(self.eta + S_prop)) - np.sum(loggamma(self.eta + self.S[t_indices,:]))
                acceptance_ratio -= np.sum(loggamma(np.sum(self.eta + S_prop, axis=1))) - np.sum(loggamma(np.sum(self.eta + self.S[t_indices], axis=1)))
                if self.secondary_topic:
                    acceptance_ratio += np.sum(loggamma(self.alpha + Z_prop)) + np.sum(loggamma(self.alpha0 + M_ast_prop - Z_prop))
                    acceptance_ratio -= np.sum(loggamma(self.alpha + self.alpha0 + M_ast_prop))
                    acceptance_ratio += np.sum(loggamma(self.alpha + self.Z[t_indices])) + np.sum(loggamma(self.alpha0 + self.M_star[t_indices] - self.Z[t_indices]))
                    acceptance_ratio -= np.sum(loggamma(self.alpha + self.alpha0 + self.M_star[t_indices]))
            else:
                acceptance_ratio += np.sum(loggamma(self.tau + W_prop)) - np.sum(loggamma(self.tau + self.W[t_indices + (1 if self.secondary_topic else 0)]))
                acceptance_ratio -= np.sum(loggamma(np.sum(self.tau + W_prop, axis=1))) - np.sum(loggamma(np.sum(self.tau + self.W[t_indices + (1 if self.secondary_topic else 0)], axis=1)))
            if split: 
                acceptance_ratio -= probs_proposal
            else:
                acceptance_ratio += probs_proposal
            # Accept / reject using Metropolis-Hastings
            accept = (-np.random.exponential(1) < acceptance_ratio)
            # Update if move is accepted
            if accept:
                self.T[t] = T_prop[0]; self.T[t_ast] = T_prop[1]
                if self.command_level_topics:
                    self.S[t] = S_prop[0]; self.S[t_ast] = S_prop[1]
                else:
                    self.W[t + (1 if self.secondary_topic else 0)] = W_prop[0]; self.W[t_ast + (1 if self.secondary_topic else 0)] = W_prop[1] 
                    if self.secondary_topic:
                        self.M_star[t] = M_ast_prop[0]; self.M_star[t_ast] = M_ast_prop[1]
                        self.Z[t] = Z_prop[0]; self.Z[t_ast] = Z_prop[1]

    ## Split-merge move for session-level topics
    def split_merge_command(self, random_allocation=False):
        if not self.command_level_topics:
            raise TypeError('Command-level topics cannot be resampled if command_level_topics is not used.')
        indices_int = np.random.choice(self.N_cumsum[-1], size=2, replace=False)
        ## First index
        d = np.argmax(self.N_cumsum[indices_int[0] < self.N_cumsum])
        j = indices_int[0] - self.N_cumsum[d]
        ## Second index
        d_prime = np.argmax(self.N_cumsum[indices_int[1] < self.N_cumsum])
        j_prime = indices_int[1] - self.N_cumsum[d]
        # Propose a split or merge move according to the sampled values
        boundary = False
        if self.s[d][j] == self.s[d_prime][j_prime]:
            if np.sum(self.S,axis=0) == 0:
                boundary = True
            else:
                split = True
                s = self.s[d][j]
                s_ast = np.min(np.where(np.sum(self.S,axis=0) == 0)[0])
        else:
            if np.sum(self.S,axis=0) < self.H:
                split = False
                s = np.min([self.s[d][j],self.s[d_prime][j_prime]])
                s_ast = np.max([self.s[d][j],self.s[d_prime][j_prime]])
            else:
                boundary = True
        # Check if the proposed move is not at the boundary
        if not boundary:
            # Preprocessing for split / merge move
            if split:
                # Split move
                indices = {} 
                for doc in self.s:
                    indices[doc] = np.where(self.s[doc] == s)[0]
                W_prop = ((2,self.V))
                if random_allocation:
                    S_prop = np.zeros((2,self.K))
                    if self.secondary_topic:
                        M_ast_prop = np.zeros(2); Z_prop = np.zeros(2)
                    for doc in self.s:
                        allocation = np.random.choice(2,size=len(indices[doc]))
                        S_prop[0,self.t[doc]] += np.sum(allocation)
                        S_prop[1,self.t[doc]] += np.sum(1-allocation)
                        for command in self.w[d]:
                            if allocation[command] == 1:
                                if self.secondary_topic:    
                                    M_ast_prop[0] += self.M[doc][command] 
                                    Z_prop[0] += np.sum(self.z[doc][command])
                                Q = Counter(self.w[doc][command][self.z[doc][command] == 1] if self.secondary_topic else self.w[doc][command])
                                for v in Q:
                                    W_prop[0,v] += Q[v]
                            else:
                                if self.secondary_topic: 
                                    M_ast_prop[1] += self.M[doc][command] 
                                    Z_prop[1] += np.sum(self.z[doc][command])
                                Q = Counter(self.w[doc][command][self.z[doc][command] == 1] if self.secondary_topic else self.w[doc][command])
                                for v in Q:
                                    W_prop[1,v] += Q[v]
            else:
                # Merge move
                indices = {} 
                for doc in self.s:
                    indices[doc] = np.where(self.s[doc] == s or self.s[doc] == s_ast)[0]
                S_prop = np.zeros((2,self.K)); S_prop[0] = self.S[:,s] + self.S[:,s_ast]
                S_temp = np.zeros((2,self.K)); S_temp[0,self.t[d]] = 1; S_temp[1,self.t[d_prime]] = 1
                W_prop = np.zeros((2,self.V)); W_prop[0] = self.W[s + (1 if self.secondary_topic else 0)] + self.W[s_ast + (1 if self.secondary_topic else 0)]
                W_temp = np.zeros((2,self.V))
                Q = Counter(self.w[d][j] if not self.secondary_topic else self.w[d][j][self.z[d][j] == 1])
                for v in Q:
                    W_temp[0,v] = Q[v]                    
                Q = Counter(self.w[d_prime][j_prime] if not self.secondary_topic else self.w[d_prime][j_prime][self.w[d_prime][j_prime] == 1])
                for v in Q:
                    W_temp[1,v] = Q[v]
                if self.secondary_topic:
                    M_ast_prop = np.zeros(2); M_ast_prop[0] = self.M_star[s] + self.M_star[s_ast]
                    Z_prop = np.zeros(2); Z_prop[0] = self.Z[s] + self.Z[s_ast]
                    M_ast_temp = np.zeros(2); Z_temp = np.zeros(2)
                    M_ast_temp[0] += self.M[d][j]
                    Z_temp[0] += np.sum(self.w[d][j])
                    M_ast_temp[1] += self.M[d_prime][j_prime]
                    Z_temp[1] += np.sum(self.w[d_prime][j_prime])
            # Caclulate proposal probability
            if not random_allocation:
                probs_proposal = 0
                for doc in np.random.choice(list(indices.keys()), size=len(indices), replace=False):
                    td = self.t[doc]
                    for command in np.random.choice(list(indices[doc].keys()), size=len(indices[doc]), replace=False):
                        if self.secondary_topic:
                            Zjd = np.sum(self.z[doc][command])
                            Wjd = Counter(self.w[doc][command][self.z[doc][command] == 1])
                        else:
                            Wjd = Counter(self.w[doc][command])
                        # Calculate allocation probabilities
                        if split:
                            probs = np.log(self.eta + S_prop[:,td])
                            for v in Wjd:
                                probs += np.sum(np.log(np.add.outer(self.tau + W_prop[:,v], np.arange(Wjd[v]))), axis=1)
                            probs -= np.sum(np.log(np.add.outer(np.sum(self.tau + W_prop, axis=1), np.arange(np.sum(Wjd.values())))), axis=1)
                            if self.secondary_topic:
                                probs += np.sum(np.log(self.alpha + Z_prop, np.arange(Zjd)), axis=1)
                                probs += np.sum(np.log(self.alpha0 + M_ast_prop - Z_prop, np.arange(self.M[doc][command] - Zjd)), axis=1)
                                probs -= np.sum(np.log(self.alpha0 + self.alpha + M_ast_prop, np.arange(self.M[doc][command])), axis=1)              
                        else:
                            probs = np.log(self.eta + S_temp)
                            for v in Wjd:
                                probs += np.sum(np.log(np.add.outer(self.tau + W_temp[:,v], np.arange(Wjd[v]))), axis=1)
                            probs -= np.sum(np.log(np.add.outer(np.sum(self.tau + W_temp, axis=1), np.arange(np.sum(Wjd.values())))), axis=1)
                            if self.secondary_topic:
                                probs += np.sum(np.log(self.alpha + Z_temp, np.arange(Zjd)), axis=1)
                                probs += np.sum(np.log(self.alpha0 + M_ast_temp - Z_temp, np.arange(self.M[doc][command] - Zjd)), axis=1)
                                probs -= np.sum(np.log(self.alpha0 + self.alpha + M_ast_temp, np.arange(self.M[doc][command])), axis=1)                                         
                    # Transform the probabilities
                    probs = np.exp(probs - logsumexp(probs))
                    # Resample
                    sjd_new = np.random.choice(2, p=probs)
                    # Calculate Q's for the MH ratio
                    probs_proposal += np.log(probs[sjd_new])
                    if split:
                        # Update counts
                        S_prop[sjd_new,td] += 1
                        for v in Wjd:
                            W_prop[sjd_new,v] += Wjd[v]
                        if self.secondary_topic:
                            M_ast_prop[sjd_new] += self.M[doc][command]
                            Z_prop[sjd_new] += Zjd
                    else:
                        S_temp[sjd_new,td] += 1
                        for v in Wjd:
                            W_temp[sjd_new,v] += Wjd[v]
                            if self.secondary_topic:
                                M_ast_temp[sjd_new] += self.M[doc][command]
                                Z_temp[sjd_new] += Zjd
            else:
                probs_proposal = len(indices) * np.log(2)
            # Calculate the Metropolis-Hastings acceptance ratio
            s_indices = np.array([s,s_ast])
            acceptance_ratio = np.sum(loggamma(self.gamma + S_prop)) - np.sum(loggamma(self.gamma + self.S[:,s_indices]))
            acceptance_ratio += np.sum(loggamma(self.tau + W_prop)) - np.sum(loggamma(self.eta + self.W[s_indices + (1 if self.secondary_topic else 0)]))
            acceptance_ratio -= np.sum(loggamma(np.sum(self.eta + W_prop, axis=1))) - np.sum(loggamma(np.sum(self.eta + self.W[s_indices + (1 if self.secondary_topic else 0)], axis=1)))
            if self.secondary_topic:
                acceptance_ratio += np.sum(loggamma(self.alpha + Z_prop)) + np.sum(loggamma(self.alpha0 + M_ast_prop - Z_prop))
                acceptance_ratio -= np.sum(loggamma(self.alpha + self.alpha0 + M_ast_prop))
                acceptance_ratio += np.sum(loggamma(self.alpha + self.Z[s_indices])) + np.sum(loggamma(self.alpha0 + self.M_star[s_indices] - self.Z[s_indices]))
                acceptance_ratio -= np.sum(loggamma(self.alpha + self.alpha0 + self.M_star[s_indices]))
            if split:
                acceptance_ratio -= probs_proposal
            else:
                acceptance_ratio += probs_proposal
            # Accept / reject using Metropolis-Hastings
            accept = (-np.random.exponential(1) < acceptance_ratio)
            # Update if move is accepted
            if accept:
                self.S[s] = S_prop[0]; self.S[s_ast] = S_prop[1]
                self.W[s + (1 if self.secondary_topic else 0)] = W_prop[0]; self.W[s_ast + (1 if self.secondary_topic else 0)] = W_prop[1] 
                if self.secondary_topic:
                    self.M_star[s] = M_ast_prop[0]; self.M_star[s_ast] = M_ast_prop[1]
                    self.Z[s] = Z_prop[0]; self.Z[s_ast] = Z_prop[1]