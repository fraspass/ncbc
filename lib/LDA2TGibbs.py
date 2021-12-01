#! /usr/bin/env python3
import numpy as np
from scipy.special import gammaln
from collections import Counter
import copy
from scipy.special import logsumexp

class LDA2TGibbs:
    ## Purpose: does gibbs sampling of z and t for LDA2
    ## Required input: W - list of lists containing the words (as consecutive integers starting at 0)
    ##                     for each document
    ##                 K - integer number of primary topics
    ##                 alpha,  eta, gamma - hyper-parameters as lists of floats
    ## Optional: V - size of vocabulary

    def __init__(self, W, K, alpha, eta, gamma, V=0):
        self.W = W
        self.K = K
        self.alpha = alpha
        self.eta = eta
        self.gamma = gamma
        self.gammaSum = np.sum(self.gamma)
        self.etaSum = np.sum(self.eta)
        # number of documents
        self.D = W.shape[0]
        # length of each document
        self.N_d = [int(len(W[d])) for d in range(self.D)]
        # determine V if not given
        if V > 1:
            self.V = V
        else:
            self.V = 0
            for d in range(self.D):
                for i in range(self.N_d[d]):
                    v = self.W[d][i]
                    if v > self.V:
                        self.V = v
            self.V += 1
        # Z = topic allocations (containing z^d_i)
        self.Z = [[] for d in range(self.D)]
        # T = primary topics for each document T (containing t^d),
        self.T = np.zeros(self.D, dtype=np.int64)
        # P_d = number of words in document d allocated the primary topic
        self.P_d = np.zeros(self.D, dtype=np.int64)
        # M_r = number of documents allocated with primary topic r
        # note here M_r = 0 for convenience
        self.M_r = np.zeros(self.K + 1, dtype=np.int64)
        # N_vt = number of words v assigned topic t
        self.N_vt = np.zeros((self.V, self.K + 1), dtype=np.int64)
        # N_t = number of words assigned topic t
        self.N_t = np.zeros(self.K + 1, dtype=np.int64)
        # For storing MCMC samples
        self.T_samples = []
        self.Z_samples = []
        # likelihood
        self.mll = []
    
    ## Computes logarithm of normalizing constant B of a Dirichlet distribution
    def logB(self, vec):
        out = 0
        for v in vec:
            out += gammaln(v)
        out -= gammaln(np.sum(vec))
        return out
    
    ## Computes marginal loglikelihood up to proportionality
    def marginal_loglike(self):
        ll = 0
        ll += self.logB(self.gamma + self.M_r[1:])
        for k in range(self.K+1):
            ll += self.logB(self.eta + self.N_vt[:, k])
        for d in range(self.D):
            ll += self.logB(np.array([self.P_d[d] + self.alpha[self.T[d]]], self.N_d[d] - self.P_d[d] + self.alpha[0]))
        return ll

    ## Initializes chain at given Z_init, T_init
    def custom_init(self, Z_init, T_init):        
        self.T = copy.deepcopy(T_init)
        self.Z = copy.deepcopy(Z_init)
        for d in range(self.D):
            t = self.T[d]
            self.M_r[t] += 1
            for i in range(self.N_d[d]):
                z = self.Z[d][i]
                v = self.W[d][i]
                t_di = z*t
                self.N_vt[v, t_di] += 1
                self.N_t[t_di] += 1
                self.P_d[d] += z
        self.T_samples.append(self.T)
        self.Z_samples += [copy.deepcopy(self.Z)]
        self.mll.append(np.array(self.marginal_loglike()))
    
    ## Initializes chain at uniform random T and Z
    def random_init(self):
        for d in range(self.D):
            self.T[d] = np.random.choice(range(1, self.K + 1))
            for i in range(self.N_d[d]):
                self.Z[d].append(np.random.choice(range(2)))
            # Count P_d
            self.P_d[d] = self.Z[d].count(1)
        # Do remainder of counting
        M_counts = Counter(self.T)
        for r in range(1, self.K + 1):
            self.M_r[r] = M_counts[r]
        for d in range(self.D):
            for i in range(self.N_d[d]):
                t_di = int(self.Z[d][i] * self.T[d])
                v = int(self.W[d][i])
                self.N_vt[v, t_di] += 1
                self.N_t[t_di] += 1
    
    ## Initializes chain utilizing gensim   
    def gensim_init(self, chunksize = 2000, passes = 100, iterations = 1000, eval_every= None):
        import gensim 
        from gensim.models import LdaModel
        from gensim.models.coherencemodel import CoherenceModel
        # this maps words to consecutive integers (whatever the words)
        unique_words = {}
        for d in range(self.D):
            for i in range(self.N_d[d]):
                v = self.W[d][i]
                if v not in unique_words:
                    unique_words[v] = len(unique_words)
                self.W[d][i] = unique_words[v]
        # turn numbers into strings because that's what gensim needs
        docs = [[] for d in range(len(self.W))]
        for d in range(len(self.W)):
            for v in self.W[d]:
                docs[d].append(str(v))
        # create dictionary
        from gensim.corpora import Dictionary
        dictionary = Dictionary(docs)
        # create corpus
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        # run LdaModel
        from gensim.models import LdaModel
        num_topics = self.K+1
        id2word = dictionary.id2token
        model = LdaModel(
                    corpus = corpus,
                    id2word=id2word,
                    chunksize=chunksize,
                    alpha='auto',
                    eta='auto',
                    iterations=iterations,
                    num_topics=num_topics,
                    passes=passes,
                    eval_every=eval_every
                    )
        # sample Z as in LDA
        temp_stoch_Z = [[] for i in range(len(self.W))] 
        topic_term = model.get_topics()
        for d in range(len(self.W)):
            for i in range(len(self.W[d])):
                v = self.W[d][i]
                probs = topic_term[:,v] / sum(topic_term[:, v])
                temp_stoch_Z[d].append(np.random.choice(range(0,num_topics), p=probs))
        # count occurences of all topics
        all_counter = Counter(temp_stoch_Z[0])
        for i in range(1, len(self.W)):
            all_counter = all_counter + Counter(temp_stoch_Z[i])
        # set secondary topic to be most frequently occuring topic
        secondary_t = max(all_counter, key=all_counter.get) 
        # sample T
        T_init = []
        for d in range(len(self.W)):
            doc_topics = model.get_document_topics(corpus[d],  minimum_probability=0)
            sorted_doc_topics = sorted(doc_topics, key=lambda c: c[1], reverse=True)
            top_topic = sorted_doc_topics[0][0]
            if top_topic == secondary_t:
                top_topic = sorted_doc_topics[1][0]
            T_init.append(top_topic)
        # sample Z
        Z_init =[[] for i in range(len(self.W))]
        for d in range(len(self.W)):
            for i in range(len(self.W[d])):
                v = self.W[d][i]
                primary_t = T_init[d]
                probs_secondary = topic_term[secondary_t, v]
                probs_primary = topic_term[primary_t, v]
                probs = np.array([probs_secondary, probs_primary])
                probs = probs / sum(probs)
                z = np.random.choice([0, 1], p=probs)
                Z_init[d].append(z)
        # relabel for T_init for the sampler
        unique_t = [i for i in set(T_init)]
        for i in range(len(T_init)):
            t = copy.deepcopy(T_init[i])
            T_init[i] = unique_t.index(t) + 1
        self.custom_init(Z_init, T_init)
        
   ## Resamples t^d for given subset of documents
    def resample_topics(self, subset=None):
        ## Optional input: subset - list of integers d
        ## Notes: - by default sample just one document
        if subset is None:
            subset = [np.random.choice(range(self.D))]
        # resample for each document in subset
        for d in subset:
            t = self.T[d]
            # remove relevant counts
            self.M_r[t] -= 1
            # calculate probabilities
            logprob = np.zeros(self.K)
            for r in range(1, self.K + 1):
                # update counts according to t = r
                self.M_r[r] += 1
                for i in range(self.N_d[d]):
                    z = self.Z[d][i]
                    if z == 1:
                        v = self.W[d][i]
                        self.N_vt[v, r] += 1
                # calculate prob
                logprob[r-1] += np.log(self.gamma[r-1] + self.M_r[r])
                for k in range(self.K + 1):
                    logprob[r-1] += self.logB(self.eta + self.N_vt[:, k])
                # remove counts for t = r
                self.M_r[r] -= 1
                for i in range(self.N_d[d]):
                    z = self.Z[d][i]
                    if z == 1:
                        v = self.W[d][i]
                        self.N_vt[v, r] -= 1
            logprob = logprob - max(logprob)
            prob = np.exp(logprob)
            prob = prob / np.sum(prob)
            # choose new topic
            t_new = np.random.choice(range(1, self.K + 1), p=prob)
            # update relevant counts
            self.M_r[t_new] += 1
            for i in range(self.N_d[d]):
                z = self.Z[d][i]
                if z == 1:
                    v = self.W[d][i]
                    self.N_vt[v, t] -= 1
                    self.N_t[t] -= 1
                    self.N_vt[v, t_new] += 1
                    self.N_t[t_new] += 1
            self.T[d] = t_new
            
    ## Resamples z^d_i for given subset of documents
    def resample_allocations(self, subset=None):
        ## Input: subset - list of tuples (i,d)
        ## Notes: - by default sample a single topic allocation
        if subset is None:
            subset = []
            d = np.random.choice(range(self.D))
            i = np.random.choice(range(self.N_d[d]))
            subset.append((i, d))
        # resample z^d_i for those in subset
        for i, d in subset:
            v = int(self.W[d][i])
            z = int(self.Z[d][i])
            t = self.T[d]
            t_di = z * t
            # remove current allocation from counts
            self.N_vt[v, t_di] -= 1
            self.N_t[t_di] -= 1
            self.P_d[d] -= z
            # calculate probabilities
            prob = np.zeros(2)
            prob[1] = np.log(self.eta[v] + self.N_vt[v,t]) - np.log(self.etaSum + self.N_t[t]) + np.log(self.P_d[d] + self.alpha[t])
            prob[0] = np.log(self.eta[v] + self.N_vt[v,0]) - np.log(self.etaSum + self.N_t[0]) + np.log(self.N_d[d] - 1 - self.P_d[d] + self.alpha[0])
            prob = np.exp(prob - logsumexp(prob))
            prob = prob / np.sum(prob)
            # sample z
            z_new = np.random.choice(range(2), p=prob)
            # update allocation
            t_di_new = z_new * t
            # update counts
            self.N_vt[v, t_di_new] += 1
            self.N_t[t_di_new] += 1
            self.P_d[d] += z_new
            self.Z[d][i] = z_new
    
    ## Computes marginal loglikelihood for merge/split move
    def merge_split_ll(self, W, Z, T, gamma, eta, alpha, D, K, V):
        # P_d = number of words in a document allocated the primary topic
        P_d = np.zeros(self.D, dtype=np.int64)
        # M_r = number of documents allocated with primary topic r
        # note here M_r = 0 for convenience
        M_r = np.zeros(self.K + 1, dtype=np.int64)
        # N_vt = number of words v assigned topic t
        N_vt = np.zeros((self.V, self.K + 1), dtype=np.int64)
        # N_t = number of words assigned topic t
        N_t = np.zeros(self.K + 1, dtype=np.int64)
        N_d = np.array([len(W[d]) for d in range(self.D)])
        for d in range(self.D):
            t = T[d]
            M_r[t] += 1
            for i in range(N_d[d]):
                z = Z[d][i]
                v = W[d][i]
                t_di = z*t
                N_vt[v, t_di] += 1
                N_t[t_di] += 1
                P_d[d] += z
        ll = 0
        ll += self.logB(gamma + M_r[1:])
        for k in range(K+1):
            ll += self.logB(eta + N_vt[:, k])
        for d in range(D):
            ll += self.logB(np.array([P_d[d] + alpha[T[d]]], N_d[d] - P_d[d] + alpha[0]))
        return ll
    
    ## Approximative merge/split move
    def merge_split(self):
        t = np.random.choice(range(1,self.K+1))
        # probably want to adjust this
        move = np.random.choice(['merge', 'split'], p=[0.5, 0.5])
        if move == 'merge':
            self.merge(t)
        else:
            self.split(t)
        return None
    
    ## Approximative split move
    def split(self, t):
        # cannot split empty topic
        if self.N_t[t] == 0:
            return None
        # choose arbitrary empty topic 
        split_t = -1
        for i in range(1, self.K + 1):
            if self.N_t[i] == 0:
                split_t = i
                break
        # cannot split if there are no empty topics
        if split_t == -1:
            return None
        marg_ll_M1 = self.mll[-1]
        T_alt = copy.deepcopy(self.T)
        for d in range(self.D):
            if self.T[d] == t:
                t_new = np.random.choice([t, split_t])
                T_alt[d] = t_new
        marg_ll_M2 = self.merge_split_ll(W=self.W, Z=self.Z, T=T_alt, gamma=self.gamma, 
                                         eta=self.eta, alpha=self.eta, D=self.D, K=self.K, V=self.V)
        alpha = np.exp(marg_ll_M2 - marg_ll_M1)
        u = np.random.uniform()
        if u < alpha:
            self.T = T_alt
            #update relevant counts
            for d in range(self.D):
                if T_alt[d] == split_t:
                    self.M_r[t] -= 1
                    self.M_r[split_t] += 1
                    for i in range(self.N_d[d]):
                        v = self.W[d][i]
                        if self.Z[d][i] == 1:
                            self.N_vt[v, t] -= 1
                            self.N_vt[v, split_t] += 1
                    self.N_t[t] -= self.P_d[d]
                    self.N_t[split_t] += self.P_d[d]
 
        return None
    
    ## Approximative merge move    
    def merge(self, t):
        try_merge = t
        T_current = copy.deepcopy(self.T)
        try_k = t
        for k in range(1, self.K+1):
            if k == try_k:
                continue
            if self.N_t[k] == 0:
                continue
            # M1 is not merge
            marg_ll_M1 = self.mll[-1]
            T_alt = copy.deepcopy(T_current)
            for i in range(len(T_alt)):
                if T_alt[i] == k:
                    T_alt[i] = try_k
            # M2 is merge try_k and k
            marg_ll_M2 = self.merge_split_ll(W=self.W, Z=self.Z, T=T_alt, gamma=self.gamma, 
                                         eta=self.eta, alpha=self.eta, D=self.D, K=self.K, V=self.V)
            alpha = np.exp(marg_ll_M2 - marg_ll_M1)
            u = np.random.uniform()
            if u < alpha:
                self.T = T_alt
                #update relevant counts
                self.M_r[try_merge] += copy.deepcopy(self.M_r[k])
                self.M_r[k] = 0
                for v in range(self.V):
                    self.N_vt[v,try_merge] += copy.deepcopy(self.N_vt[v, k])
                    self.N_vt[v, k] = 0
                self.N_t[try_merge] += copy.deepcopy(self.N_t[k])
                self.N_t[k] =0
                break
        return None

    ## Runs MCMC chain
    def MCMC(self, stored_T_file, stored_Z_file, stored_ll_file, last_T_file, last_Z_file,
             store_samples=True, iterations=100, print_progress=True, load_last = True, 
             move_probs=[0.45,0.45, 0.1]):
        ## Inputs: stored_X_file - stored self.T_samples or self.Z_samples or self.mll array as .npy
        ##         last_X_file - stored self.T or self.Z used to initialize from 
        ##         store_samples - logical detailing whether to store each sample or not
        ##         iterations - integer of times to run chain
        ##         print_progress - whether to print progress or not
        ##         load_last - whether to load from last_T_file and last_Z_file
        ##         move_probs - probabilities of the three moves
        if load_last == True:
            last_T = np.load(last_T_file)
            last_Z = np.load(last_Z_file, allow_pickle=True)
            self.custom_init(last_Z, last_T)
        for it in range(iterations):
            # sample move
            move = np.random.choice(['z', 't', 'm/s'], p=move_probs)
            # do move
            if move == 'z':
                self.resample_allocations()
            elif move == 't':
                self.resample_topics()
            else:
                self.merge_split()
            # store every 100 samples in memory
            if it % 100 == 0:
                if store_samples is True:
                    self.T_samples.append(copy.deepcopy(self.T))
                    self.Z_samples.append(copy.deepcopy(self.Z))
                    self.mll.append(np.array(self.marginal_loglike()))
            if print_progress is True:
                print('\tProgression: ', str(round(it / iterations * 100,2)), '%', sep='', end='')
            # save every 2000 samples to disk
            if it % 2000 == 0:
                np.save(stored_T_file, self.T_samples)
                np.save(stored_Z_file, self.Z_samples)
                np.save(stored_ll_file, self.mll)
        return None
