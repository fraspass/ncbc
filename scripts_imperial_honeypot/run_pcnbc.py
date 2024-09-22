#! /usr/bin/env python3
import pickle
import ncbc

## Import data
with open('../data/ICL_data_10_01.pkl', 'rb') as f:
    x = pickle.load(f)

## Import vocabulary
with open('../data/ICL_wordmap_10_01_redacted.pkl', 'rb') as f:
    V = pickle.load(f)

# Wa is a dictionary containing the first words for each command in each document (parent words)
Wa = {}
for d in x:
    Wa[d] = [x[d][j][0] for j in range(len(x[d]))]

# Wc is a dictionary containing the remaining words for each command in each document (child words)
Wc = {}
for d in x:
    Wc[d] = {}
    for j in range(len(x[d])):
        Wc[d][j] = x[d][j][1:]

## Create a directory for results if not already existing
import os
results_path = 'results_pcnbc/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

## Define a NCBC topic model object
m = ncbc.parent_child_model(W_a=Wa, W_c=Wc, K=30, H=50, tau=0.1, chi=0.1, gamma=0.1, eta=0.1)

## Initialise the model via spectral clustering
m.spectral_init()

## Perform inference via MCMC
ll = m.MCMC(iterations=100000, burnin=10000, size=100, verbose=True, jupy_out=False,
              return_t=True, return_u=True, calculate_ll=True, thinning=50)

## Save results
with open(results_path + 'icl_res_pcnbc_diri_chain_30_50.pkl', 'wb') as f:
        pickle.dump(ll, f)
        
with open(results_path + 'icl_res_pcnbc_diri_model_30_50.pkl', 'wb') as f:
        pickle.dump(m, f)
