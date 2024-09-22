#! /usr/bin/env python3
import pickle
import ncbc

## Import data
with open('../data/ICL_data_10_01.pkl', 'rb') as f:
    x = pickle.load(f)

## Import vocabulary
with open('../data/ICL_wordmap_10_01_redacted.pkl', 'rb') as f:
    V = pickle.load(f)

## Create a directory for results if not already existing
import os
results_path = 'results_mod1/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

## Define a NCBC topic model object with K=30 session-level topics, no secondary topics and no command-level topics
m = ncbc.topic_model(W=x, K=30, fixed_V=True, secondary_topic=False, lambda_gem=True,
                command_level_topics=False, gamma=0.1, eta=1.0, numpyfy=True)

## Initialise the model via spectral clustering
m.spectral_init()

## Perform inference via MCMC
ll = m.MCMC(iterations=250000, burnin=50000, size=250, verbose=True, jupy_out=False,
              return_t=True, return_s=False, return_z=False, calculate_ll=True,
              random_allocation=False, thinning=50, track_moves=True)

## Save results
with open(results_path + 'icl_res_mod1_gem_chain.pkl', 'wb') as f:
        pickle.dump(ll, f)
        
with open(results_path + 'icl_res_mod1_gem_model.pkl', 'wb') as f:
        pickle.dump(m, f)
