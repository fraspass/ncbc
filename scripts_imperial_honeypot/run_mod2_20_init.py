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
results_path = 'results_mod2/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

## Define the model
m = ncbc.topic_model(W=x, K=20, fixed_V=True, secondary_topic=True,
                command_level_topics=False, gamma=0.1, eta=1.0, numpyfy=True)

## Import the clustering initialisation from previous model
with open('results_mod1_v1/icl_res_mod1_clust.pkl', 'rb') as f:
    U = pickle.load(f)

## Initialise the model via custom initialisation
U[100] = 16
m.custom_init(t=U); 

## Run the MCMC
ll = m.MCMC(iterations=100000, burnin=10000, size=250, verbose=True,
            calculate_ll=True, random_allocation=False,
            jupy_out=False, return_t=True, return_s=True, return_z=True, thinning=50)

## Save the results
with open(results_path + 'icl_res_mod2_diri_chain_20_init.pkl', 'wb') as f:
        pickle.dump(ll, f)
        
with open(results_path + 'icl_res_mod2_diri_model_20_init.pkl', 'wb') as f:
        pickle.dump(m, f)