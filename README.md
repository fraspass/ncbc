# Nested Dirichlet models for unsupervised attack pattern detection in honeypot data

This repository contains a _Python_ library supporting the paper *Sanna Passino, F., Mantziou A., Ghani, D., Thiede, P., Bevington, R. and Heard, N. A. (2023) "Nested Dirichlet models for unsupervised attack pattern detection in honeypot data"*, available as a preprint on [arXiv](https://arxiv.org/abs/2301.02505). 

The library `ncbc` can be installed in edit mode as follows:
```
pip3 install -e lib/
```
The library can then be imported in any _Python_ session:
```python3
import ncbc
```

## The library `ncbc` 

A quick demo on how to use the library can be found in `notebooks/test_library.ipynb`.

The core of the library is the class `topic_model` contained in the *Python* script `lib/ncbc/topic_model.py`. The class `topic_model` contains instances for tasks related to the models presented in the paper. According to the class parameters, different models can be obtained. The class can be initialised with the following arguments: 
* `W`: a dictionary of dictionaries, where each key corresponds to the session, with second-level keys denoting the commands. Each command is represented by a sequence of integers mapped to a vocabulary;
* `K`: an integer representing the number of session-level topics;
* `H`: an integer representing the number of command-level topics;
* `V`: an integer representing the vocabulary size (default: `V=0`). If `V=0` is used (or any negative value), then `V` is set to the number of unique observed words in `W`;
* `fixed_V`: a Boolean variable (`True` or `False`, default: `fixed_V=True`) denoting whether the number of words in the vocabulary should be treated as *unknown* or *known and fixed*. In the former case, a GEM prior is used, whereas for the latter, a Dirichlet distribution is used;
* `secondary_topic`: a Boolean variable (`True` or `False`, default: `secondary_topic=False`), denoting whether secondary topics should be used;
* `shared_Z`: a Boolean variable (`True` or `False`, default: `shared_Z=True`), indicating if probabilities associated with the occurrence of secondary topics are shared across session-level topics (`True`) or are different for each session (`False`);
* `command_level_topics`: a Boolean variable (`True` or `False`, default: `command_level_topics=False`), indicating if command-level topics are used;
* `gamma`: a float representing the hyperparameter for the Dirichlet prior on session-level topic distributions;
* `tau`: a float representing the hyperparameter for the Dirichlet prior on command-level topic distributions;
* `eta`: a float representing the hyperparameter for the Dirichlet prior on word distributions;
* `alpha`: a float representing the first hyperparameter for the Beta prior on the secondary topic probabilities;
* `alpha0`: a float representing the second hyperparameter for the Beta prior on the secondary topic probabilities;
* `lambda_gem`: a Boolean variable (default: `lambda_gem=False`), indicating if GEM prior is used for session-level topics;
* `psi_gem`: a Boolean variable (default: `psi_gem=False`), indicating if GEM prior is used for command-level topics;
* `phi_gem`: a Boolean variable (default: `phi_gem=False`), indicating if GEM prior is used for word distributions.

After the model is defined via the parameters above, it is possible to run initialisation procedures for the Markov Chain Monte Carlo sampler used for Bayesian inference. The possible options for initialisation are: 
* `init_from_other`: initialise the model from the state of another `topic_model` object;
* `custom_init`: initialise the chain at given values of `t`, `s` and `z`, representing the session-level topics, command-level topics, and secondary topic indicators;
* `random_init`: initialise uniformly at random;
* `gensim_init`: initialise using the output of Latent Dirichlet Allocation, fitted via `gensim`;
* `spectral_init`: initialise via spectral clustering. 

Next, the MCMC procedure could be run on an initialised `topic_model` object via the instance `MCMC`. The instance `MCMC` utilises five main functions: 
* `resample_session_topics`, used to resample session-level topics given the other model parameters;
* `resample_command_topics`, used to resample command-level topics given the other model parameters;
* `resample_indicators`, used to resample secondary topic indicators;
* `split_merge_session`, used to propose a split-merge move for session-level topics;
* `split_merge_command`, used to propose a split-merge move for command-level topics.
The instance `MCMC` has a number of parameters which can be used to control, for example, the number of samples (`iterations`), the burn-in period (`burnin`) and the thinning (`thinning`).

Note that the PCNBC model can be fitted only via a different class, called `parent_child_model`, available in the file `lib/ncbc/parent_child_model.py`. 

Also, the file `lib/ncbc/simulate_data.py` contains a function (`simulate_data`) which can be used to simulate data from NCBC, using similar parameters to the `topic_model` class described above. The file `lib/ncbc/utils.py` contains utility functions, particularly useful for the postprocessing and analysis of the results.

In conclusion, the files `lib/ncbc/preprocess_honeypot.py`, `lib/ncbc/extract_honeypot_data.py` and `lib/ncbc/clean_commands.py` contain commands used to preprocess honeypot data.