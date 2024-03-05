#! /usr/bin/env python3
from distutils import command
import numpy as np

## Normalise numpy 1D array
def normalise(x):
    return x / np.sum(x)

## Simulate data from the topic model
def simulate_data(D, K=0, fixed_K = True, H=0, fixed_H = True, V=0, fixed_V = True, 
                    N_num=0, fixed_N = False, M_num=0, fixed_M = False, psi_dic=0, fixed_psi = False,
                    secondary_topic = False, shared_Z = True, command_level_topics = False, phi_last=True,
                    phi_dic=0, fixed_phi=False, fixed_prop=False, fixed_theta=False, theta=1.0,
                    distinct_psi=False, distinct_phi= False,
                    gamma=1.0, eta=1.0, alpha=1.0, alpha0=1.0, tau=1.0,
                    csi=1, omega=10, stick_truncation=100, seed=111):
    # Check if the provided value of seed is appropriate
    if (not isinstance(seed, int) or seed < 1):
        raise ValueError('seed must be an integer value larger or equal to 1.') 
    else:
        np.random.seed(seed)
    # Check if the provided value of D is appropriate
    if (not isinstance(D, int) or D < 1):
        raise ValueError('D must be an integer value larger or equal to 1.') 
    # Check if the provided value for K is appropriate
    if not isinstance(fixed_K, bool):
        return TypeError('fixed_K must be True or False.')
    else:
        if fixed_K and (not isinstance(K, int) or K < 2):
            raise ValueError('K must be an integer value larger or equal to 2.') 
    # Check if the provided value for H is appropriate
    if not isinstance(fixed_H, bool):
        return TypeError('fixed_H must be True or False.')
    else:
        if fixed_H and (not isinstance(H, int) or (command_level_topics and H < 2)):
            raise ValueError('H must be an integer value larger or equal to 2 if command-level topics are used.') 
    if isinstance(H, int) and H > 0 and not command_level_topics:
        raise ValueError('H can only be specified when command-level topics are used. Proposed solution: initialise H=0.')
    # Check if the provided value for V is appropriate
    if not isinstance(fixed_V, bool):
        raise TypeError('fixed_V must be True or False.')
    else:
        if not isinstance(V, int) or V < 2:
            raise ValueError('V must be an integer value larger or equal to 2.')
    # Prior parameters
    if isinstance(csi, float) or isinstance(csi, int):
        if not csi > 0:
            raise ValueError('The prior parameters csi must be positive.')
    else: 
        raise TypeError('The prior parameter csi must be a float or integer.')
    if isinstance(omega, float) or isinstance(omega, int):
        if not omega > 0:
            raise ValueError('The prior parameters omega must be positive.')
    else: 
        raise TypeError('The prior parameter omega must be a float or integer.')
    if isinstance(gamma, float) or isinstance(gamma, int):
        if not gamma > 0:
            raise ValueError('The prior parameters gamma must be positive.')
    else: 
        raise TypeError('The prior parameter gamma must be a float or integer.')
    if isinstance(eta, float) or isinstance(eta, int):
        if not eta > 0:
            raise ValueError('The prior parameters eta must be positive.')
    else: 
        raise TypeError('The prior parameter eta must be a float or integer.')
    # Secondary topics
    if not isinstance(secondary_topic, bool):
        raise TypeError('secondary_topic must be True or False.')
    else:
        if isinstance(alpha, float) or isinstance(alpha, int):
            if not alpha > 0:
                raise ValueError('The prior parameters alpha must be positive.')
        else: 
            raise TypeError('The prior parameter alpha must be a float or integer.')        
        if isinstance(alpha0, float) or isinstance(alpha0, int):
            if not alpha0 > 0:
                raise ValueError('The prior parameters alpha0 must be positive.')
        else:
            raise TypeError('The prior parameter alpha0 must be a float or integer.')
        if not isinstance(fixed_theta, bool):
            raise TypeError('fixed_theta must be True or False.')
        else:
            if not isinstance(theta, float) or theta > 1 or theta < 0:
                raise TypeError('theta must be a float between 0 and 1.')
    # Command-level topics
    if not isinstance(command_level_topics, bool):
        raise ValueError('command_level_topics must be True or False.')
    else:
        if isinstance(tau, float) or isinstance(tau, int):
            if not tau > 0:
                raise ValueError('The prior parameters tau must be positive.')
        else:
            raise TypeError('The prior parameter tau must be a float or integer.')
    # Sample the number of commands and words for each session or use fixed values for those N_num, M_num
    if fixed_N:
        N = np.repeat(N_num,D)
    else:
        N = np.random.poisson(lam=csi, size=D) + 1
    M = {}
    if fixed_M:
        for d in range(D):
            M[d] = np.repeat(M_num,N[d])
    else:
        for d in range(D):
            M[d] = np.random.poisson(lam=omega, size=N[d]) + 1
    # Sample the session-level allocations
    if fixed_K:
        if fixed_prop: 
            lam = np.ones(K) / K
        else:
            lam = np.random.dirichlet(alpha=np.ones(K)*gamma)
    else:
        # Use stick-breaking representation of Dirichlet process
        b = np.random.beta(a=1, b=gamma, size=stick_truncation)
        lam = np.ones(stick_truncation)
        lam[0] = b[0]
        lam[1:-1] = b[1:-1] * np.cumprod(1-b)[:-2]
        lam[-1] = 1 - np.sum(lam[:-1])
    # Sample t
    t = np.random.choice(K if fixed_K else stick_truncation, size=D, p=lam)
    # Sample phi
    if not command_level_topics:
        rr = range((K if fixed_K else stick_truncation) + (1 if secondary_topic else 0))
    else:
        rr = range((H if fixed_H else stick_truncation) + (1 if secondary_topic else 0))
    if fixed_phi: # for given dictionary of probs as argument
        phi = phi_dic
    else:
        if distinct_phi: # avoid overlap when not given dictionary as argument
            phi_sample = np.random.dirichlet(alpha=np.ones(V)*eta,size = len(rr)*10) 
            v_ind = [np.argmax(ii) for ii in phi_sample] # get index for topic h where peak for each sample drawn 
            uniq_v = np.unique(v_ind) # get unique topic h for which samples peak
            sample_ind_v = [v_ind.index(ii) for ii in uniq_v[0:len(rr)]]
            phi = phi_sample[sample_ind_v]
        else:    # allow randomeness
            phi = {}
            for k in rr:
                if not fixed_V:
                    b = np.random.beta(a=1, b=eta, size=stick_truncation)
                    phi[k] = np.ones(stick_truncation)
                    phi[k][0] = b[0]
                    phi[k][1:-1] = b[1:-1] * np.cumprod(1-b)[:-2]
                    phi[k][-1] = 1 - np.sum(phi[k][:-1])
                else:
                    phi[k] = np.random.dirichlet(alpha=np.ones(V)*eta)         
    if command_level_topics:
        if fixed_psi:
            psi=psi_dic
        else:
            if distinct_psi:
                psi_sample = np.random.dirichlet(alpha=np.ones(H)*tau,size = K*10)
                h_ind = [np.argmax(ii) for ii in psi_sample] # get index for topic h where peak for each sample drawn 
                uniq_h = np.unique(h_ind) # get unique topic h for which samples peak
                sample_ind_h = [h_ind.index(ii) for ii in uniq_h[0:K]]
                psi = psi_sample[sample_ind_h]
            else:    
                psi = {}
                for k in range(K if fixed_K else stick_truncation):
                    if not fixed_H:
                        b = np.random.beta(a=1, b=tau, size=stick_truncation)
                        psi[k] = np.ones(stick_truncation)
                        psi[k][0] = b[0]
                        psi[k][1:-1] = b[1:-1] * np.cumprod(1-b)[:-2]
                        psi[k][-1] = 1 - np.sum(phi[k][:-1])
                    else:
                        psi[k] = np.random.dirichlet(alpha=np.ones(H)*tau)
        # Sample s
        s = {}
        for d in range(D):
            s[d] = np.random.choice(H if fixed_H else stick_truncation, size=N[d], p=psi[t[d]])
    # Sample theta for secondary topics
    if secondary_topic:
        if not shared_Z: 
            if not fixed_theta:
                theta = np.random.beta(a=alpha, b=alpha0, size=D)
            else:
                theta = theta * np.ones(D)
        else:
            if not command_level_topics:
                if not fixed_theta:
                    theta = np.random.beta(a=alpha, b=alpha0, size=K if fixed_K else stick_truncation)
                else:
                    theta = theta * np.ones(K if fixed_K else stick_truncation)
            else:
                if not fixed_theta:
                    theta = np.random.beta(a=alpha, b=alpha0, size=H if fixed_H else stick_truncation)
                else:
                    theta = theta * np.ones(H if fixed_H else stick_truncation)
    # Sample the words
    w = {}
    if secondary_topic:
        z = {}
    for d in range(D):
        w[d] = {}
        if secondary_topic:
            z[d] = {}
        for j in range(N[d]):
            if secondary_topic:
                w[d][j] = np.zeros(M[d][j], dtype=int)
            if not secondary_topic:
                w[d][j] = np.random.choice(V if fixed_V else stick_truncation, size=M[d][j], p=phi[s[d][j] if command_level_topics else t[d]])
            else:
                if command_level_topics:
                    z[d][j] = np.random.choice(2, size=M[d][j], p=[1-theta[s[d][j] if shared_Z else d], theta[s[d][j] if shared_Z else d]])
                else:
                    z[d][j] = np.random.choice(2, size=M[d][j], p=[1-theta[t[d] if shared_Z else d], theta[t[d] if shared_Z else d]])
                w[d][j][z[d][j] == 0] = np.random.choice(V if fixed_V else stick_truncation, size=np.sum(1-z[d][j]), p=phi[list(phi.keys())[-1]] if phi_last else phi[0])
                w[d][j][z[d][j] == 1] = np.random.choice(V if fixed_V else stick_truncation, size=np.sum(z[d][j]), p=phi[(s[d][j] if command_level_topics else t[d]) + (0 if phi_last else 1)])     
    # Define output
    out = {}
    out['t'] = t
    out['N'] = N
    out['M'] = M
    out['w'] = w
    out['lambda'] = lam
    out['phi'] = phi
    if command_level_topics:
        out['s'] = s
        out['psi'] = psi
    if secondary_topic:
        out['z'] = z
        out['theta'] = theta
    return out 
