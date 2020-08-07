# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:57:24 2018
Created for gastric cancer immunotherapy project.
@author: bnl2108
"""

# Simulated annealing algorithm
# Functions:
    # generate_new_params: creates random t_matrix
    # gof: compares model output to target data (chi-square test)
    # acceptance_prob: gives acceptance probability based on change in gof and T
        # (higher T -> lower prob -> less likely to accept; and vice versa)
    # anneal_markov: runs the model to get the output
    # anneal: runs the simulated annealing

# NOTES:
    # mutation state: everyone starts here
    # nono state: healthy state for natural history model
    # current state: healthy state for running the current recommended screening schedule
    # new state: healthy state for running the new screening schedules
    # There are different transition probabilities for transitioning out of healthy states depending on
        # screening schedule
        # The transition probabilities out of the new state are proportions of the transition
            # probabilities out of the current state
    # No transitions between cancer stages
    # The adenoma stage can be skipped because it's not always detected before cancer development
    # Calibrate each genotype separately
    # What will our target data be? Adenoma incidence by age, cancer incidence by age (for each stage?),
    #   cancer death by age? Each of these specific to the genotype?
    # Which transition probabilities are set vs calibrated? Look at connectivity matrix as guide.
        # Set: all-cause mortality
        # Calibrated: all others
    # Will we only be calibrating the natural history?
        # We can calibrate the existence of true states.
        # The screening model would then have likelihoods for detecting true states.
    # How to decide randomization bounds for transition probabilities?

# OUTLINE:
    # generate_rand_t_matrix() will generate an initial transition matrix for the simulated 
    #   annealing algorithm. Some transition probabilities may be fixed (depending on data).
    #   Other transition probabilities will be randomly selected within bounds.
    # Create a run_markov() function in lynch_simulator.py that inputs the t_matrix and can be
    #   used for both running the model and running the calibration. Outputs distribution matrix.
    # Create numpy arrays of target data (adenoma, cancer, and death?)
    # Subset distrubition matrix to create numpy arrays of observed data corresponding to target data.
    # Calculate gof between each set of target and observed data and sum.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lynch_presets as ps
import lynch_simulator as sim

sim_anneal_params = {'starting_T': 1.0,
                    'final_T': 0.001,
                    'cooling_rate': 0.9,
                    'iterations': 100}

def generate_rand_t_matrix(time): 
    # Inputs: dictionaries of states and connectivity, list of time values
    # Output: somewhat random t_matrix
    
    # ASSUMPTIONS:
        # Probabilities of remaining in SD and PD states increase over time
        # Probability of dying from cancer decreases over time
        # Probability of progressing decreases over time
    
    states = ps.ALL_STATES
    connect = ps.CONNECTIVITY

    # Generate c_matrix:
    c_matrix = sim.dict_to_connect_matrix(states, connect)

    # Initiate t_matrix
    t_matrix = np.zeros((len(states), len(states)))
    t_matrix[0][1] = 1
    
    # Set constant probs
    PD_c_death = 0.1476125 # remains constant across all treatment arms (bsc -> cancer death)
    SD_c_death = np.random.uniform(0, PD_c_death) 
        # unknown but remains constant over time
        # should be smaller than PD_c_death
    # TRAE death probs defined in loop below
    
    # Set first random probs
    SD = np.random.uniform(0.5, 1) # high prob of staying in SD
    progress = np.random.uniform(0, 0.5)
    PD = np.random.uniform(0.5, 1) # high prob of staying in PD
    # Set second random probs
    SD2 = np.random.uniform(SD, 1)
    progress2 = np.random.uniform(0, progress)
    PD2 = np.random.uniform(PD, 1)
    # Set third random probs
    SD3 = np.random.uniform(SD2, 1)
    progress3 = np.random.uniform(0, progress2)
    PD3 = np.random.uniform(PD2, 1)
    
    for t in range(1, len(time)):
        t_layer = c_matrix
        
        # TRAE death
        if t <= 6:
            trae_death = params['prob trae death']
        else:
            trae_death = 0
            
        # Three time segments selected somewhat arbitrarily (but roughly follows KM curves)
        # First time period (0-3 months)
        if t <= 3:
            t_layer[1,1] = SD
            t_layer[1,2] = use.rate_to_prob((use.prob_to_rate(progress, 1) + use.prob_to_rate(params['prob discontinue'], 1)), 1) 
                # adds discontinuation due to TRAEs
                # convert probabilities to rates before adding -> convert back to probs
            t_layer[2,2] = PD
        # Second time period (3-10 months)
        elif t > 3 and t <= 10:
            t_layer[1,1] = SD2
            t_layer[1,2] = use.rate_to_prob((use.prob_to_rate(progress2, 1) + use.prob_to_rate(params['prob discontinue'], 1)), 1) 
                # add discontinuation due to TRAEs
            t_layer[2,2] = PD2
        else: # probs of progressing and dying decrease
            t_layer[1,1] = SD3
            t_layer[1,2] = progress3 # assume no discontinuation due to TRAEs after 10 months
            t_layer[2,2] = PD3
            
        # Normalize probs before adding constant probabilities
        # Set all constant probabilities to zero initially
        t_layer[1, 3] = 0
        t_layer[1, 4] = 0
        t_layer[2, 4] = 0
        t_layer[1, 5] = 0
        t_layer[2, 5] = 0
        
        # Normalize random probs so that they add up to 1 after constant probs are added
        t_layer[1] = use.normalize_choose(t_layer[1], 
               1 - ps.cut_ac_mortality[t] - SD_c_death - trae_death)
        t_layer[2] = use.normalize_choose(t_layer[2], 
               1 - ps.cut_ac_mortality[t] - PD_c_death)
        
         # Known probabilities that remain constant
        # TRAE death (assume only in the first 6 months, and only in SD)
        t_layer[1, 3] = trae_death
        # All-cause death
        t_layer[1, 4] = ps.cut_ac_mortality[t]
        t_layer[2, 4] = ps.cut_ac_mortality[t]
        # Cancer-related death
        t_layer[1, 5] = SD_c_death
        t_layer[2, 5] = PD_c_death
        
        # Check that rows add up to 1
        # print(sum(t_layer[1]))
        # print(sum(t_layer[2]))

        t_matrix = np.vstack((t_matrix, t_layer))

    # Create 3D matrix
    t_matrix = np.reshape(t_matrix, (len(time), len(states), len(states)))
    # print(t_matrix[5])

    return t_matrix

#t_matrix = generate_new_params(ps.ALL_STATES, ps.CONNECTIVITY, range(20), 'Pembro MSI-H')
#print(t_matrix)
    
def anneal_markov(states, time, t_matrix):
    # Inputs: 
        # states dictionary
        # list of time values
        # transition probabilities matrix
    # Output: overall survival over time (model output)
    
    # Initialize distribution matrix to store results
        # Contains proportion of population in each state at each time point
        # Dimensions: number of cycles x number of states
    d_matrix = np.zeros((len(time), len(states)))
    OS_model = list()
    PFS_model = list()
    
    # Fill in d_matrix over time (matrix multiplication):    
    for t in range(len(time)): # for every cycle
        # Initial distribution:
        if t == 0:
            temp = ps.IC * np.transpose(t_matrix[t])
            # Transposition necessary to multiply d_matrix along columns of t_matrix
        else:
            temp = d_matrix[t-1]*np.transpose(t_matrix[t])
            
        d_slice = np.sum(temp, axis=1) # sums down column
        # The sum down the column gives the proportion of the population in that state
        d_matrix[t] = d_slice

        OS_model.append(sum(d_slice[n] for n in [1,2]))
        PFS_model.append(d_slice[1])

    OS_model = np.asarray(OS_model)
    PFS_model = np.asarray(PFS_model)

    return OS_model, PFS_model

def gof(obs, exp):
    # chi-squared
    # inputs: numpy arrays of observed and expected values
    chi = ((obs-exp)**2)/exp
    chi_sq = sum(chi)
    return chi_sq

def acceptance_prob(old_gof, new_gof, T):
    return np.exp((old_gof - new_gof) / T)

def anneal(Tx, Tx_bio, time):
    # Inputs: treatment name (without biomaker specification), treatment name 
        # (with biomarker specification), list of time values
    # Output: calibrated transition matrix
    
    # Initiate dictionary for output
#    all_t_matrices = {'gof': [], 't_matrices': [], 'OS_model': [], 'PFS_model': []}
    
    # find first solution
    first_params = generate_new_params(ps.ALL_STATES, ps.CONNECTIVITY, time, Tx)
    OS_model, PFS_model = anneal_markov(ps.ALL_STATES, time, first_params)
    model_out = np.append(OS_model, PFS_model)
#    print(len(model_out))
    
    # get trial data
    trial_out = km.trial_data[Tx_bio]
    KM_OS = trial_out[0]
    KM_PFS = trial_out[1]
    trial_out = np.append(KM_OS, KM_PFS)
#    print(len(trial_out))
    old_gof = gof(model_out, trial_out)
    best_params = first_params
    
    T = sim_anneal_params['starting_T']

    # start temperature loop
    # annealing schedule
    while T > sim_anneal_params['final_T']:

        # sampling at T
        for i in range(sim_anneal_params['iterations']):

            # find new candidate solution
            new_params = generate_new_params(ps.ALL_STATES, ps.CONNECTIVITY, time, Tx)
            OS_model, PFS_model = anneal_markov(ps.ALL_STATES, time, new_params)
            model_out = np.append(OS_model, PFS_model)
                
            new_gof = gof(model_out, trial_out)
            ap =  acceptance_prob(old_gof, new_gof, T)
                
            # decide if the new solution is accepted
            if np.random.uniform() < ap:
                best_params = new_params
                old_gof = new_gof
#                final_OS = OS_model
#                final_PFS = PFS_model
                print(T, i)
                plt.plot(time, KM_OS, 'o')
                plt.plot(time, KM_PFS, 'o')
                plt.plot(time, OS_model)
                plt.plot(time, PFS_model)
                plt.show()
#                all_t_matrices['gof'].append(old_gof)
#                all_t_matrices['t_matrices'].append(best_params)
#                all_t_matrices['OS_model'].append(OS_model)
#                all_t_matrices['PFS_model'].append(PFS_model)

        T = T * sim_anneal_params['cooling_rate']
        
#    print(T, i)
#    plt.plot(time, KM_OS, 'o')
#    plt.plot(time, KM_PFS, 'o')
#    plt.plot(time, final_OS)
#    plt.plot(time, final_PFS)
#    plt.show()
    
    return best_params#, all_t_matrices

#anneal('Pembro', 'Pembro MSI-H', range(20))

def anneal_only_OS(Tx, Tx_bio, time):
    # Use when only OS data (and not PFS data) is available
    
    # Initiate dictionary for output
#    all_t_matrices = {'gof': [], 't_matrices': [], 'OS_model': [], 'PFS_model': []}
    
    # find first solution
    first_params = generate_new_params(ps.ALL_STATES, ps.CONNECTIVITY, time, Tx)
#    print('t_matrix:', len(first_params))
    OS_model, PFS_model = anneal_markov(ps.ALL_STATES, time, first_params)
    model_out = OS_model # only use OS for goodness of fit comparison
    
    trial_out = km.trial_data[Tx_bio]
    KM_OS = trial_out
    old_gof = gof(model_out, trial_out)
    best_params = first_params
    
    T = sim_anneal_params['starting_T']

    # start temperature loop
    # annealing schedule
    while T > sim_anneal_params['final_T']:

        # sampling at T
        for i in range(sim_anneal_params['iterations']):

            # find new candidate solution
            new_params = generate_new_params(ps.ALL_STATES, ps.CONNECTIVITY, time, Tx)
            OS_model, PFS_model = anneal_markov(ps.ALL_STATES, time, new_params)
            model_out = OS_model 
                
            new_gof = gof(model_out, trial_out)
            ap =  acceptance_prob(old_gof, new_gof, T)
                
            # decide if the new solution is accepted
            if np.random.uniform() < ap:
                best_params = new_params
                old_gof = new_gof
                final_OS = OS_model
                final_PFS = PFS_model
#                all_t_matrices['gof'].append(old_gof)
#                all_t_matrices['t_matrices'].append(best_params)
#                all_t_matrices['OS_model'].append(OS_model)
#                all_t_matrices['PFS_model'].append(PFS_model)

        T = T * sim_anneal_params['cooling_rate']
    
#    print(T, i)
    plt.plot(time, KM_OS, 'o')
    plt.plot(time, final_OS)
    plt.plot(time, final_PFS)
    plt.show()
    
    return best_params#, all_t_matrices

def save_t_matrix(t_matrix_path, t_matrix):
    # Input: path to transition matrix npy file; output from anneal function
    np.save(t_matrix_path, t_matrix)
    
def run_all_sim_anneal(Tx_bio, time):
    # Note: Assumes only running different pembro arms
    # Inputs: list of treatments with biomarker specification, time range
    
    # Set conditions for comparisons between different treatment transition probs
    SD_c_death_probs = {'Pembro PDL1': 0, 'Pembro PDL1 10': 1, 'Pembro MSI-H': 2}
    progress2_probs = {'Pembro PDL1': 0, 'Pembro PDL1 10': 1, 'Pembro MSI-H': 2}
    t_matrices = {'Pembro PDL1': [], 'Pembro PDL1 10': [], 'Pembro MSI-H': []}
    t_matrix = np.load('.\\Transition Matrices\\pembro_pdl1.npy')
    SD_c_death_probs['Pembro PDL1'] = t_matrix[1,1,5]
    progress2_probs['Pembro PDL1'] = t_matrix[4,1,2]
    t_matrices['Pembro PDL1'] = t_matrix
    t_matrix = np.load('.\\Transition Matrices\\pembro_msih.npy')
    SD_c_death_probs['Pembro MSI-H'] = t_matrix[1,1,5]
    progress2_probs['Pembro MSI-H'] = t_matrix[4,1,2]
    t_matrices['Pembro MSI-H'] = t_matrix
    
    while (SD_c_death_probs['Pembro PDL1'] < SD_c_death_probs['Pembro PDL1 10'] or
           SD_c_death_probs['Pembro PDL1 10'] < SD_c_death_probs['Pembro MSI-H'] or
           progress2_probs['Pembro PDL1'] < progress2_probs['Pembro PDL1 10'] or
           progress2_probs['Pembro PDL1 10'] < progress2_probs['Pembro MSI-H']):
#    SD_c_death_probs = {'Pembro PDL1': 0, 'Pembro MSI-H': 2}
#    t_matrices = {'Pembro PDL1': [], 'Pembro MSI-H': []}
#    while (SD_c_death_probs['Pembro PDL1'] < SD_c_death_probs['Pembro MSI-H']):
        
#        for Tx in Tx_bio:
#            if Tx == 'Pembro PDL1 10':
#                t_matrix = anneal_only_OS('Pembro', Tx, time)
#                t_matrices[Tx] = t_matrix
#                SD_c_death_probs[Tx] = t_matrix[1,1,5]
#            else:
#                t_matrix = anneal('Pembro', Tx, time)
#                t_matrices[Tx] = t_matrix
#                SD_c_death_probs[Tx] = t_matrix[1,1,5]
        t_matrix = anneal_only_OS('Pembro', Tx_bio, time)
        t_matrices[Tx_bio] = t_matrix
        SD_c_death_probs[Tx_bio] = t_matrix[1,1,5]
        progress2_probs[Tx_bio] = t_matrix[4,1,2]
                
        print(SD_c_death_probs)
        print(progress2_probs)
                
#    save_t_matrix('.\\Transition Matrices\\pembro_pdl1.npy', t_matrices['Pembro PDL1'])
    save_t_matrix('.\\Transition Matrices\\pembro_pdl1_10.npy', t_matrices['Pembro PDL1 10'])
#    save_t_matrix('.\\Transition Matrices\\pembro_msih.npy', t_matrices['Pembro MSI-H'])
    
    return SD_c_death_probs

#print(run_all_sim_anneal(['Pembro PDL1', 'Pembro MSI-H'], range(20)))
#run_all_sim_anneal('Pembro PDL1 10', range(20))

def process_t_matrices(all_t_matrices, df, Tx):
    # Inputs: output from SA_sim_anneal 
        #(dictionary of gof scores and t_matrices for one treatment), 
        # dataframe to be updated for all treatments,
        # treatment you're on
    
    # Convert dictionary to pandas dataframe
    all_t_matrices = pd.DataFrame(data=all_t_matrices)
    # Order by increasing gof score
    sorted_t_matrices = all_t_matrices.sort_values(by=['gof'])
    print('number of solutions:', len(sorted_t_matrices['t_matrices']))
    
    # Make lists of SD_cancer death probabilities, and differences between model 
        # outputs and trial KM curves
    SD_c_death_probs = list()
    sum_diff_OS = list()
    sum_diff_PFS = list()
    trial_out = km.trial_data[Tx]
    if len(trial_out) == 2:
        KM_OS, KM_PFS = trial_out[0], trial_out[1]
    else:
        KM_OS = trial_out
    for i in range(len(sorted_t_matrices)):
        t_matrix = sorted_t_matrices['t_matrices'].iloc[i]
        SD_c_death_probs.append(t_matrix[1,1,5])
        OS_model = sorted_t_matrices['OS_model'].iloc[i]
        PFS_model = sorted_t_matrices['PFS_model'].iloc[i]
        sum_diff_OS.append(sum(OS_model - KM_OS))
        if len(trial_out) == 2: 
            sum_diff_PFS.append(sum(PFS_model - KM_PFS))
        else: # just add up all data points
            sum_diff_PFS.append(sum(PFS_model))
        
    # Select SD_death prob from best fit, and find max and min values
    best_prob = SD_c_death_probs[0]
    pos_error = max(SD_c_death_probs) - best_prob
    neg_error = best_prob - min(SD_c_death_probs)
    df.loc[Tx] = {'best prob': best_prob, 'pos error': pos_error,
                  'neg error': neg_error}
    print('best prob:', best_prob)
    print('max prob:', max(SD_c_death_probs))
    print('min prob:', min(SD_c_death_probs))
    
    # Determine maximum differences between calibrated models and KM curves
    ind1 = sum_diff_OS.index(max(sum_diff_OS))
    pos_error_OS = sorted_t_matrices['OS_model'].iloc[ind1]
    ind2 = sum_diff_OS.index(min(sum_diff_OS))
    neg_error_OS = sorted_t_matrices['OS_model'].iloc[ind2]
    ind3 = sum_diff_PFS.index(max(sum_diff_PFS))
    pos_error_PFS = sorted_t_matrices['PFS_model'].iloc[ind3]
    ind4 = sum_diff_PFS.index(min(sum_diff_PFS))
    neg_error_PFS = sorted_t_matrices['PFS_model'].iloc[ind4]
        
    # Graph range of OS curves and range of PFS curves
    plt.plot(range(20), KM_OS)
    plt.plot(range(20), pos_error_OS)
    plt.plot(range(20), neg_error_OS)
    plt.title(Tx+' OS')
    plt.show()
    if len(trial_out) == 2: plt.plot(range(20), KM_PFS)
    plt.plot(range(20), pos_error_PFS)
    plt.plot(range(20), neg_error_PFS)
    plt.title(Tx+' PFS')
    plt.show()

#df = pd.DataFrame(columns=['best prob', 'pos error', 'neg error'])
#process_t_matrices(all_t_matrices, df, 'Pembro PDL1')
#df.to_excel('.\\output\\test.xlsx')
  
def SA_post_processing(Tx_bio, time):
    # Assumes only running different pembro arms
    # Run simulated annealing for all treatments
    # Save best prob, positive error, and negative error for each treatment
        # in an excel sheet
    df = pd.DataFrame(columns=['best prob', 'pos error', 'neg error'])
    for bio in Tx_bio:
        if bio == 'Pembro PDL1 10':
            __, all_t_matrices = anneal_only_OS('Pembro', bio, time)
        else:
            __, all_t_matrices = anneal('Pembro', bio, time)
        process_t_matrices(all_t_matrices, df, bio)
    df.to_excel('.\\output\\SA_SD_death_probs.xlsx')
        
#SA_post_processing(['Pembro PDL1', 'Pembro PDL1 10', 'Pembro MSI-H'], range(20))
    
        

#anneal('.\\Transition Matrices\\test.npy', 'Pembro', 'Pembro MSI-H', range(20))
#anneal('.\\Transition Matrices\\Rampac_t_matrix_simAnneal.npy', 'Ram+Pac', 'Ram+Pac', range(20))
#anneal('.\\Transition Matrices\\MSI_H_t_matrix_simAnneal5.npy', 'Pembro', 'Pembro MSI-H', range(20))
#anneal_only_OS('.\\Transition Matrices\\pdl1_10_t_matrix_simAnneal5.npy', 'Pembro', 'Pembro PDL1 10', range(20))
#t_matrix = anneal('BSC', 'BSC', range(20))
#save_t_matrix('.\\Transition Matrices\\bsc_t_matrix_simAnneal.npy')
#anneal('.\\Transition Matrices\\pembro_t_matrix_simAnneal.npy', 'Pembro', 'Pembro', range(20))
#anneal('.\\Transition Matrices\\pac_pdl1_t_matrix.npy', 'Pac', 'Pac PDL1', range(20))
#t_matrix = anneal('Pac', 'Pac', range(20))
#save_t_matrix('.\\Transition Matrices\\pac_simAnneal.npy', t_matrix)
#anneal_only_OS('.\\Transition Matrices\\pac_msih_simAnneal.npy', 'Pac', 'Pac', range(20))
#t_matrix = anneal('Pembro', 'Pembro', range(25))
#save_t_matrix('.\\Transition Matrices\\pembro_simAnneal.npy', t_matrix)
#t_matrix = anneal('Pac', 'Pac PDL1-', range(25))
#save_t_matrix('.\\Transition Matrices\\pac_pdl1_neg.npy', t_matrix)
#print(anneal())
#anneal()
