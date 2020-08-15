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
    # There are different transition probabilities for transitioning out of healthy states depending on
        # screening schedule
        # The transition probabilities out of the new state are proportions of the transition
            # probabilities out of the current state
    # Calibrate each genotype separately
    # What will our target data be? Adenoma incidence by age, cancer incidence by age (for each stage?),
    #   cancer death by age? Each of these specific to the genotype?
    # Which transition probabilities are set vs calibrated? Look at connectivity matrix as guide.
        # Set: all-cause mortality, colonoscopy mortality, cancer mortality
        # Calibrated: all others
    # Will we only be calibrating the natural history?
        # We can calibrate the existence of true states.
        # The screening model would then have likelihoods for detecting true states.
    # How to decide randomization bounds for transition probabilities?

# TODO:
    # Edits sim.run_markov_simple to input t_matrix
    # create_t_matrix won't be needed anymore?
    # Subset model output by test_ages for comparison to target data
    # Review t_matrix that Myles sends and set semi-random transition probabilities
        # New plan: vary current t_matrices randomly by +/-30%

# OUTLINE:
    # generate_rand_t_matrix() will generate an initial transition matrix for the simulated 
    #   annealing algorithm. Some transition probabilities may be fixed (depending on data).
    #   Other transition probabilities will be randomly selected within bounds. Make sure to
    #   normalize probabilities so they add to 1.
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
import data_manipulation as dm
import gender as g
import probability_functions as pf

sim_anneal_params = {
    'starting_T': 1.0,
    'final_T': 0.001,
    'cooling_rate': 0.9,
    'iterations': 100}

# Ages to compare target and model output data
test_ages = range(25,61,5)

# Load target data
adenoma_target_raw = {
    'MLH1': pd.read_excel(ps.params_male, sheet_name = 'MLH1_Adenoma'),
    'MSH2': pd.read_excel(ps.params_male, sheet_name = 'MSH2_Adenoma'),
    'MSH6': pd.read_excel(ps.params_male, sheet_name = 'MSH6_Adenoma'),
    'PMS2': pd.read_excel(ps.params_male, sheet_name = 'PMS2_Adenoma')
}
adv_adenoma_target_raw = {
    'MLH1': pd.read_excel(ps.params_male, sheet_name = 'MLH1_Adv_Adenoma'),
    'MSH2': pd.read_excel(ps.params_male, sheet_name = 'MSH2_Adv_Adenoma'),
    'MSH6': pd.read_excel(ps.params_male, sheet_name = 'MSH6_Adv_Adenoma'),
    'PMS2': pd.read_excel(ps.params_male, sheet_name = 'PMS2_Adv_Adenoma')
}
crc_target_raw = {
    'MLH1': pd.read_excel(ps.params_male, sheet_name = 'MLH1', usecols = [0,1]),
    'MSH2': pd.read_excel(ps.params_male, sheet_name = 'MSH2', usecols = [0,1]),
    'MSH6': pd.read_excel(ps.params_male, sheet_name = 'MSH6', usecols = [0,1]),
    'PMS2': pd.read_excel(ps.params_male, sheet_name = 'PMS2', usecols = [0,1])
}

# Interpolate target data to standardize ages (every 5 years, ages 25-60) for comparison to model output
adenoma_target = dict()
adv_adenoma_target = dict()
crc_target = dict()
for gene in ps.genes:
    adn = adenoma_target_raw[gene]
    adenoma_target[gene] = np.interp(test_ages, adn.Age, adn.Cumul_prob)
    adv_adn = adv_adenoma_target_raw[gene]
    adv_adenoma_target[gene] = np.interp(test_ages, adv_adn.Age, adv_adn.Cumul_prob)
    crc = crc_target_raw[gene]
    crc_target[gene] = np.interp(test_ages, crc.Age, crc.Cumul_prob)

def select_new_prob(step, old_prob):
    new_prob = np.random.uniform(old_prob-old_prob*step, old_prob+old_prob*step)
    return new_prob

def generate_rand_t_matrix(run_spec, current_t_matrix, step): 
    # Input: run specifications, current transition matrix, proportion to change t_matrix (between 0 and 1)
    # Output: somewhat random t_matrix
    
    states = ps.ALL_STATES
    state_names = dm.flip(ps.ALL_STATES)
    time = ps.time
    # List of transitions to change
    trans_to_change = [
        # Normal -> adenoma
        [state_names[run_spec.guidelines], state_names['init adenoma']],
        [state_names[run_spec.guidelines], state_names['init adv adenoma']],
        # Normal -> cancer
        [state_names[run_spec.guidelines], state_names['init dx stage I']],
        [state_names[run_spec.guidelines], state_names['init dx stage II']],
        [state_names[run_spec.guidelines], state_names['init dx stage III']],
        [state_names[run_spec.guidelines], state_names['init dx stage IV']],
        # Adenoma -> advanced adenoma
        [state_names['init adenoma'], state_names['init adv adenoma']],
        # Adenoma -> cancer
        [state_names['init adenoma'], state_names['init dx stage I']],
        [state_names['init adenoma'], state_names['init dx stage II']],
        [state_names['init adenoma'], state_names['init dx stage III']],
        [state_names['init adenoma'], state_names['init dx stage IV']],
        # Advanced adenoma -> cancer
        [state_names['init adv adenoma'], state_names['init dx stage I']],
        [state_names['init adv adenoma'], state_names['init dx stage II']],
        [state_names['init adv adenoma'], state_names['init dx stage III']],
        [state_names['init adv adenoma'], state_names['init dx stage IV']],
    ]

    for t in time:
        t_layer = current_t_matrix[t,:,:].copy()
        for trans in trans_to_change:
            t_layer[trans[0], trans[1]] = select_new_prob(step, t_layer[trans[0], trans[1]])
        # Normalize probabilities, excluding probabilities that are not calibrated
        for row in range(14):
            # Normalize row, keeping all transitions to death states static
            pf.normalize_static(t_layer[row], [range(14,22)])
        # Normalize death state rows such that all transitions other than same -> same == 0
        for row in range(14, 18):
            pf.normalize(t_layer[row], row)
        # Stack t_layers to create full t_matrix
        if t == 0:
            t_matrix = t_layer
        else:
            t_matrix = np.vstack((t_matrix, t_layer))

    # Create 3D matrix
    t_matrix = np.reshape(t_matrix, (len(time), len(states), len(states)))

    return t_matrix

def gof(obs, exp):
    # chi-squared
    # inputs: numpy arrays of observed and expected values
    chi = ((obs-exp)**2)/exp
    chi_sq = sum(chi)
    return chi_sq

def acceptance_prob(old_gof, new_gof, T):
    return np.exp((old_gof - new_gof) / T)

def anneal(run_spec):
    
    # find first solution
    t_matrix = generate_rand_t_matrix(run_spec)
    d_matrix = sim.run_markov_simple(run_spec, t_matrix)

    # Calculate gof
    # TODO: fix subsetting by age (cumulative_sum_state does not output age)
    # Adenoma
    adn_model = sim.cumulative_sum_state(d_matrix, 'init adenoma')
    adn_model = adn_model[adn_model.age in test_ages]
    adn_gof = gof(adn_model, adenoma_target)
    # Advanced adenoma
    adv_adn_model = sim.cumulative_sum_state(d_matrix, 'init adv adenoma')
    adv_adn_model = adv_adn_model[adv_adn_model.age in test_ages]
    adv_adn_gof = gof(adv_adn_model, adv_adenoma_target)
    # Cancer
    cancer_states = [
        'init dx stage I', 'init dx stage II', 'init dx stage III', 'init dx stage IV']
    crc_model = np.zeros(len(test_ages))
    for state in cancer_states:
        temp = sim.cumulative_sum_state(d_matrix, state)
        temp = temp[temp.age in test_ages]
        crc_model += temp
    crc_gof = gof(crc_model, crc_target)
    old_gof = adn_gof + adv_adn_gof + crc_gof
    best_t_matrix = t_matrix
    
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

def save_t_matrix(t_matrix_path, t_matrix):
    # Input: path to transition matrix npy file; output from anneal function
    np.save(t_matrix_path, t_matrix)
