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

def generate_rand_t_matrix(run_spec): 
    # Input: natural history, current screening guidelines, or new screening guidelines
        # ('nono', 'current', 'new')
    # Output: somewhat random t_matrix
    
    # ASSUMPTIONS:
        # Probabilities of remaining in SD and PD states increase over time
        # Probability of dying from cancer decreases over time
        # Probability of progressing decreases over time
    
    states = ps.ALL_STATES
    state_names = dm.flip(ps.ALL_STATES)
    time = ps.time
    this_gender = g.gender_obj(run_spec.gender)
    
    for t in time:
        # Initialize t_matrix for time, t
        t_layer = np.zeros((len(states), len(states)))
        # Calculate age
        age = t + ps.START_AGE
        
        ### Set constant probabilities (not calibrated)
        # All-cause mortality
        all_cause_states = [
            state_names["current"], state_names["new"], state_names["nono"],
            state_names["init adenoma"], state_names["adenoma"],state_names["init adv adenoma"],
            state_names["init adv adenoma"]]
        all_cause_dx_states = [
            state_names['init dx stage I'],
            state_names['init dx stage II'], state_names['init dx stage III'],
            state_names['init dx stage IV'], state_names['dx stage I'],
            state_names['dx stage II'], state_names['dx stage III'],
            state_names['dx stage IV']]
        for i in all_cause_states:
            t_layer[i, state_names["all cause"]] = this_gender.lynch_ac_mortality[age]
        for i in all_cause_dx_states:
            t_layer[i, state_names['all cause dx']] = this_gender.lynch_ac_mortality[age]
        # Colonoscopy mortality
        csy_death_states = [
            state_names['current'], state_names['new'], state_names['init adenoma'],
            state_names['adenoma'], state_names["init adv adenoma"],
            state_names["init adv adenoma"]]
        for i in csy_death_states:
            if run_spec.guidelines != 'nono':
                t_layer[i, state_names['csy death']] = ps.p_csy_death
            # else: no colonoscopy during natural history -> prob remains 0
        # Cancer mortality
        # Colectomy death is set as stage 1 death
        if age < 60:
            colectomy_death = ps.colectomy_death_risk[0]
        elif age < 70:
            colectomy_death = ps.colectomy_death_risk[1]
        else:
            colectomy_death = ps.colectomy_death_risk[2]
        t_layer[state_names['init dx stage I'], state_names['stage I death']] = colectomy_death
        stage_2_death, stage_3_death, stage_4_death = pf.get_cancer_death_probs(age, this_gender)
        t_layer[state_names['dx stage II'], state_names['stage II death']] = stage_2_death
        t_layer[state_names['init dx stage II'], state_names['stage II death']] = stage_2_death
        t_layer[state_names['dx stage III'], state_names['stage III death']] = stage_3_death
        t_layer[state_names['init dx stage III'], state_names['stage III death']] = stage_3_death
        t_layer[state_names['dx stage IV'], state_names['stage IV death']] = stage_4_death
        t_layer[state_names['init dx stage IV'], state_names['stage IV death']] = stage_4_death

        ### Randomize remaining probabilities
        # Normal -> adenoma/advanced adenoma (does not apply to natural history)
        t_layer[state_names[run_spec.guidelines], state_names['init adenoma']] = risk_adn
        # Normal -> cancer
        t_layer[state_names[run_spec.guidelines],
                state_names['init dx stage I']] = dx_risk_prob * ps.staging.loc[run_spec.interval, 'stage_1']
        t_layer[state_names[run_spec.guidelines], 
                state_names['init dx stage II']] = dx_risk_prob * ps.staging.loc[run_spec.interval, 'stage_2'] 
        t_layer[state_names[run_spec.guidelines], 
                state_names['init dx stage III']] = dx_risk_prob * ps.staging.loc[run_spec.interval, 'stage_3'] 
        t_layer[state_names[run_spec.guidelines], 
                state_names['init dx stage IV']] = dx_risk_prob * ps.staging.loc[run_spec.interval, 'stage_4'] 

        # Adenoma/advanced adenoma -> advanced adenoma or cancer
        t_layer[state_names['adenoma'], state_names['init adv adenoma']] = 1 - this_gender.lynch_ac_mortality[age] - adn_dx_risk - csy_death_risk
        t_layer[state_names['adenoma'], 
                state_names['init dx stage I']] = adn_dx_risk * ps.staging.loc[run_spec.interval, 'stage_1']
        t_layer[state_names['adenoma'], 
                state_names['init dx stage II']] = adn_dx_risk * ps.staging.loc[run_spec.interval, 'stage_2']
        t_layer[state_names['adenoma'], 
                state_names['init dx stage III']] = adn_dx_risk * ps.staging.loc[run_spec.interval, 'stage_3']
        t_layer[state_names['adenoma'], 
                state_names['init dx stage IV']] = adn_dx_risk * ps.staging.loc[run_spec.interval, 'stage_4']
        
        t_layer[state_names['init adenoma'], 
                state_names['init dx stage I']] = adn_dx_risk * ps.staging.loc[run_spec.interval, 'stage_1']
        t_layer[state_names['init adenoma'], 
                state_names['init dx stage II']] = adn_dx_risk * ps.staging.loc[run_spec.interval, 'stage_2']
        t_layer[state_names['init adenoma'], 
                state_names['init dx stage III']] = adn_dx_risk * ps.staging.loc[run_spec.interval, 'stage_3']
        t_layer[state_names['init adenoma'], 
                state_names['init dx stage IV']] = adn_dx_risk * ps.staging.loc[run_spec.interval, 'stage_4']
            
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

        if t == 0:
            t_matrix = t_layer
        else:
            t_matrix = np.vstack((t_matrix, t_layer))

    # Create 3D matrix
    t_matrix = np.reshape(t_matrix, (len(time), len(states), len(states)))
    # print(t_matrix[5])

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
