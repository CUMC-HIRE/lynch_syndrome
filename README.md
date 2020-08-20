# lynch_syndrome
Gene-Specific Variation in Colorectal Cancer Surveillance Strategies for Lynch Syndrome 

## Input (Data folder)
* model_inputs.xlsx: contains several sheets with model inputs
    * MLH1, MSH2, MSH6, and PMS2: cumulative risk of colon cancer by age for each gene
    * Sporadic and Survival are not used ?????
    * CRC Survival: probability of cancer mortality by stage; used in `get_cancer_death_prob()` of probability_functions.py
    * Stage Dists: probabilities of each cancer stage at diagnosis, by screening interval; used to set normal/adenoma -> cancer probabilities in `create_t_matrix()` of lynch_simulator.py

## Code
### lynch_presets.py
* `genes`: list of genes to be analyzed (MLH1, MSH2, MSH6, and PMS2)
* `csy_protocols`: colonoscopy protocols
    * `nono`: natural history (no colonoscopy screening)
    * `current`: current screening recommendation (annual screening starting at age 25)
    * `new`: new screening strategies (varying start age in 5-year increments between 25 and 40 years, and surveillance intervals of between 1 and 5 years)
* `ALL_STATES`: dictionary containing all states of the model
    * `mutation`: everyone starts here (with one of the genetic mutations)
    * `nono`: healthy state for running the natural history model
    * `current`: healthy state for running the current recommended screening schedule
    * `new`: healthy state for running the new screening schedules
* `CONNECTIVITY`: dictionary indicating possible transitions between states
    * No transitions between cancer stages
    * The adenoma state can be skipped because it's not always detected before cancer development
* `risk_ratios`: ?????????????????? risk of what? based on screening inverval?
* `run_type`: class containing information about a run
    * Input: screening interval, gene, gender, start age (default 25)
    * risk_ratio: based on screening interval
    * interval: screening interval (1-5 years)
    * gene
    * gender
    * start_age: defaults to 25
    * guidelines: one of the csy_protocols described above (`nono`, `current`, or `new`)
* `risk_adn_male_data`: 
### lynch_simulatory.py
`create_t_matrix()` function:
* Input: instance of run_type class (`run_spec`), time (defaults to ps.time), start age (defaults to ps.START_AGE)
* `this_gender`: contains parameters specific to gender; instance of gender_obj class of gender.py
* `names`: flipped version of the ALL_STATES dictionary (state names are dictionary keys)
* Set natural history risk of colon cancer by age using `cumul_prob_to_annual()` of probability_functions.py (nodes_nono, risk_probs_nono, risk_rates_nono)
* If not a natural history run: 
    * Set risk of colon cancer by age as above, but using a risk ratio dependent on screening interval (nodes, risk_probs, risk_rates)
    * Set 

## Output