import numpy as np
import pandas as pd
from fit_utils import *
from scipy.optimize import minimize
from scipy.stats import norm
import json

# Model 1 - only emulation
def model_1_objective(params, data, transform = True):
    """
    This is the objective function for model 1. 
    Model 1 is a model of goal emulation, with lambda being estimated (both beta and lambda are fitted).
    :param params: the parameters to fit the model to
    :param data: the data to fit the model to
    :return: negative log likelihood of the data given the model
    """
    if transform:
        transformed = transform_params(params, ['05_1', '0_30'])
        lambd, beta = transformed[0], transformed[1]
    else:
        lambd, beta = params[0], params[1]
    
    token_probs = [1/3, 1/3, 1/3]
    log_likelihood = 0
    transition_prob = json.loads(data.iloc[0]['transition_prob'])

    for _, row in data.iterrows():
        # Reset values if new block
        if row ['day'] in [23, 45, 67, 89, 111]:
            token_probs = [1/3, 1/3, 1/3]
        available_boats = get_available_boats(row)
        if row['task_type'] == 'observe':
            # Update token probabilities based on observed choice
            optimal_choice = int(row['boat_chosen_id'])
            token_probs = update_emulation_values(available_boats, token_probs, transition_prob, optimal_choice, lambd)
        elif row['task_type'] == 'play':
            # Calculate choice probabilities for available boats
            probs = decide_emulation(available_boats, token_probs, transition_prob, beta)
            # Calculate log likelihood of actual choice
            actual_choice = int(row['boat_chosen_id'])
            choice_idx = available_boats.index(actual_choice)
            log_likelihood += np.log(probs[choice_idx] + 1e-10)
    
    return -log_likelihood

# model 2 - only RL imitation
def model_2_objective(params, data, transform = True):
    """
    This is the objective function for model 1. 
    Model 2 is a model of only imitation, alpha being estimated (both beta and alpha are fitted).
    :param params: the parameters to fit the model to
    :param data: the data to fit the model to
    :return: negative log likelihood of the data given the model
    """
    if transform:
        transformed = transform_params(params, ['01', '0_30'])
        alpha, beta = transformed[0], transformed[1]
    else:
        alpha, beta = params[0], params[1]
    
    action_values = [0, 0, 0]
    log_likelihood = 0

    for _, row in data.iterrows():
        # Reset values if new block
        if row ['day'] in [23, 45, 67, 89, 111]:
            action_values = [0, 0, 0]
        available_boats = get_available_boats(row)
        if row['task_type'] == 'observe':
            # Update values based on observed choice
            optimal_choice = int(row['boat_chosen_id']) 
            action_values = update_imitation_values(action_values, available_boats, optimal_choice, alpha)
        elif row['task_type'] == 'play':
            # Calculate choice probabilities
            probs = decide_imitation(action_values, available_boats, beta)
            actual_choice = int(row['boat_chosen_id'])
            choice_idx = available_boats.index(actual_choice)
            log_likelihood += np.log(probs[choice_idx] + 1e-10)
    
    return -log_likelihood

# model 3 - only copy last action
def model_3_objective(params, data, transform = True):
    """
    This is the objective function for model 3. 
    Model 3 is a very simple imitation - copy last choice. If the last choice is not available, random choice.
    The only parameter to be fitted is beta.
    """
    if transform:
        beta = transform_params(params, ['0_30'])[0]
    else:
        beta = params[0]

    log_likelihood = 0
    last_observed = None  # Track the last observed choice
    
    # Iterate through trials
    for idx, row in data.iterrows():
        # Reset values if new block
        if row ['day'] in [23, 45, 67, 89, 111]:
            last_observed = None
        available_boats = get_available_boats(row)
        if row['task_type'] == 'observe':
            last_observed = int(row['boat_chosen_id'])
        elif row['task_type'] == 'play':
            available_boats = get_available_boats(row)
            probs = np.zeros(len(available_boats))
            if last_observed is None:
                # If no observation yet, assume random choice
                probs = np.ones(len(available_boats)) / len(available_boats)
            else:
                # If last_observed is available, probability is 1, else 0
                if last_observed in available_boats:
                    probs[available_boats.index(last_observed)] = 1.0
                else:
                    probs = np.ones(len(available_boats)) / len(available_boats)
            # Apply softmax to probabilities
            probs = softmax(probs, beta)
            # Calculate log likelihood of actual choice
            actual_choice = int(row['boat_chosen_id'])
            choice_idx = available_boats.index(actual_choice)
            log_likelihood += np.log(probs[choice_idx] + 1e-10)
    
    return -log_likelihood

# model 4 - emulation and RL imitation, only one weight
def model_4_objective(params, data, constants = None, transform = True):
    """
    This is the objective function for model 4. 
    Model 4 is a model of mixture model of both imitation with estimated alpha
    (model 2) and goal emulation with estimated lambda (model 1).
    Fitted parameters are beta, w_em, alpha, lambda.
    """
    if constants:
        alpha, beta, lambd = constants[0], constants[1], constants[2]
        if transform:
            w_em = transform_params(params, ['01'])[0]
        else:
            w_em = params[0]
    else:
        if transform:
            transformed = transform_params(params, ['01', '0_30', '05_1', '01'])
            alpha, beta, lambd = transformed[0], transformed[1], transformed[2]
            w_em = transformed[3]
        else:
            alpha, beta, lambd = params[0], params[1], params[2]
            w_em = params[3]
    
    # Calculate complementary weights
    w_im = 1 - w_em
    
    # Initialize values
    action_values_im = [0, 0, 0]  # Imitation values
    token_probs = [1/3, 1/3, 1/3]  # Emulation token probabilities
    log_likelihood = 0
    
    # Get transition probabilities from first row
    transition_prob = json.loads(data.iloc[0]['transition_prob'])
    
    for idx, row in data.iterrows():
        #Reset values if new block
        if row ['day'] in [23, 45, 67, 89, 111]:
            action_values_im = [0, 0, 0]
            token_probs = [1/3, 1/3, 1/3]
        available_boats = get_available_boats(row)
        
        if row['task_type'] == 'observe':
            # Update both models
            optimal_choice = int(row['boat_chosen_id'])
            
            # Update imitation values
            action_values_im = update_imitation_values(action_values_im, available_boats, optimal_choice, alpha)
            
            # Update emulation token probabilities
            token_probs = update_emulation_values(available_boats, token_probs, transition_prob, optimal_choice, lambd)
                
        elif row['task_type'] == 'play':
            # Get values from both models
            # Imitation values
            im_probs = decide_imitation(action_values_im, available_boats, beta)
            
            # Emulation values
            em_probs = decide_emulation(available_boats, token_probs, transition_prob, beta)
            # Combine probabilities using mixture weights
            final_probs = []
            for i in range(len(available_boats)):
                prob = w_im * im_probs[i] + w_em * em_probs[i]
                final_probs.append(prob)
            
            # Calculate log likelihood of actual choice
            actual_choice = int(row['boat_chosen_id'])
            choice_idx = available_boats.index(actual_choice)
            log_likelihood += np.log(final_probs[choice_idx] + 1e-10)
    
    return -log_likelihood

# model 5 - emulation and RL imitation, both weights
def model_5_objective(params, data, constants = None, transform = True):
    """
    This is the objective function for model 5. 
    Model 5 is a model of mixture model of both imitation with estimated alpha
    (model 2) and goal emulation with estimated lambda (model 1).
    Fitted parameters are beta, w_em, alpha, lambda.
    """
    if constants:
        alpha, beta, lambd = constants[0], constants[1], constants[2]
        if transform:
            transformed = transform_params(params, ['01', '01'])
            w_em_low, w_em_high = transformed[0], transformed[1]
        else:
            w_em_low, w_em_high = params[0], params[1]
    else:
        if transform:
            transformed = transform_params(params, ['01', '0_30', '05_1', '01', '01'])
            alpha, beta, lambd = transformed[0], transformed[1], transformed[2]
            w_em_low, w_em_high = transformed[3], transformed[4]
        else:
            alpha, beta, lambd = params[0], params[1], params[2]
            w_em_low, w_em_high = params[3], params[4]
    
    # Calculate complementary weights
    w_im_low = 1 - w_em_low
    w_im_high = 1 - w_em_high
    
    # Initialize values
    action_values_im = [0, 0, 0]  # Imitation values
    token_probs = [1/3, 1/3, 1/3]  # Emulation token probabilities
    log_likelihood = 0
    
    # Get transition probabilities from first row
    transition_prob = json.loads(data.iloc[0]['transition_prob'])
    
    for idx, row in data.iterrows():
        #Reset values if new block
        if row ['day'] in [23, 45, 67, 89, 111]:
            action_values_im = [0, 0, 0]
            token_probs = [1/3, 1/3, 1/3]
        available_boats = get_available_boats(row)
        
        # Select weights based on stakes condition
        if row['stakes'] == 'low':
            w_im, w_em = w_im_low, w_em_low
        elif row['stakes'] == 'high':  # high stakes
            w_im, w_em = w_im_high, w_em_high
        
        if row['task_type'] == 'observe':
            # Update both models
            optimal_choice = int(row['boat_chosen_id'])
            
            # Update imitation values
            action_values_im = update_imitation_values(action_values_im, available_boats, optimal_choice, alpha)
            
            # Update emulation token probabilities
            token_probs = update_emulation_values(available_boats, token_probs, transition_prob, optimal_choice, lambd)
                
        elif row['task_type'] == 'play':
            # Get values from both models
            # Imitation values
            im_probs = decide_imitation(action_values_im, available_boats, beta)
            
            # Emulation values
            em_probs = decide_emulation(available_boats, token_probs, transition_prob, beta)
            # Combine probabilities using mixture weights
            final_probs = []
            for i in range(len(available_boats)):
                prob = w_im * im_probs[i] + w_em * em_probs[i]
                final_probs.append(prob)
            
            # Calculate log likelihood of actual choice
            actual_choice = int(row['boat_chosen_id'])
            choice_idx = available_boats.index(actual_choice)
            log_likelihood += np.log(final_probs[choice_idx] + 1e-10)
    
    return -log_likelihood

# model 6 - Emulation and copy last action, only one weight
def model_6_objective(params, data, constants = None, transform = True):
    """
    This is the objective function for model 6. 
    Model 6 is a mixture model of copy last choice (model 3) and goal emulation (model 1).
    Fitted parameters are beta, w_em, lambda.
    """
    if constants:
        beta, lambd = constants[0], constants[1]
        if transform:
            w_em = transform_params(params, ['01'])[0]
        else:
            w_em = params[0]
    else:
        transformed = transform_params(params, ['0_30', '05_1', '01'])
        beta, lambd, w_em = transformed[0], transformed[1], transformed[2]

    # Calculate complementary weights
    w_copy = 1 - w_em
    
    # Initialize values for emulation
    token_probs = [1/3, 1/3, 1/3]
    log_likelihood = 0
    last_observed = None  # For copy-last model
    
    # Get transition probabilities from first row
    transition_prob = json.loads(data.iloc[0]['transition_prob'])
    
    # Iterate through trials
    for idx, row in data.iterrows():
        # Reset values if new block
        if row ['day'] in [23, 45, 67, 89, 111]:
            last_observed = None
            token_probs = [1/3, 1/3, 1/3]
        available_boats = get_available_boats(row)
        
        if row['task_type'] == 'observe':
            # Update copy-last memory
            optimal_choice = int(row['boat_chosen_id'])
            last_observed = optimal_choice  # Update copy-last memory
            
            # Update token probabilities for emulation
            token_probs = update_emulation_values(available_boats, token_probs, transition_prob, optimal_choice, lambd)
                
        elif row['task_type'] == 'play':
            # Get probabilities from both models
            
            # Copy-last probabilities
            copy_probs = np.zeros(len(available_boats))
            if last_observed is None:
                copy_probs = np.ones(len(available_boats)) / len(available_boats)
            else:
                if last_observed in available_boats:
                    copy_probs[available_boats.index(last_observed)] = 1.0
                else:
                    copy_probs = np.ones(len(available_boats)) / len(available_boats)
            copy_probs = softmax(copy_probs, beta)
            
            # Emulation probabilities
            em_probs = decide_emulation(available_boats, token_probs, transition_prob, beta)
            
            # Combine probabilities using mixture weights
            final_probs = w_copy * copy_probs + w_em * em_probs
            
            # Calculate log likelihood of actual choice
            actual_choice = int(row['boat_chosen_id'])
            choice_idx = available_boats.index(actual_choice)
            log_likelihood += np.log(final_probs[choice_idx] + 1e-10)
    
    return -log_likelihood

# model 7 - Emulation and copy last  action, both weights
def model_7_objective(params, data, constants = None, transform = True):
    """
    This is the objective function for model 7. 
    Model 7 is a mixture model of copy last choice (model 5) and goal emulation (model 4).
    Fitted parameters are beta, w_em_low, w_em_high, lambda.
    """
    if constants:
        beta, lambd = constants[0], constants[1]
        if transform:
            w_em_low, w_em_high = transform_params(params, ['01', '01'])[0], transform_params(params, ['01', '01'])[1]
        else:
            w_em_low, w_em_high = params[0], params[1]
    else:
        transformed = transform_params(params, ['0_30', '05_1', '01', '01'])
        beta, lambd, w_em_low, w_em_high = transformed[0], transformed[1], transformed[2], transformed[3]
    
    # Calculate complementary weights
    w_copy_low = 1 - w_em_low
    w_copy_high = 1 - w_em_high
    
    # Initialize values for emulation
    token_probs = [1/3, 1/3, 1/3]
    log_likelihood = 0
    last_observed = None  # For copy-last model
    
    # Get transition probabilities from first row
    transition_prob = json.loads(data.iloc[0]['transition_prob'])
    
    # Iterate through trials
    for idx, row in data.iterrows():
        # Reset values if new block
        if row ['day'] in [23, 45, 67, 89, 111]:
            last_observed = None
            token_probs = [1/3, 1/3, 1/3]
        available_boats = get_available_boats(row)
        
        # Select weights based on stakes condition
        if row['stakes'] == 'low':
            w_em, w_copy = w_em_low, w_copy_low
        else:  # high stakes
            w_em, w_copy = w_em_high, w_copy_high
        
        if row['task_type'] == 'observe':
            # Update copy-last memory
            optimal_choice = int(row['boat_chosen_id'])
            last_observed = optimal_choice  # Update copy-last memory
            
            # Update token probabilities for emulation
            token_probs = update_emulation_values(available_boats, token_probs, transition_prob, optimal_choice, lambd)
                
        elif row['task_type'] == 'play':
            # Get probabilities from both models
            
            # Copy-last probabilities
            copy_probs = np.zeros(len(available_boats))
            if last_observed is None:
                copy_probs = np.ones(len(available_boats)) / len(available_boats)
            else:
                if last_observed in available_boats:
                    copy_probs[available_boats.index(last_observed)] = 1.0
                else:
                    copy_probs = np.ones(len(available_boats)) / len(available_boats)
            copy_probs = softmax(copy_probs, beta)
            
            # Emulation probabilities
            em_probs = decide_emulation(available_boats, token_probs, transition_prob, beta)
            
            # Combine probabilities using mixture weights
            final_probs = w_copy * copy_probs + w_em * em_probs
            
            # Calculate log likelihood of actual choice
            actual_choice = int(row['boat_chosen_id'])
            choice_idx = available_boats.index(actual_choice)
            log_likelihood += np.log(final_probs[choice_idx] + 1e-10)
    
    return -log_likelihood

def get_model_specs(model_num):
    """
    Get the model specifications for a given model number
    """
    # Define model-specific parameters and objective function
    model_specs = {
        1: {'params': ['lambd', 'beta'], 
            'types': ['05_1', '0_30'],
            'objective': model_1_objective},
        2: {'params': ['alpha', 'beta'], 
            'types': ['01', '0_30'],
            'objective': model_2_objective},
        3: {'params': ['beta'], 
            'types': ['0_30'],
            'objective': model_3_objective},
        4: {'params': ['alpha', 'beta', 'lambd', 'w_em'], 
            'types': ['01', '0_30', '05_1', '01'],
            'objective': model_4_objective},
        5: {'params': ['alpha', 'beta', 'lambd', 'w_em_low', 'w_em_high'],
            'types': ['01', '0_30', '05_1', '01', '01'],
            'objective': model_5_objective},
        6: {'params': ['beta', 'lambd', 'w_em'],
            'types': ['0_30', '05_1', '01'],
            'objective': model_6_objective},
        7: {'params': ['beta', 'lambd', 'w_em_low', 'w_em_high'],
            'types': ['0_30', '05_1', '01', '01'],
            'objective': model_7_objective},
    }
    if model_num not in model_specs:
        raise ValueError(f"Model {model_num} not implemented")
    return model_specs[model_num]




