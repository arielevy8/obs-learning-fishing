import numpy as np
import pandas as pd
import os
import itertools
from scipy.optimize import approx_fprime

def load_data(filename):
    """Load and preprocess experimental data"""
    data = pd.read_csv(filename)
    # Filter only experimental phase
    exp_data = data[data['phase'] == 'experimental'].reset_index(drop=True)
    return exp_data

def softmax(values, beta):
    # Subtract maximum value for numerical stability
    values = np.array(values)
    values_scaled = beta * (values - np.max(values))
    exp_values = np.exp(values_scaled)
    return exp_values / np.sum(exp_values)

def get_available_boats(row):
    """Extract available boats from the available_boats string"""
    # Remove brackets and split by commas
    boats_str = row['available_boats'].strip('[]')
    return [int(x) for x in boats_str.split(',')]

def update_imitation_values(action_values_im, available_boats, optimal_choice, alpha):
    for boat in available_boats:
        if boat == optimal_choice:
            action_values_im[boat] += alpha * (1 - action_values_im[boat])
        else:
            action_values_im[boat] += alpha * (-1 - action_values_im[boat])
    return action_values_im

def update_emulation_values(available_boats, token_probs, transition_prob, optimal_choice, lambd):
    new_token_probs = np.zeros(3)
    for i in range(3):  # For each token
        # Calculate prior (Equation 2 in Charpentier et al. 2020)
        prior = lambd * token_probs[i] + (1-lambd) * (sum(token_probs)-token_probs[i])/2    
        prob_per_boat = []
        for j in available_boats:
            prob_per_boat.append(transition_prob[j][i])
        expected_choice = available_boats[np.argmax(prob_per_boat)]
        likelihood = 1 if expected_choice == optimal_choice else 0
        new_token_probs[i] = prior * likelihood
    # normalize
    if sum(new_token_probs) > 0:
        new_token_probs = new_token_probs / sum(new_token_probs)
    else:
        new_token_probs = np.array([1/3, 1/3, 1/3])
    return new_token_probs

def decide_imitation(action_values_im, available_boats, beta_im):
    im_values = [action_values_im[i] for i in available_boats]
    im_probs = softmax(im_values, beta_im)
    return im_probs

def decide_emulation(available_boats, token_probs, transition_prob, beta_em):
    em_values = []
    for boat in available_boats:
        em_values.append(np.dot(token_probs, transition_prob[boat]))
    em_probs = softmax(em_values, beta_em)
    return em_probs

# Add function to organize data by participant
def organize_participant_data(data_dir):
    """
    Load and organize all participant data into a dictionary
    """
    participant_data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            participant_id = filename.split('_')[0]
            data = load_data(os.path.join(data_dir, filename))
            participant_data[participant_id] = data
    return participant_data

def transform_01(x):
    """Transform from (-∞,∞) to (0,1) using sigmoid"""
    return 1 / (1 + np.exp(-x))

def inverse_transform_01(p):
    """Transform from (0,1) to (-∞,∞)"""
    p = np.clip(p, 1e-10, 1-1e-10)
    return np.log(p / (1 - p))

def transform_05_1(x):
    """Transform from (-∞,∞) to (0.5,1) using modified sigmoid"""
    return 0.5 + 0.5 / (1 + np.exp(-x))

def inverse_transform_05_1(p):
    """Transform from (0.5,1) to (-∞,∞)"""
    p = np.clip(p, 0.5 + 1e-10, 1-1e-10)
    return np.log((p - 0.5) / (1 - p))

def transform_0_30(x):
    """Transform from (-∞,∞) to (0,30) using modified exponential"""
    return 30 / (1 + np.exp(-x))

def inverse_transform_0_30(p):
    """Transform from (0,30) to (-∞,∞)"""
    p = np.clip(p, 1e-10, 30-1e-10)
    return np.log(p / (30 - p))

def transform_params(params, param_types):
    """Transform parameters from unconstrained to constrained space"""
    transformed = []
    
    # Handle dictionary input
    if isinstance(params, dict):
        for param_name, ptype in zip(params.keys(), param_types):
            param = float(params[param_name])
            if ptype == '01':  # for alpha and w_em
                transformed.append(float(transform_01(param)))
            elif ptype == '05_1':  # for lambda
                transformed.append(float(transform_05_1(param)))
            elif ptype == '0_30':  # for beta
                transformed.append(float(transform_0_30(param)))
            else:
                transformed.append(float(param))
    # Handle list input
    else:
        for param, ptype in zip(params, param_types):
            param = float(param)
            if ptype == '01':
                transformed.append(float(transform_01(param)))
            elif ptype == '05_1':
                transformed.append(float(transform_05_1(param)))
            elif ptype == '0_30':
                transformed.append(float(transform_0_30(param)))
            else:
                transformed.append(float(param))
    
    return transformed

def inverse_transform_params(params, param_types):
    """Transform parameters from constrained to unconstrained space"""
    transformed = []
    
    # Handle dictionary input
    if isinstance(params, dict):
        for param_name, ptype in zip(params.keys(), param_types):
            param = float(params[param_name])
            if ptype == '01':  # for alpha and w_em
                transformed.append(inverse_transform_01(param))
            elif ptype == '05_1':  # for lambda
                transformed.append(inverse_transform_05_1(param))
            elif ptype == '0_30':  # for beta
                transformed.append(inverse_transform_0_30(param))
            else:
                transformed.append(param)
    # Handle list input
    else:
        for param, ptype in zip(params, param_types):
            param = float(param)
            if ptype == '01':
                transformed.append(inverse_transform_01(param))
            elif ptype == '05_1':
                transformed.append(inverse_transform_05_1(param))
            elif ptype == '0_30':
                transformed.append(inverse_transform_0_30(param))
            else:
                transformed.append(param)
    
    return transformed

