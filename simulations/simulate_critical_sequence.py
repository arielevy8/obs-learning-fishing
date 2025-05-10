import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Environment import Environment
from Agents import OptimalAgent, ModelFreeObserver, ModelBasedObserver
from datetime import datetime


def simulate_critical_sequence(volatility, certainty, num_repetitions, alpha, lambd, beta, play_mode_all):
    """
    Simulate a critical sequence of trials, that critically distinguishes between model-based and model-free agents
    :param volatility: float, the rate of change in token values
    :param certainty: array with size num_tokens, representing probability distribution for tokens in slot machines
    :param num_repetitions: int, number of repetitions
    :param alpha: float, learning rate
    :param lambd: float, discount factor
    :param beta: float, inverse temperature parameter
    :return: normalized_reward_diff: float, difference in cumulative reward between model-based and model-free agents
    """
    env = Environment(volatility, certainty, num_repetitions, play_mode_all)
    opt = OptimalAgent(env)
    model_based = ModelBasedObserver(env, lambd, beta)
    model_free = ModelFreeObserver(env, alpha, beta)
    
    # Trial 1 
    env.cur_repetition = 0
    env.get_valuable_token()
    available_machines = []
    print(f'valuable token: {env.mem_valuable_token[0]}')
    for i in range(env.NUM_MACHINES):
        if i != env.mem_valuable_token[0]: # only choose the machine that does not have the valuable token as the most probable token
            available_machines.append(i)
    env.mem_available_machines_id.append(available_machines)
    print(f'trial 1, available machines: {available_machines}')
    env.cur_pull_seed = np.random.randint(0, 1000) # set seed for pull lever
    opt.do_repetition()
    print(f'trial 1, optimal agent choice: {env.mem_choice_index_optimal[0]}')
    model_free.observe_behavior()
    print(f'trial 1, model-free action values: {model_free.action_values}')
    model_based.observe_behavior()
    print(f'trial 1, model-based token probabilities: {model_based.token_probs}')
    # Trial 2
    env.cur_repetition +=1
    env.get_valuable_token()
    available_machines = []
    for i in range(env.NUM_MACHINES):
        if i == env.mem_valuable_token[1]:
            available_machines.append(i)
        elif i == env.mem_valuable_token[1]+1 or i == env.mem_valuable_token[1]-2: # Put also the machine with the least probability of having the valuable token
            available_machines.append(i)
    print(f'trial 2, available machines: {available_machines}')
    env.mem_available_machines_id.append(available_machines)
    env.cur_pull_seed = np.random.randint(0, 1000) # set seed for pull lever
    opt.do_repetition()
    print(f'trial 2, optimal agent choice: {env.mem_choice_index_optimal[1]}')
    model_free.observe_behavior()
    print(f'trial 2, model-free action values: {model_free.action_values}')
    model_based.observe_behavior()
    print(f'trial 2, model-based token probabilities: {model_based.token_probs}')
    #trial 3
    env.cur_repetition += 1
    env.get_valuable_token()
    available_machines = []
    for i in range(env.NUM_MACHINES):
        if i != env.mem_valuable_token[2]: # only choose the machine that does not have the valuable token as the most probable token
            available_machines.append(i)
    print(f'trial 3, available machines: {available_machines}')
    env.mem_available_machines_id.append(available_machines)
    env.cur_pull_seed = np.random.randint(0, 1000) # set seed for pull lever
    opt.do_repetition()
    print(f'trial 3, optimal agent choice: {env.mem_choice_index_optimal[2]}')
    model_free.observe_behavior()
    print(f'trial 3, model-free action values: {model_free.action_values}')
    model_based.observe_behavior()
    print(f'trial 3, model-based token probabilities: {model_based.token_probs}')
    #Trial 4: play
    env.cur_repetition += 1
    env.get_valuable_token()
    available_machines = []
    for i in range(env.NUM_MACHINES):
        if i == env.mem_valuable_token[3]:
            available_machines.append(i)
        # Put also the machine that has benn chosen by the optimal agent
        elif i == env.mem_choice_index_optimal[2]:
            available_machines.append(i)
    print(f'trial 4, available machines: {available_machines}')
    env.mem_available_machines_id.append(available_machines)
    env.cur_pull_seed = np.random.randint(0, 1000) # set seed for pull lever
    opt.do_repetition()
    print(f'trial 4, optimal agent choice: {env.mem_choice_index_optimal[3]}')
    model_free.do_repetition()
    print(f'trial 4, model-free choice: {env.mem_choice_index_model_free[3]}')
    model_based.do_repetition()
    print(f'trial 4, model-based choice: {env.mem_choice_index_model_based[3]}')
    df = env.organize_data()
    print(df)
    return (env.mem_choice_index_model_based[3] == env.mem_choice_index_optimal[3],
            env.mem_choice_index_model_free[3] == env.mem_choice_index_optimal[3],
            env.mem_choice_index_model_based[3] == env.mem_choice_index_model_free[3])
    
if __name__ == '__main__':
    volatility = 1/5.5
    certainty = [0.8, 0.2, 0.0]
    alpha_range = [0.1,0.3,0.5,0.7,0.9,0.99]
    lambd_range = [0.1,0.2,0.3,0.4,0.49,0.5,0.51,0.6,0.7,0.8,0.9,0.99]
    beta_range = [100] # Only interested in the machine with the higest probability of being chosen by the agent
    num_repetitions = 4
    play_mode_all = False
    mat_model_based_correct = np.zeros((len(alpha_range), len(lambd_range)))
    mat_model_free_correct = np.zeros((len(alpha_range), len(lambd_range)))
    mat_model_based_free_equal= np.zeros((len(alpha_range), len(lambd_range)))
    for alpha_idx, alpha in enumerate(alpha_range):
        for lambd_idx, lambd in enumerate(lambd_range):
            for beta in beta_range:
                result = simulate_critical_sequence(volatility, certainty, num_repetitions, alpha, lambd, beta, play_mode_all)
                mat_model_based_correct[alpha_idx, lambd_idx] = result[0]
                mat_model_free_correct[alpha_idx, lambd_idx] = result[1]
                mat_model_based_free_equal[alpha_idx, lambd_idx] = result[2]
    print(mat_model_based_correct)
    print(mat_model_free_correct)
    print(mat_model_based_free_equal)
    # Results: for all alpha values, and for lambda values lambda > 0.5, only model-based agent is correct
