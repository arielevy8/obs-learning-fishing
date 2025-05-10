import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Environment import Environment
from Agents import *
from datetime import datetime

# define function to create sequence of 'observe' and 'play' repetitions, so that the play repetition appear every [2,3,4] repetitions and the observe repetition appear every [1,2,3] repetitions
def create_repetition_sequence(num_repetitions):
    sequence = []
    possible_lengths = [2,3,4] # 1-3 observe trials between play trials
    play_trials = [2]
    for i in range (num_repetitions//3):
        play_trials.append(play_trials[-1] + np.random.choice(possible_lengths))
    for i in range (num_repetitions):
        if i in play_trials:
            sequence.append('play')
        else:
            sequence.append('observe')
    # The last repetition should always be a play repetition
    sequence[-1] = 'play'
    return sequence


def simulate_experiment(volatility, certainty, num_repetitions, lambd, beta, play_mode_all):
    """
    Simulate an experiment with a given volatility, certainty, number of repetitions, learning rate, and agent type
    :param volatility: float, the rate of change in token values
    :param certainty: array with size num_tokens, representing probability distribution for tokens in slot machines
    :param num_repetitions: int, number of repetitions
    :param alpha: float, learning rate
    :param lambd: float, discount factor
    :param beta: float, inverse temperature parameter
    :return: normalized_reward_diff: float, difference in cumulative reward between model-based and model-free agents
        """
    repetition_sequence = create_repetition_sequence(num_repetitions)
    env = Environment(volatility, certainty, num_repetitions, play_mode_all)
    opt = OptimalAgent(env)
    model_based = ModelBasedObserver(env, lambd, beta)
    copy_last = CopyLastObserver(env, beta)
    for i in range(env.num_repetitions):
        if repetition_sequence[i] == 'observe':
            env.new_trial('observe')
            opt.do_repetition()
            copy_last.observe_behavior()
            model_based.observe_behavior()
        else:
            env.new_trial('play')
            model_based.do_repetition()
            model_based.observe_behavior()
            copy_last.do_repetition()
            copy_last.observe_behavior()
    df = env.organize_data()
    df['token_probabilities'] = model_based.mem_token_probs
    df['expected_valuable_token'] = model_based.mem_expected_valuable_token
    # Compute accuracy of model based and model free agents - wheter the boat they choose maximizes the expected valuable token
    # Calculate accuracy compared to optimal agent choices
    df['model_based_accuracy'] = np.nan
    df['copy_last_accuracy'] = np.nan
    
    # Only calculate accuracy for play trials
    play_trials = df['trial_type'] == 'play'
    
    for idx in df[play_trials].index:
        available_machines = df.loc[idx, 'available_machines_id']
        valuable_token = df.loc[idx, 'valuable_token']
        
        # Calculate what optimal agent would choose
        locs_in_machine = []
        for i in available_machines:
            cur_machine = env.MACHINES[i]
            locs_in_machine.append(np.where(cur_machine==valuable_token)[0][0])
        optimal_choice = available_machines[np.argmin(locs_in_machine)]
        
        # Compare model-based choice to optimal
        if not np.isnan(df.loc[idx, 'choice_index_model_based']):
            df.loc[idx, 'model_based_accuracy'] = int(df.loc[idx, 'choice_index_model_based'] == optimal_choice)
            
        # Compare model-free choice to optimal    
        if not np.isnan(df.loc[idx, 'choice_index_copy_last']):
            df.loc[idx, 'copy_last_accuracy'] = int(df.loc[idx, 'choice_index_copy_last'] == optimal_choice)
    
    model_based_accuracy = np.nanmean(df['model_based_accuracy'])
    copy_last_accuracy = np.nanmean(df['copy_last_accuracy'])

    
    model_based_reward = np.sum(df['reward_model_based'])
    copy_last_reward = np.sum(df['reward_copy_last'])
    reward_diff = model_based_reward - copy_last_reward
    normalized_reward_diff = reward_diff/(model_based_reward + copy_last_reward)
    # calculate proportion of identical choices between model-based and model-free agents
    # first index trials in which choice index is not nan
    choice_index_not_nan_model_based = np.array(df['choice_index_model_based'][~np.isnan(df['choice_index_model_based'])])
    choice_index_not_nan_copy_last = np.array(df['choice_index_copy_last'][~np.isnan(df['choice_index_copy_last'])])
    proportion_identical_choices = np.mean(choice_index_not_nan_copy_last == choice_index_not_nan_model_based)
    return normalized_reward_diff, proportion_identical_choices, copy_last_accuracy, model_based_accuracy, copy_last_reward, model_based_reward

num_trials = 150
play_mode_all = False
beta = 100 # Inverse temprature is set to high value, so the agent is deterministic and we simulate the difference between the two 'pure' agents
lambd = [0.6,0.8,0.9,0.99]
volatility = [1/2.5,1/3.5,1/4.5,1/5.5,1/6.5,1/8.5,1/10.5,1/16.5]
uncertainty = [[0.9,0.1,0],[0.8,0.2,0],[0.7,0.3,0],[0.6,0.4,0]]
# Create a list to store all results
results_list = []

# Run the experiment for all combinations of parameters. For each combination of
# alpha, lambd and beta, create a heatmap of the difference between model-based and model-free agents
# in cumulative reward, so that the x axis is volatility, the y axis is certainty, and the color
# represents the difference in cumulative reward between the two agents
for l in lambd:
    reward_diff_mat = np.zeros((len(volatility), len(uncertainty)))
    proportion_identical_mat = np.zeros((len(volatility), len(uncertainty)))
    copy_last_accuracy_mat = np.zeros((len(volatility), len(uncertainty)))
    model_based_accuracy_mat = np.zeros((len(volatility), len(uncertainty)))
    for v in volatility:
        for c in uncertainty:
            reward_diff_list = []
            proportion_identical_list = []
            copy_last_accuracy_list = []
            model_based_accuracy_list = []
            for i in range(250):
                reward_diff, proportion_identical, copy_last_accuracy, model_based_accuracy, copy_last_reward, model_based_reward = simulate_experiment(v, c, num_trials, l, beta, play_mode_all)
                # Add result to list instead of using concat
                results_list.append({
                    'lambd': l,
                    'beta': beta,
                    'play_mode_all': play_mode_all,
                    'volatility': v,
                    'Uncertainty': c,
                    'normalized_reward_diff': reward_diff,
                    'proportion_identical': proportion_identical,
                    'copy_last_accuracy': copy_last_accuracy,
                    'model_based_accuracy': model_based_accuracy,
                    'copy_last_reward': copy_last_reward,
                    'model_based_reward': model_based_reward
                })
                reward_diff_list.append(reward_diff)
                proportion_identical_list.append(proportion_identical)
                copy_last_accuracy_list.append(copy_last_accuracy)
                model_based_accuracy_list.append(model_based_accuracy)
            print(f'Volatility: {v}, Uncertainty: {c}, Mean Reward Diff: {np.mean(reward_diff_list)}, Mean Proportion Identical: {np.nanmean(proportion_identical_list)}')
            reward_diff_mat[volatility.index(v), uncertainty.index(c)] = np.mean(reward_diff_list)
            proportion_identical_mat[volatility.index(v), uncertainty.index(c)] = np.nanmean(proportion_identical_list)
            copy_last_accuracy_mat[volatility.index(v), uncertainty.index(c)] = np.nanmean(copy_last_accuracy_list)
            model_based_accuracy_mat[volatility.index(v), uncertainty.index(c)] = np.nanmean(model_based_accuracy_list)

    # create and save one heatmap of the difference in cumulative reward between model-based and model-free agents
    # as function of volatility and certainty, with the title showing the values of alpha, lambd, and beta
    # set plot size
    # plt.figure(figsize=(7,5.5))
    plt.imshow(reward_diff_mat, cmap='coolwarm', interpolation='nearest')
    plt.xlabel('Unertainty',fontsize = 12)
    plt.ylabel('Volatility',fontsize = 12)
    # use uncertainty values as ticks labels
    plt.xticks(range(len(uncertainty)), [str(c) for c in uncertainty],fontsize = 7.5)
    plt.yticks(range(len(volatility)), [str(round(v,3)) for v in volatility])
    # add colorbar label
    plt.colorbar().set_label('Normalized Reward Difference (Model-Based - Copy Last)')
    plt.title(f'Lambda={l}, Beta={beta}, Play Mode All={play_mode_all}')
    # Add the numbers on the heatmap
    for i in range(len(volatility)):
        for j in range(len(uncertainty)):
            plt.text(j, i, f'{reward_diff_mat[i, j]:.2f}', ha='center', va='center', color='black',fontsize=12)
    # save heatmap to output folder
    plt.savefig(f'output/reward_diff_emu_vs_copy_lambda_{l}.png')
    plt.close()
 # After all loops complete, create DataFrame from list
results = pd.DataFrame(results_list)
# save results to csv with the date and time of the simulation
results.to_csv(f'results_emu_vs_copy_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv', index=False)


