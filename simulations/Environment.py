import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, volatility, certainty, num_repetitions=30, play_mode_all=False):
        """
        This class defines an environment for an n-armed bandit observational learning task, following Charpantier
        et al., 2020.
        :param volatility: float, the rate of change in token values
        :param certainty: array with size num_tokens, representing probability distribution for tokens in slot machines
        :param num_repetitions: int, number of repetitions
        :param play_mode_all: boolean, if True, in 'play' repetitions all machines are available, otherwise only 2
        """
        # Constants:
        self.NUM_MACHINES = 3  # number of slot machines overall
        self.NUM_AVAILABLE_MACHINES = 2  # number of slot machines available in each repetition
        self.NUM_TOKENS = 3  # number of colors in each slot machine
        self.MACHINES = [[0,1,2], [1,2,0], [2,0,1]]  # Define machine mappings (Order tokens in each machine)
        # Parameters
        self.volatility = volatility
        self.certainty = certainty
        self.num_repetitions = num_repetitions
        self.play_mode_all = play_mode_all
        # More parameters
        self.cumulative_reward = 0
        self.resample_steps = 1/self.volatility  # in how many steps the valuable token resampled
        self.cur_repetition = -1     # initialize current repetition
        # memory lists
        self.mem_token_resampled = np.full(self.num_repetitions, np.nan)
        self.mem_valuable_token = np.full(self.num_repetitions, np.nan)
        self.mem_available_machines_id = []
        self.mem_available_machines = np.full(self.num_repetitions, np.nan)
        self.mem_choice_index_model_free = np.full(self.num_repetitions, np.nan)
        self.mem_choice_index_model_based = np.full(self.num_repetitions, np.nan)
        self.mem_choice_index_optimal = np.full(self.num_repetitions, np.nan)
        self.mem_choice_index_copy_last = np.full(self.num_repetitions, np.nan)
        self.mem_which_token_model_based = np.full(self.num_repetitions, np.nan)
        self.mem_which_token_model_free = np.full(self.num_repetitions, np.nan)
        self.mem_which_token_optimal = np.full(self.num_repetitions, np.nan)
        self.mem_which_token_copy_last = np.full(self.num_repetitions, np.nan)
        self.mem_reward_model_based = np.full(self.num_repetitions, np.nan)
        self.mem_reward_model_free = np.full(self.num_repetitions, np.nan)
        self.mem_reward_optimal = np.full(self.num_repetitions, np.nan)
        self.mem_reward_copy_last = np.full(self.num_repetitions, np.nan)
        self.mem_trial_type = pd.Series([''] * self.num_repetitions)

    def get_valuable_token(self):
        """
        Which token is valuable in the current repetition
        For whole numbers (e.g., 4.0): resamples exactly at that step
        For fractional numbers (e.g., 4.5): resamples randomly at floor or ceil step
        Always resamples to a different token than before
        """
        resample_valuable_token = False
        
        if self.cur_repetition == 0:
            resample_valuable_token = True
        else:
            last_resample = max([i for i in range(self.cur_repetition) 
                               if self.mem_token_resampled[i]] + [0])
            steps_since_resample = self.cur_repetition - last_resample
            
            resample_valuable_token = (steps_since_resample >= np.random.choice(
                [np.floor(self.resample_steps), np.ceil(self.resample_steps)]
            ))

        if resample_valuable_token:
            # Get all tokens except the current valuable one
            previous_token = self.mem_valuable_token[self.cur_repetition-1] if self.cur_repetition > 0 else None
            available_tokens = [t for t in range(self.NUM_TOKENS) if t != previous_token]
            valuable_token = np.random.choice(available_tokens, 1)
        else:
            valuable_token = self.mem_valuable_token[self.cur_repetition-1]
        
        self.mem_token_resampled[self.cur_repetition] = resample_valuable_token
        self.mem_valuable_token[self.cur_repetition] = valuable_token
        return valuable_token

    def get_available_machines(self, trial_type):
        """
        Randomly select machines for the current repetition, maintaining consistency with previous 2-machine trials
        :return: list of available machine indices
        """
        #print('Previous machines (from last 2-machine trial):', self.mem_available_machines_id[-1] if self.mem_available_machines_id else None)
        
        # If all machines needed (in play_mode_all during play trials)
        is_play_trial = True if trial_type == 'play' else False
        if self.play_mode_all and is_play_trial:
            available_machines_id = [0, 1, 2]
            self.mem_available_machines_id.append(available_machines_id)
            return available_machines_id
        # For 2-machine trials
        # available_machines_id = np.random.choice(range(self.NUM_MACHINES),
        #                                            self.NUM_AVAILABLE_MACHINES,
        #                                            replace=False)
        # self.mem_available_machines_id.append(available_machines_id)
        # return available_machines_id
    
        if not self.mem_available_machines_id or len(self.mem_available_machines_id[-1]) != 2:
            # If no previous 2-machine trial, sample randomly
            available_machines_id = np.random.choice(range(self.NUM_MACHINES),
                                                   self.NUM_AVAILABLE_MACHINES,
                                                   replace=False)
        else:
            # Use one machine from previous trial and one unused machine
            prev_machines = self.mem_available_machines_id[-1]
            unused_machine = list(set(range(self.NUM_MACHINES)) - set(prev_machines))[0]
            prev_machine = np.random.choice(prev_machines, size=1)[0]
            available_machines_id = [prev_machine, unused_machine]

        self.mem_available_machines_id.append(available_machines_id)
        return available_machines_id

    def new_trial(self, trial_type):
        self.cur_repetition += 1
        self.get_valuable_token()
        self.get_available_machines(trial_type)
        # self.get_available_machines(trial_type)
        # self.get_available_machines(trial_type)
        self.cur_pull_seed = np.random.randint(0, 1000)

    def pull_lever(self, choice, agent_type):
        """
        Operate chosen slot machine and get the reward!
        :param choice: int, the chosen slot machine
        :param agent_type: string, the type of agent who chose the machine
        :return: reward for current repetition
        """
        if choice not in self.mem_available_machines_id[self.cur_repetition]:
            raise ValueError("Chosen machine is not available")
        # add choice to the relevant memory numpy array
        if agent_type == 'optimal':
            self.mem_choice_index_optimal[self.cur_repetition] = choice
        elif agent_type == 'model_free':
            self.mem_choice_index_model_free[self.cur_repetition] = choice
        elif agent_type == 'model_based':
            self.mem_choice_index_model_based[self.cur_repetition] = choice
        elif agent_type == 'copy_last':
            self.mem_choice_index_copy_last[self.cur_repetition] = choice
        chosen_machine = self.MACHINES[choice] # get the chosen machine
        rng = np.random.default_rng(self.cur_pull_seed)
        which_token = rng.choice(chosen_machine, size=1, p=self.certainty)
        if which_token == self.mem_valuable_token[self.cur_repetition]:
            reward = 1
        else:
            reward = 0
        self.cumulative_reward += reward
        # add which token and reward to the relevant memory numpy array
        if agent_type == 'optimal':
            self.mem_which_token_optimal[self.cur_repetition] = which_token
            self.mem_reward_optimal[self.cur_repetition] = reward
        elif agent_type == 'model_free':
            self.mem_which_token_model_free[self.cur_repetition] = which_token
            self.mem_reward_model_free[self.cur_repetition] = reward
        elif agent_type == 'model_based':
            self.mem_which_token_model_based[self.cur_repetition] = which_token
            self.mem_reward_model_based[self.cur_repetition] = reward
        elif agent_type == 'copy_last':
            self.mem_which_token_copy_last[self.cur_repetition] = which_token
            self.mem_reward_copy_last[self.cur_repetition] = reward
        self.mem_trial_type[self.cur_repetition] = 'observe' if agent_type == 'optimal' else 'play'


    def organize_data(self):
        """
        Organize all memory lists of the environment to dataframe
        :return: data frame
        """
        # Creating a repetition index column
        repetition_index = list(range(len(self.mem_token_resampled)))  # assuming all lists have the same length

        # Creating a DataFrame
        df = pd.DataFrame({
            'Repetition_Index': repetition_index,
            'token_resampled': self.mem_token_resampled,
            'valuable_token': self.mem_valuable_token,
            'available_machines_id': self.mem_available_machines_id,
            'trial_type': self.mem_trial_type,
            'choice_index_optimal': self.mem_choice_index_optimal,
            'choice_index_model_free': self.mem_choice_index_model_free,
            'choice_index_model_based': self.mem_choice_index_model_based,
            'choice_index_copy_last': self.mem_choice_index_copy_last,
            'which_token_model_based': self.mem_which_token_model_based,
            'which_token_model_free': self.mem_which_token_model_free,
            'which_token_copy_last': self.mem_which_token_copy_last,
            'reward_model_based': self.mem_reward_model_based,
            'reward_model_free': self.mem_reward_model_free,
            'reward_copy_last': self.mem_reward_copy_last
        })
        return df
    
    def plot_machines(self, rep_to_plot=-1):
        pass







        

  

        




