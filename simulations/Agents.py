import numpy as np
from abc import abstractmethod


class Agent:
    """
    This is an abstract class for agents in the observational learning task
    :param env: Environment object
    """
    def __init__(self,env):
        self.env = env

    def do_repetition(self):
        choice, agent_type = self.choose_machine()
        self.env.pull_lever(choice, agent_type)

    @abstractmethod
    def choose_machine(self):
        pass

    @abstractmethod
    def observe_behavior(self):
        pass


class OptimalAgent(Agent):
    """
    This class implements an optimal, know-all, agent who always chooses the best slot machine
    out of the available machines. The other agents are learning by observing this agent
    :param env: Environment object
    """
    def __init__(self,env):
        super().__init__(env)

    def choose_machine(self):
        available_machines_id = self.env.mem_available_machines_id[self.env.cur_repetition]
        valuable_token = self.env.mem_valuable_token[self.env.cur_repetition]
        # Find for which of the available machines, the valuable token is located earlier (i.e., have higher chance)
        locs_in_machine = []
        for i in available_machines_id:
            cur_machine = self.env.MACHINES[i]
            locs_in_machine.append(np.where(cur_machine==valuable_token)[0][0])
        choice = available_machines_id[np.argmin(locs_in_machine)]
        agent_type = 'optimal'

        return choice, agent_type

    def observe_behavior(self):
        pass
    
class CopyLastObserver(Agent):
    """
    This class implements a copy-last observer, equivalent to the 1 step imitation model in Charpentier et al., 2020
    """
    def __init__(self, env, beta):
        super().__init__(env)
        self.beta = beta
        self.last_observed_choice = None

    def choose_machine(self):
        """
        If last observed choice is available, choose it with softmax. Otherwise, random choice
        """
        available_machines_id = self.env.mem_available_machines_id[self.env.cur_repetition]
        # initialize machine values list in length of available machines
        machine_values = [0] * len(available_machines_id)
        if self.last_observed_choice in available_machines_id:
            machine_values[available_machines_id.index(self.last_observed_choice)] = 1
            # softmax probabilities
            softmax_probs = np.exp(self.beta * np.array(machine_values))
            softmax_probs = softmax_probs / np.sum(softmax_probs)
            choice = np.random.choice(available_machines_id, p=softmax_probs)
        else:
            choice = np.random.choice(available_machines_id)
        agent_type = 'copy_last'
        return choice, agent_type

    def observe_behavior(self):
        if self.env.mem_trial_type[self.env.cur_repetition] == 'observe':
            self.last_observed_choice = self.env.mem_choice_index_optimal[self.env.cur_repetition]


class ModelFreeObserver(Agent):
    """
    This class implements a model-free social learner (imitation social learner), equivalent to model #3 in
    Charpentier et al., 2020
    :param env: Environment object
    :param alpha: float, learning rate
    """
    def __init__(self, env, alpha, beta):
        super().__init__(env)
        self.alpha = alpha
        self.beta = beta
        self.action_values = [0, 0, 0]  # Initial action values
        self.mem_action_values = []

    def choose_machine(self):
        """
        Choose machine based on softmax probabilities using paper's equation:
        P_left(t) = 1 / (1 + exp(β*(AV_right(t) - AV_left(t))))
        """
        available_machines_id = self.env.mem_available_machines_id[self.env.cur_repetition]
        available_machines_values = [self.action_values[i] for i in available_machines_id]
        
        # For two machines:
        if len(available_machines_values) == 2:
            # Probability of choosing left machine (index 0)
            p_left = 1 / (1 + np.exp(self.beta * (available_machines_values[1] - available_machines_values[0])))
            
            # Choose based on this probability
            if np.random.random() < p_left:
                choice_idx = 0
            else:
                choice_idx = 1
        else:
            # For three machines, use standard softmax
            softmax_probs = np.exp(self.beta * np.array(available_machines_values))
            softmax_probs = softmax_probs / np.sum(softmax_probs)
            choice_idx = np.random.choice(len(available_machines_id), p=softmax_probs)
        
        choice = available_machines_id[choice_idx]
        agent_type = 'model_free'
        return choice, agent_type

    def observe_behavior(self):
        """
        Update action values based on the optimal agent's choice
        """
        if self.env.mem_trial_type[self.env.cur_repetition] == 'observe':
            for i in self.env.mem_available_machines_id[self.env.cur_repetition]:  # For each available machine
                if i == self.env.mem_choice_index_optimal[self.env.cur_repetition]:  # If optimal agent chose machine
                    self.action_values[i] += self.alpha * (1 - self.action_values[i])
                else:  # if the optimal agent chose the other machine
                    self.action_values[i] += self.alpha * (-1 - self.action_values[i])
        self.mem_action_values.append(np.round(self.action_values.copy(), 4))  # Save action values for later analysis


class ModelBasedObserver(Agent):
    """
    This class implements a model-based social learner (goal emulation social learner), equivalent to model #2 in
    Charpentier et al., 2020
    :param env: Environment object
    """
    def __init__(self, env, lambd, beta):
        super().__init__(env)
        self.lambd = lambd
        self.token_probs = [1/3, 1/3, 1/3] # initial probabilities for each token bein valuable
        self.mem_token_probs = []
        self.mem_expected_valuable_token = []
        self.beta = beta

    def choose_machine(self):
        """
        Choose machine based on softmax probabilities using paper's equation:
        P_left(t) = 1 / (1 + exp(β*(AV_right(t) - AV_left(t))))
        """
        available_machines_id = self.env.mem_available_machines_id[self.env.cur_repetition]
        
        # Calculate action values for each machine
        action_values = []
        for i in available_machines_id:
            cur_machine = self.env.MACHINES[i]
            cur_machine_prob = [self.env.certainty[cur_machine.index(j)] for j in range(len(self.env.certainty))]
            action_values.append(np.dot(self.token_probs, cur_machine_prob))
        
        # For two machines:
        if len(action_values) == 2:
            # Probability of choosing left machine (index 0)
            p_left = 1 / (1 + np.exp(self.beta * (action_values[1] - action_values[0])))
            
            # Choose based on this probability
            if np.random.random() < p_left:
                choice_idx = 0
            else:
                choice_idx = 1
        else:
            # For three machines, use standard softmax
            softmax_probs = np.exp(self.beta * np.array(action_values))
            softmax_probs = softmax_probs / np.sum(softmax_probs)
            choice_idx = np.random.choice(len(available_machines_id), p=softmax_probs)
        
        choice = available_machines_id[choice_idx]
        agent_type = 'model_based'
        return choice, agent_type

    def observe_behavior(self):
        """
        Update token probabilities to be valuable based on the optimal agent's choice
        """
        available_machines_id = self.env.mem_available_machines_id[self.env.cur_repetition]
        if self.env.mem_trial_type[self.env.cur_repetition] == 'observe':
            new_token_probs = []
            for i in range(self.env.NUM_TOKENS): # For each token
                # prior: Equation 2 in Charpentier et al., 2020
                prior = self.lambd*self.token_probs[i] + (1-self.lambd)*(np.sum(self.token_probs)-self.token_probs[i])/2
                # likelihood: 1 when given token is valuable, the optimal agent would choose the chosen machine, 0 otherwise
                locs_in_machine = []
                for j in available_machines_id:
                    cur_machine = self.env.MACHINES[j]
                    locs_in_machine.append(np.where(cur_machine==np.array([i]))[0][0])
                expected_choice = available_machines_id[np.argmin(locs_in_machine)]
                likelihood = 1 if expected_choice == self.env.mem_choice_index_optimal[self.env.cur_repetition] else 0
                # posterior: Equation 1 in Charpentier et al., 2020
                new_token_probs.append(prior * likelihood)
            self.token_probs = np.array(new_token_probs)
            #normalize token probs to sum to 1
            self.token_probs = self.token_probs/np.sum(self.token_probs)
        self.mem_token_probs.append(np.round(self.token_probs.copy(), 4)) # Save token probabilities for later analysis
        self.mem_expected_valuable_token.append(np.argmax(self.token_probs)) # Save expected valuable token for later analysis