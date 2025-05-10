import numpy as np
import pandas as pd
from scipy.optimize import minimize
from fit_utils import *
from objectives import *
import itertools
from scipy import stats


def EM_step(model_num, directory, priors_dict, theta_mean, theta_var, bad_subs=[], n_initializations=10):
    """
    Perform one step of the EM algorithm
    """
    spec = get_model_specs(model_num)
    n_participants = len([f for f in os.listdir(directory) if f.endswith('.csv')])-len(bad_subs)
    n_params = len(spec['params'])
    diagadd = 0
    posteriors_dict = {}  # Changed to dictionary
    hessians_dict = {}    # Changed to dictionary
    sigmas_dict = {}      # Changed to dictionary

    options = {
        'maxiter': 200,
        'disp': False,
        'gtol': 1e-2,
    }
    
    # Create objective function that takes params array
    # and returns the negative log likelihood plus the negative log prior
    def objective(param_array):
        param_dict = {name: val for name, val in zip(spec['params'], param_array)}
        nll = spec['objective'](param_dict, data)
        neg_log_prior = 0.5 * np.sum(((param_array - theta_mean)**2) / theta_var)
        return nll + neg_log_prior
    
    # E-step ------------------------------------------------------------
    counter = 0
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            data = load_data(os.path.join(directory, file))
            participant_id = data['participant_id'].iloc[0]
            if participant_id in bad_subs:
                continue

            # Print progress every 10 participants
            if counter % 10 == 0:
                print(f'Fitting participant number {counter} of {n_participants}')
            
            cur_previous_params = priors_dict[participant_id]  # Get previous parameters using participant_id
            # Initialize  search near previous parameters
            is_valid = False
            best_result = None
            for i in range(n_initializations):
                # First initialization is in the previous parameters
                if i == 0:
                    init_params = cur_previous_params
                # For next initializations sample from a uniform distribution around the previous parameters
                else:
                    init_params = np.random.uniform(-3, 3, size=n_params)
                result = minimize(objective, init_params, method='BFGS',
                                  options=options)   
                hess_inv_diag = np.diag(result.hess_inv)
                #Check validity of results
                if np.all(hess_inv_diag > 0.001) and result.success and np.isreal(result.fun):
                    if best_result is None or result.fun < best_result.fun:
                        best_result = result
                        is_valid = True
            # If no valid result found, enter while loop with random initializations
            if not is_valid:
                print(f'No valid result found for participant {counter}, entering while loop')
                while_counter = 0
                while not is_valid:
                    while_counter += 1
                    if while_counter > 10:
                        best_result = result
                        break
                    for i in range(n_initializations):
                        init_params = np.random.uniform(-3, 3, size=n_params)
                        result = minimize(objective, init_params, method='BFGS',
                                  options=options)   
                        hess_inv_diag = np.diag(result.hess_inv)
                        if np.all(hess_inv_diag > 0.001) and result.success and np.isreal(result.fun):
                            is_valid = True
                            best_result = result
            # Save best result
            posteriors_dict[participant_id] = best_result.x
            hessians_dict[participant_id] = np.diag(np.linalg.inv(best_result.hess_inv))# Add diagonal of the hessian
            counter += 1
            
    # M-step ------------------------------------------------------------
    # Convert dictionaries to arrays for calculations
    posteriors_array = np.array([posteriors_dict[pid] for pid in posteriors_dict])
    new_theta_mean = np.mean(posteriors_array, axis=0)
    
    A = np.zeros(n_params)
    for pid in posteriors_dict:
        sigma = 1 / hessians_dict[pid]
        sigmas_dict[pid] = sigma
        A += posteriors_dict[pid] ** 2 + sigma
    
    new_theta_sigma = A / n_participants - new_theta_mean ** 2 + diagadd * np.ones(n_params)
    
    # Convert sigmas_dict to array for calculation
    sigmas_array = np.array([sigmas_dict[pid] for pid in sigmas_dict])
    
    # Compute joint distribution (log-posterior)
    log_posterior = (
        np.log(np.prod(2 * np.pi * new_theta_sigma)) +
        np.sum(
            np.sum(
                0.5 * ((new_theta_mean - posteriors_array) ** 2 / new_theta_sigma), axis=1
            )
        ) +
        0.5 * np.sum(1 / new_theta_sigma * sigmas_array)
    )
            
    # Convert final parameters back to dictionary and back to interpetable values
    final_params = transform_params(
        {name: val for name, val in zip(spec['params'], new_theta_mean)}, 
        spec['types']
    )
    param_dict = {name: float(val) for name, val in zip(spec['params'], final_params)}
    
    return param_dict, log_posterior, new_theta_mean, new_theta_sigma, posteriors_dict, sigmas_dict

    


def calculate_iBIC(model_num, directory, theta_mean, theta_sigma, sample_size, bad_subs):
    """
    Compute iBIC using Monte Carlo sampling as in Charpentier et al., 2020.
    """
    # Get model specifications (parameters, values, objective function)
    spec = get_model_specs(model_num)
    n_params = len(spec['params'])

    # Sample from the group-level prior distribution
    samples = np.random.multivariate_normal(theta_mean, np.diag(theta_sigma), sample_size)

    n_participants = len([f for f in os.listdir(directory) if f.endswith('.csv') and f not in bad_subs])
    bics = np.zeros(n_participants)
    l_sessions = np.zeros(n_participants)

    counter = 0
    for file in os.listdir(directory):
        if file.endswith('.csv'): 
            data = load_data(os.path.join(directory, file))
            if data['participant_id'].iloc[0] in bad_subs:
                continue
            print(f'Integrating BIC for participant number {counter} of {n_participants}')
            likelihoods = np.zeros(sample_size)

            # Compute likelihood for each sample
            for i, sample in enumerate(samples):
                likelihoods[i] = spec['objective'](sample, data)  # Likelihood function (not posterior)

            L = -likelihoods
            delL = L - np.max(L)  # Shift for numerical stability
            A = np.exp(delL)
            bics[counter] = np.log(np.mean(A)) + np.max(L)
            l_sessions[counter] = np.sum(data['task_type'] == 'play')

            counter += 1

    # Compute the iBIC score
    total_choices = np.sum(l_sessions)
    iBIC = -2 * np.sum(bics) + n_params * 2 * np.log(total_choices)

    return iBIC



def fit_hierarchical(model_num, directory, max_iter=1000, tol=10**-4, bad_subs=[]):
    """
    Fit a hierarchical model to the data
    
    Parameters:
    -----------
    model_num : int
        Model number
    directory : str
        Directory containing the data
    n_initalizations_individual : int, optional
        Number of random initializations for indevidual participant model fitting (default: 2)
     max_iter : int, optional
        Maximum number of iterations for the global model fitting (default: 20)
    tol : float, optional
        Convergence tolerance for relative change in log posterior (default: 1e-4)
    """
    # Get model specifications (parameters, values, objective function)
    spec = get_model_specs(model_num)
    
    # Initialize group-level parameters - zero in the unbounded space
    theta_mean = np.zeros(len(spec['params']))
    theta_var = np.ones(len(spec['params']))
    priors_dict = {file.split('.')[0]: np.zeros(len(spec['params'])) for file in os.listdir(directory) if file.split('.')[0] not in bad_subs}

    current_iter = 0
    log_posterior = float('-inf')
    while current_iter <= max_iter:
        current_iter += 1
        param_dict, new_log_posterior, new_theta_mean, new_theta_sigma, posteriors_dict, sigmas_dict = EM_step(model_num, directory, priors_dict, theta_mean, theta_var, bad_subs)
        # Iteratively compute group level nll
        nll = 0
        counter = 0
        for file in os.listdir(directory):
            if file.endswith('.csv'):
                data = load_data(os.path.join(directory, file))
                participant_id = data['participant_id'].iloc[0]  # Get the single value
                if participant_id in bad_subs:
                    continue
                nll += spec['objective'](posteriors_dict[participant_id], data)
                counter += 1
        
        print(f'Iteration {current_iter}, NLL: {nll}, log_posterior: {new_log_posterior}, parameters: {param_dict}, new_theta_mean: {new_theta_mean}, new_theta_sigma: {new_theta_sigma}')
        # Check for convergence
        if log_posterior != float('-inf'):
            relative_change = np.abs(new_log_posterior - log_posterior)/abs(log_posterior)
            print(f'Relative change in log posterior: {relative_change}')
            if relative_change < tol:
                print(f'Converged with relative change {relative_change} < tolerance {tol}')
                break
        # Update priors
        priors_dict = posteriors_dict
        theta_mean = new_theta_mean
        theta_var = new_theta_sigma
        log_posterior = new_log_posterior
    iBIC = calculate_iBIC(model_num, directory, theta_mean, theta_var, 1000, bad_subs)
    print(f'EM converged after {current_iter} iterations, iBIC: {iBIC}')
    return param_dict, log_posterior, new_theta_mean, new_theta_sigma, posteriors_dict, sigmas_dict, iBIC, nll

if __name__ == '__main__':
    directory = 'data'
    bad_subs = (pd.read_csv('Ss_to_exclude.csv')['x']).tolist()
    # Initialize summary dataframe with columns for model comparison
    summary_columns = [
        'model_num', 
        'model_description',
        'negative_log_likelihood',
        'iBIC',
        'parameter_means',
        'ttest_results',
        'bayesian_simulation_results',
        'cohens_d'
    ]
    # Create empty dataframe
    summary_df = pd.DataFrame(columns=summary_columns)
    
    # Dictionary mapping model numbers to descriptions
    model_descriptions = {
        1: "Goal emulation with estimated lambda",
        2: "RL imitation with estimated alpha",
        3: "Copy last action",
        4: "Emulation and RL imitation with one weight",
        5: "Emulation and RL imitation with stakes-dependent weights", 
        6: "Emulation and copy last action with one weight",
        7: "Emulation and copy last action with stakes-dependent weights"
    }
    for model_num in range(1,8):
        #initialize summary dictionary for the overall dataframe
        summary_dict = {}
        print(f'Fitting model {model_num}')
        # add model number and description to summary dictionary
        summary_dict['model_num'] = model_num
        summary_dict['model_description'] = model_descriptions[model_num]
        # Get model specification for parameter transformations
        spec = get_model_specs(model_num)
        output_file = f'model{model_num}_posteriors.csv'
        # Fit the hierarchical model
        param_dict, log_posterior, new_theta_mean, new_theta_sigma, posteriors_dict, sigmas_dict, iBIC, nll = fit_hierarchical(model_num, directory, tol=10**-3, bad_subs=bad_subs)
        # Add param dict as a single string to summary dataframe
        summary_dict['parameter_means'] = str(param_dict)
        # Add iBIC to summary dataframe
        summary_dict['iBIC'] = iBIC
        # Add nll to summary dataframe
        summary_dict['negative_log_likelihood'] = nll
        
        # Create list to store transformed posteriors
        transformed_posteriors = []
        
        # Transform parameters for each participant
        for participant_id, posterior in posteriors_dict.items():
            # Transform parameters back to interpretable units
            transformed_params = transform_params(posterior, spec['types'])
            
            # Create row with participant ID and transformed parameters
            row = [participant_id] + list(transformed_params)  # Convert to list
            transformed_posteriors.append(row)
            
        # Create column names
        columns = ['participant_id'] + spec['params']
        
        # Convert to dataframe
        posteriors_df = pd.DataFrame(transformed_posteriors, columns=columns)
        # Save to CSV
        posteriors_df.to_csv(output_file, index=False)
        print(f"Saved transformed posteriors to {output_file}")
        print(param_dict)
        
        
        # re-transform the posteriors
        posteriors_array = np.array(posteriors_df[spec['params']])
        posteriors_array_unbounded = []
        for row in posteriors_array:
            row = inverse_transform_params(row, spec['types'])
            posteriors_array_unbounded.append(row)
        posteriors_array_unbounded = np.array(posteriors_array_unbounded)
        means_unbounded = np.mean(posteriors_array_unbounded, axis=0)
        variances_unbounded = np.var(posteriors_array_unbounded, axis=0)
        
        # Only do group-level comparisons for mixture models with stakes-dependent weights
        if model_num in [5, 7]:
            # Group-level comparison
            print("\nStatistical comparison of w_em_high vs w_em_low:")
            high_idx = spec['params'].index('w_em_high')
            low_idx = spec['params'].index('w_em_low')
            
            print("Group-level analysis:")
            print(f"Mean w_em_high: {means_unbounded[high_idx]:.3f} ± {np.sqrt(variances_unbounded[high_idx]):.3f}")
            print(f"Mean w_em_low: {means_unbounded[low_idx]:.3f} ± {np.sqrt(variances_unbounded[low_idx]):.3f}")
            n_participants = len(posteriors_dict)
            #1. Paired t.test
            t_stat, p_value = stats.ttest_rel(posteriors_array_unbounded[:, high_idx], posteriors_array_unbounded[:, low_idx])
            print(f"Paired t-test: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")
            # Save ttest result as string to summary dataframe
            summary_dict['ttest_results'] = f"t({n_participants-1}) = {t_stat:.3f}, p = {p_value:.3f}"
            
            # 2. monte-carlo sampling-based inference
            n_samples = 10000
            group_means_high = np.zeros(n_samples)
            group_means_low = np.zeros(n_samples)
            
            # For each sample
            for i in range(n_samples):
                # Simulate each participant
                sample_high = []
                sample_low = []
                for pid in posteriors_dict:
                    # Get individual mean and variance for each parameter
                    individual_params = posteriors_dict[pid]
                    individual_vars = sigmas_dict[pid]  # These are already inverted hessian diagonals
                    
                    # Sample from individual posterior
                    high_sample = np.random.normal(individual_params[high_idx], 
                                                np.sqrt(individual_vars[high_idx]))
                    low_sample = np.random.normal(individual_params[low_idx], 
                                                np.sqrt(individual_vars[low_idx]))
                    
                    sample_high.append(high_sample)
                    sample_low.append(low_sample)
                
                # Calculate group means for this sample
                group_means_high[i] = np.mean(sample_high)
                group_means_low[i] = np.mean(sample_low)
            
            # Calculate difference distribution
            diff_samples = group_means_high - group_means_low
            
            # Compute probability and credible intervals
            prob_positive = np.mean(diff_samples > 0)
            ci_lower, ci_upper = np.percentile(diff_samples, [2.5, 97.5])
            
            print("\nMCMC-based analysis:")
            print(f"P(mean w_em_high > mean w_em_low) = {prob_positive:.3f}")
            print(f"95% Credible Interval for difference: [{ci_lower:.3f}, {ci_upper:.3f}]")
            print(f"Mean difference: {np.mean(diff_samples):.3f}")
            # Save results to summary dataframe as a single string
            summary_dict['bayesian_simulation_results'] = f"P(mean w_em_high > mean w_em_low) = {prob_positive:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]"
            
            # Calculate Cohen's dz for paired samples
            individual_diffs = []
            for pid in posteriors_dict:
                params = posteriors_dict[pid]
                individual_diffs.append(params[high_idx] - params[low_idx])
            
            cohens_dz = np.mean(individual_diffs) / np.std(individual_diffs)
            print(f"Cohen's dz (paired): {cohens_dz:.3f}")
            # Save results to summary dataframe as a number
            summary_dict['cohens_d'] = cohens_dz
        
        # Add summary dictionary to summary dataframe
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_dict])], ignore_index=True)
    # Save summary dataframe to csv
    summary_df.to_csv('summary_hierarchical_fits.csv', index=False)

    
    