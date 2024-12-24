"""Creates and exports class to be used as one approach to optimizing HyperParameters for a Neural Network.

    Classes:
        RandomSearchTuner : Optimizes HyperParameters of a Neural Network for best performance (minimum evaluation error after training) via a Random Search algorithm.
            Child of HyperParameterTuner.
"""

import logging
import numpy as np
from .hyper_tuner import HyperParamTuner
from .plot_tuning_results import plot_tuning_results

# Set up logger
logger = logging.getLogger('log')

class RecursiveRandomSearchTuner(HyperParamTuner):
    """Optimizes HyperParameters of a Neural Network for best performance (minimum evaluation error after training) via a Random Search algorithm.
    
        Sub-class of HyperParameterTuner.
    
        Args:
            param_set (HyperParameterSet): Set of HyperParameters to vary during optimization ("tuning") process.
            save_file (str, optional): path to file where any tuning performance logs should be saved. Defaults to None.
            optimize_hypers (bool, optional): Whether to vary the values of optimizable HyperParameters ("tune" the HyperParameters), or stick to the initial values provided.
                Defaults to False.
            plot_tuning_results (bool, optional): Whether to create a plot showing the performance for each iteration of HyperParameter tuning. Defaults to False.
            n_value_combinations (int, optional): Number of random samples to take. Defaults to 1.
        
        Additional Class Attributes:
            save_file (str): path to file where tuning performance log will be saved. Filename is "genetic_tune_results.csv".

        Adds Public Attributes to Other Classes:
            HyperParameter objects within param_set:
                values (list): Array of sampled values to use in random search HyperParameter optimization.

        Public Methods:
            randomize_hp_values : Generates list of random values for each HyperParameter to use in a random search HyperParamater optimization.
            tune_hyper_parameters : Performs iterative evaluation of a function that depends on HyperParameters to find an optimal combination of HyperParameters.
    """

    def __init__(self, *args, **kwargs):
        """Constructor for RandomSearchTuner objects.

            Args:
                param_set (HyperParameterSet): Set of HyperParameters to vary during optimization ("tuning") process.
                save_file (str, optional): path to file where any tuning performance logs should be saved. Defaults to None.
                optimize_hypers (bool, optional): Whether to vary the values of optimizable HyperParameters ("tune" the HyperParameters), or stick to the initial values provided.
                    Defaults to False.
                plot_tuning_results (bool, optional): Whether to create a plot showing the performance for each iteration of HyperParameter tuning. Defaults to False.
                n_value_combinations (int, optional): Number of random samples to take. Defaults to 1.
        
            Additional Class Attributes:
                save_file (str): path to file where tuning performance log will be saved. Filename is "genetic_tune_results.csv".

            Adds Public Attributes to Other Classes:
                HyperParameter objects within param_set:
                    values (list): Array of sampled values to use in random search HyperParameter optimization.
        """

        super().__init__(*args, **kwargs)

        # Recursion algorithms as keyword arguments
        p_conf = kwargs.get('p_conf', 0.99)
        self.r_percentile = kwargs.get('r_percentile', None)
        q_conf = kwargs.get('q_conf', 0.99)
        v_expect_imp = kwargs.get('v_expect_imp', None)
        self.c_shrink_ratio = kwargs.get('c_shrink_ratio', 0.5)
        self.s_shrink_thresh = kwargs.get('s_shrink_thresh', 0.001)
        n_explore_samples = kwargs.get('n_explore_samples', None)
        self.l_exploit_samples = kwargs.get('l_exploit_samples', None)

        if self.r_percentile is not None:
            n_explore_samples = int(np.log(1 - p_conf) / np.log(1 - self.r_percentile))+1
        elif n_explore_samples is not None:
            self.r_percentile = 1 - (np.e ** (np.log(1 - p_conf) / n_explore_samples))
        else:
            raise ValueError('One of the following must be specified: r_percentile, n_explore_samples')

        if v_expect_imp is not None:
            self.l_exploit_samples = int(np.log(1 - q_conf) / np.log(1 - v_expect_imp))+1
        elif self.l_exploit_samples is None:
            raise ValueError('One of the following must be specified: v_expect_imp, l_exploit_samples')


        # Generate random values
        if self.optimize_hypers:
            self.n_value_combinations = n_explore_samples
            self.randomize_hp_values_list()
        else:
            # Adjust variables if not optimizing hyper-parameters
            self.n_value_combinations = 1

        self.max_samples = kwargs.get('max_samples', self.n_value_combinations*10)


    # PUBLIC METHODS

    def get_random_hp_value(self, hp):
        value = hp.value
        if isinstance(hp.val_range,list):
            match hp.val_scale:
                case 'linear':
                    value = np.random.uniform(hp.val_range[0], hp.val_range[1])
                case 'log':
                    uniform_vals = np.random.uniform(np.log10(hp.val_range[0]), np.log10(hp.val_range[1]))
                    value = 10**uniform_vals
                case 'none':
                    value = hp.value
                case 'selection':
                    value = np.random.choice(hp.val_range, 1)
                case _:
                    value = hp.value

        return value

    def randomize_hp_values_list(self):
        """Generates list of random values for each HyperParameter to use in a random search HyperParamater optimization.
        
            Side Effects (Modified Attributes):
                For all HyperParameter objects in self.param_set:
                    Modifies object attribute values.
                    values (list): Array of values to use for each model evaluation HyperParameter optimization process.
        """

        for hp in self.param_set.hyper_parameters:
            if isinstance(hp.val_range,list):
                match hp.val_scale:
                    case 'linear':
                        hp.values = np.random.uniform(hp.val_range[0], hp.val_range[1], self.n_value_combinations).tolist()
                    case 'log':
                        uniform_vals = np.random.uniform(np.log10(hp.val_range[0]), np.log10(hp.val_range[1]), self.n_value_combinations)
                        hp.values = (10**uniform_vals).tolist()
                    case 'none':
                        hp.values = [hp.value]*self.n_value_combinations
                    case 'selection':
                        hp.values = np.random.choice(hp.val_range, self.n_value_combinations)
                    case _:
                        hp.values = [hp.value]*self.n_value_combinations
            else:
                hp.values = [hp.value]*self.n_value_combinations

    def realign(self, optimal_ind):
        for hp in self.param_set.hyper_parameters:
            optimal_val = hp.values[optimal_ind]
            range_size = hp.val_range[1]-hp.val_range[0]
            hp.val_range = [optimal_val - range_size/2, optimal_val + range_size/2]

    def shrink(self, optimal_ind):
        self.r_percentile *= self.c_shrink_ratio
        num_dimensions = len(self.param_set.hyper_parameters)

        for hp in self.param_set.hyper_parameters:
            optimal_val = hp.values[optimal_ind]
            og_range_size = hp.val_range[1]-hp.val_range[0]
            new_range_size = og_range_size * self.c_shrink_ratio ** (1/num_dimensions)
            hp.val_range = [optimal_val - new_range_size/2, optimal_val + new_range_size/2]


    def tune_hyper_parameters(self, eval_function, save_function=None, reset_function=None,
                              eval_kwargs=None, save_kwargs=None, reset_kwargs=None, **kwargs):
        """Performs iterative evaluation of a function that depends on HyperParameters to find an optimal combination of HyperParameters. 
        
            Implements random search to optimize.

            Args:
                eval_function (function/method): Function to call in order to evaluate a model that uses the hyper-parameters and return an output value.
                    eval_function must take param_set as an input.
                save_function (function/method, optional): Function to call whenever the best performance (so far) is achieved to save the model config.
                    Defaults to None.
                reset_function (function/method, optional): Function to call when tuning is done in order to reset the model to the best config. 
                    Defaults to None.
                eval_kwargs (dict, optional): Keyword-Arguments to pass to the evaluation function. Defaults to None.
                save_kwargs (dict, optional): Keyword-Arguments to pass to the save function. Defaults to None.
                reset_kwargs (dict, optional): Keyword-Arguments to pass to the reset function. Defaults to None.

            Keyword-Arguments:
                maximize (bool, optional): whether to maximize (True) or minimize (False) the values returned from eval_function. Defaults to False (minimize).
                plot_variables (tuple | list, optional): names of 2 hyper-parameters to use as the x and y axes of the plot optionally generated after tuning 
                    (if self.plot_tuning_results == True). Defaults to None - default behavior described in plot_tuning_results.

            Returns:
                float: Optimal performance across all function evaluations.
        """

        # Handle keyword arguments for each called function
        [eval_kwargs, save_kwargs, reset_kwargs] = [{} if kwargs is None else kwargs for kwargs in [eval_kwargs, save_kwargs, reset_kwargs]]

        maximize = kwargs.get('maximize', False)

        # EXPLORATION
        # Evaluate function for all the Hyper-Parameter combinations in the current layer and track the optimum
        optimal_ind, optimal_perf = self.eval_hp_combinations(eval_function=eval_function, save_function=save_function,
                                                              eval_kwargs=eval_kwargs, save_kwargs=save_kwargs,**kwargs)
        curr_ind = len(self.perf_list)
        # EXPLOITATION
        exploiting = True
        while exploiting and curr_ind < self.max_samples:
            exploiting_ind = 0
            while self.r_percentile > self.s_shrink_thresh and curr_ind < self.max_samples:
                # Randomize value for each hp, add to list of values tested
                for hp in self.param_set.hyper_parameters:
                    hp.values.append(self.get_random_hp_value(hp))
                self.param_set.set_values(curr_ind)
                # Evaluate function
                result = eval_function(param_set=self.param_set, **eval_kwargs)
                eval_perf = result[0] if hasattr(result,'__iter__') else result
                self.perf_list.append(eval_perf)
                # Check for optimal performance
                if (maximize and eval_perf > optimal_perf) or (not maximize and eval_perf < optimal_perf):
                    # Re-align search space to center on new optimal point
                    optimal_ind = curr_ind
                    optimal_perf = eval_perf
                    self.realign(optimal_ind)
                    exploiting_ind = 0
                else:
                    exploiting_ind += 1
                if exploiting_ind == self.l_exploit_samples:
                    # Shrink search space
                    self.shrink(optimal_ind)
                    exploiting_ind = 0
                curr_ind += 1
            # This doesn't match up with the bottom of the pseudocode, but have to stop the while loop somehow...
            exploiting = False


        # Save/Print results
        if self.optimize_hypers:
            self.param_set.set_values(optimal_ind)
            # Save the results of the previous layer
            self._save_hp_tuning_results(filename=self.save_file,
                                         log_name=f'Random Search (n={self.n_value_combinations})', optimal_ind=optimal_ind)

        logger.info('Model Training Complete!')

        # After search finishes, set the model back to the highest performing config
        if reset_function is not None:
            reset_function(**reset_kwargs)

        # Optionally plot results
        if self.plot_tuning_results and len(self.hyper_tuning_table) > 0:
            plot_tuning_results(self.save_file, self.param_set, **kwargs)

        # Return optimal performance
        return self.perf_list[optimal_ind]
