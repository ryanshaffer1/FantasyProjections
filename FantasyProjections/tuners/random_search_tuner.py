"""Creates and exports class to be used as one approach to optimizing HyperParameters for a Neural Network.

    Classes:
        RandomSearchTuner : Optimizes HyperParameters of a Neural Network for best performance (minimum evaluation error after training) via a Random Search algorithm.
            Child of HyperParameterTuner.
"""  # fmt:skip

import logging

from .hyper_tuner import HyperParamTuner
from .plot_tuning_results import plot_tuning_results

# Set up logger
logger = logging.getLogger("log")


class RandomSearchTuner(HyperParamTuner):
    """Optimizes HyperParameters of a Neural Network for best performance (minimum evaluation error after training) via a Random Search algorithm.

        Sub-class of HyperParameterTuner.

        Args:
            param_set (HyperParameterSet): Set of HyperParameters to vary during optimization ("tuning") process.
            save_file (str, optional): path to file where any tuning performance logs should be saved. Defaults to None.
            optimize_hypers (bool, optional): Whether to vary the values of optimizable HyperParameters ("tune" the HyperParameters), or stick to the initial values provided.
                Defaults to False.
            plot_tuning_results (bool, optional): Whether to create a plot showing the performance for each iteration of HyperParameter tuning. Defaults to False.
            n_value_combinations (int, optional): Number of random samples to take. Defaults to 1.

        Adds Public Attributes to Other Classes:
            HyperParameter objects within param_set:
                values (list): Array of sampled values to use in random search HyperParameter optimization.

        Public Methods:
            tune_hyper_parameters : Performs iterative evaluation of a function that depends on HyperParameters to find an optimal combination of HyperParameters.

    """  # fmt:skip

    def __init__(self, *args, n_value_combinations=1, **kwargs):
        """Constructor for RandomSearchTuner objects.

            Args:
                args:
                    param_set (HyperParameterSet): Set of HyperParameters to vary during optimization ("tuning") process.
                    save_file (str, optional): path to file where any tuning performance logs should be saved. Defaults to None.
                    optimize_hypers (bool, optional): Whether to vary the values of optimizable HyperParameters ("tune" the HyperParameters), or stick to the initial values provided.
                        Defaults to False.
                    plot_tuning_results (bool, optional): Whether to create a plot showing the performance for each iteration of HyperParameter tuning. Defaults to False.
                n_value_combinations (int, optional): Number of random samples to take. Defaults to 1.
                kwargs:
                    All keyword arguments are passed to HyperParamTuner. See that class's documentation for descriptions and valid inputs.

            Additional Class Attributes:
                save_file (str): path to file where tuning performance log will be saved. Filename is "genetic_tune_results.csv".

            Adds Public Attributes to Other Classes:
                HyperParameter objects within param_set:
                    values (list): Array of sampled values to use in random search HyperParameter optimization.

        """  # fmt:skip

        super().__init__(*args, **kwargs)

        self.n_value_combinations = n_value_combinations

        # Generate random values
        if self.optimize_hypers:
            self._randomize_hp_values()
        else:
            # Adjust variables if not optimizing hyper-parameters
            self.n_value_combinations = 1

    # PUBLIC METHODS

    def tune_hyper_parameters(
        self,
        eval_function,
        save_function=None,
        reset_function=None,
        eval_kwargs=None,
        save_kwargs=None,
        reset_kwargs=None,
        **kwargs,
    ):
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
                kwargs:
                    maximize (bool, optional): whether to maximize (True) or minimize (False) the values returned from eval_function. Defaults to False (minimize).
                    plot_variables (tuple | list, optional): names of 2 hyper-parameters to use as the x and y axes of the plot optionally generated after tuning
                        (if self.plot_tuning_results == True). Defaults to None - default behavior described in plot_tuning_results.

            Returns:
                float: Optimal performance across all function evaluations.

        """  # fmt:skip

        # Handle keyword arguments for each called function
        [eval_kwargs, save_kwargs, reset_kwargs] = [
            {} if kwargs is None else kwargs for kwargs in [eval_kwargs, save_kwargs, reset_kwargs]
        ]

        # Evaluate function for all the Hyper-Parameter combinations in the current layer and track the optimum
        optimal_ind, _ = self.eval_hp_combinations(
            eval_function=eval_function,
            save_function=save_function,
            eval_kwargs=eval_kwargs,
            save_kwargs=save_kwargs,
            **kwargs,
        )

        # Save/Print results
        if self.optimize_hypers:
            self.param_set.set_values(optimal_ind)
            # Save the results of the previous layer
            self._save_hp_tuning_results(
                filename=self.save_file,
                log_name=f"Random Search (n={self.n_value_combinations})",
                optimal_ind=optimal_ind,
            )

        logger.info("Model Training Complete!")

        # After search finishes, set the model back to the highest performing config
        if reset_function is not None:
            reset_function(**reset_kwargs)

        # Optionall plot results
        if self.plot_tuning_results and len(self.hyper_tuning_table) > 0:
            plot_tuning_results(self.save_file, self.param_set, **kwargs)

        # Return optimal performance
        return self.perf_list[optimal_ind]
