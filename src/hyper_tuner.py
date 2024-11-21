import numpy as np
import pandas as pd
from nn_plot_functions import plot_grid_search_results


class HyperParameter():
    def __init__(self, name, optimizable=False, init_value=0,
                 val_range=None, val_scale='none', num_steps=1):
        if not val_range:
            val_range = [0, 0]
        self.name = name
        self.value = init_value
        self.values = init_value # This may be overwritten by a HyperParameterSet object, if optimizing
        self.optimizable = optimizable
        self.val_range = val_range
        self.val_scale = val_scale
        self.num_steps = num_steps
        self.gen_gridpoints()

    def gen_gridpoints(self):
        if self.optimizable:
            match self.val_scale:
                case 'linear':
                    self.gridpoints = np.interp(range(self.num_steps), [
                                                0, self.num_steps - 1], self.val_range)
                case 'log':
                    self.gridpoints = 10**np.interp(
                        range(self.num_steps), [0, self.num_steps - 1], np.log10(self.val_range))
                # Not as sure about the proper handling of these:
                case 'none':
                    self.val_range = self.value
                    self.gridpoints = self.val_range
                    self.num_steps = 1
                case 'selection':
                    self.gridpoints = self.val_range
                    self.num_steps = len(self.val_range)
                case _:
                    self.gridpoints = self.val_range
                    self.num_steps = 1
        else:
            self.gridpoints = [self.value]
            self.num_steps = 1

    def set_value(self, i):
        self.value = self.values[i]


class HyperParameterSet():
    def __init__(self, hyper_parameters, optimize=True):
        self.hyper_parameters = hyper_parameters
        self.optimize = optimize
        if self.optimize:
            self.gen_grid()
        else:
            self.total_gridpoints = 1
            for hp in self.hyper_parameters:
                hp.values = [hp.value]

    def get(self,hp_name):
        # Returns the hyper-parameter in the set with the provided name.
        hp_names = [hp.name for hp in self.hyper_parameters]
        return self.hyper_parameters[hp_names.index(hp_name)]

    def gen_grid(self):
        # For each hyperparameter in the set, generates a list of all values to use in order.
        # The values are organized such that every index in the list represents a unique combination
        # of values across all the hyperparamters (ie a unique point on the
        # n-dimensional grid of hyperparameters)
        self.total_gridpoints = 1
        for ind, hp in enumerate(self.hyper_parameters):
            hp.values = np.repeat(hp.gridpoints, self.total_gridpoints)
            for prev in self.hyper_parameters[:ind]:
                prev.values = np.array(list(prev.values) * hp.num_steps)

            self.total_gridpoints *= hp.num_steps

    def set_values(self, ind):
        if self.optimize:
            for hp in self.hyper_parameters:
                hp.set_value(ind)

    def refine_grid(self, ind):
        # "Zoom in" on the area of interest and generate new gridpoints closer to the provided index
        # Find new value ranges for each hyperparameter
        for hp in self.hyper_parameters:
            center_val = hp.values[ind]
            # gridpoints_ind = int(np.where(hp.gridpoints==center_val)[0])
            gridpoints_ind = list(hp.gridpoints).index(center_val)
            match hp.val_scale:
                case 'linear':
                    if gridpoints_ind == 0:
                        minval = center_val
                        maxval = (
                            hp.gridpoints[gridpoints_ind] + hp.gridpoints[gridpoints_ind + 1]) / 2
                    elif gridpoints_ind == hp.num_steps - 1:
                        minval = (
                            hp.gridpoints[gridpoints_ind - 1] + hp.gridpoints[gridpoints_ind]) / 2
                        maxval = center_val
                    else:
                        minval = (
                            hp.gridpoints[gridpoints_ind - 1] + hp.gridpoints[gridpoints_ind]) / 2
                        maxval = (
                            hp.gridpoints[gridpoints_ind] + hp.gridpoints[gridpoints_ind + 1]) / 2
                    hp.val_range = [minval, maxval]
                case 'log':
                    if gridpoints_ind == 0:
                        minval = center_val
                        maxval = 10**((np.log10(hp.gridpoints[gridpoints_ind]) + np.log10(
                            hp.gridpoints[gridpoints_ind + 1])) / 2)
                    elif gridpoints_ind == hp.num_steps - 1:
                        minval = 10**((np.log10(hp.gridpoints[gridpoints_ind - 1]) + np.log10(
                            hp.gridpoints[gridpoints_ind])) / 2)
                        maxval = center_val
                    else:
                        minval = 10**((np.log10(hp.gridpoints[gridpoints_ind - 1]) + np.log10(
                            hp.gridpoints[gridpoints_ind])) / 2)
                        maxval = 10**((np.log10(hp.gridpoints[gridpoints_ind]) + np.log10(
                            hp.gridpoints[gridpoints_ind + 1])) / 2)
                    hp.val_range = [minval, maxval]
                case 'selection':
                    # No setting a range, just select the value that performed
                    # best
                    hp.val_range = center_val
            # Generate new grid points for the hyperparameter based on its
            # "zoomed in" values of interest
            hp.gen_gridpoints()

        # After refining gridpoints, generate new array of points
        self.gen_grid()

class HyperParamTuner():
    def __init__(self, param_set, save_file, settings):
        self.param_set = param_set
        self.save_file = save_file
        # Assign settings common to all tuners
        self.optimize_hypers = settings.get('optimize_hypers',False)
        self.plot_tuning_results = settings.get('plot_tuning_results',False)

        # Initialize attributes to use later in tuning
        self.model_perf_list = [] # List of model performance values for all combos of hyperparameter values
        self.hyper_tuning_table = []


    def save_hp_tuning_results(self,layer=0,filename=None):
        # Create array of all hyperparameter values in current run ("layer") of grid search
        curr_results_table = []
        for hp in self.param_set.hyper_parameters:
            curr_results_table.append(hp.values)
        # Add performance of the model to the array
        curr_results_table.append(self.model_perf_list)
        # Add layer number to array
        curr_results_table.append([layer]*len(self.model_perf_list))

        # Append the transpose of the curr_results_table to the results_array
        self.hyper_tuning_table = self.hyper_tuning_table + list(map(list, zip(*curr_results_table)))

        if filename:
            # Convert to dataframe (in order to add column labels) and save to file
            column_names = [hp.name for hp in self.param_set.hyper_parameters]
            column_names.extend(['Model Performance','Grid Search Layer'])
            pd.DataFrame(self.hyper_tuning_table,columns=column_names).to_csv(filename)


class GridSearchTuner(HyperParamTuner):
    def __init__(self, param_set, save_file, settings):
        # Initialize HyperParamTuner
        super().__init__(param_set, save_file, settings)

        # Assign settings unique to Grid Search
        self.hyper_tuner_layers = settings.get('hyper_tuner_layers',1)
        self.hyper_tuner_steps_per_dim = settings.get('hyper_tuner_steps_per_dim',3)
        self.scale_epochs_over_layers = settings.get('scale_epoch_over_layers',False)

        # Adjust variables if not optimizing hyper-parameters
        if not self.optimize_hypers:
            self.hyper_tuner_layers = 1


    def next_hp_layer(self,neural_net,tune_layer):
        min_grid_index = np.nanargmin(self.model_perf_list)
        if self.optimize_hypers:
            self.save_hp_tuning_results(tune_layer,self.save_file)
            print(
                f'Layer {tune_layer+1} '
                f'Complete. Optimal performance: '
                f'{self.model_perf_list[min_grid_index]}. '
                f'Hyper-parameters used: '
                )
            for hp in self.param_set.hyper_parameters:
                print(f"\t{hp.name} = {hp.values[min_grid_index]}")
        if tune_layer < self.hyper_tuner_layers-1:
            self.param_set.refine_grid(min_grid_index)
            if self.scale_epochs_over_layers:
                neural_net.max_epochs*= 2
                neural_net.n_epochs_to_stop*= 2
            self.model_perf_list = []
            print('Beginning next hyper-parameter optimization iteration.')
        else:
            print('Model Training Complete!')


    def tune_neural_net(self, net, training_data, validation_data):
        for tune_layer in range(self.hyper_tuner_layers):
            print(f'\nOptimization Round {tune_layer+1} of {self.hyper_tuner_layers}\n-------------------------------')
            # Iterate through all combinations of hyperparameters
            for grid_ind in range(self.param_set.total_gridpoints):
                # Set and display hyperparameters for current run
                self.param_set.set_values(grid_ind)
                print(f'\nHP Grid Point {grid_ind+1} of {self.param_set.total_gridpoints}: -------------------- ')
                for hp in self.param_set.hyper_parameters:
                    print(f"\t{hp.name} = {hp.value}")

                train_dataloader, validation_dataloader = net.configure_for_training(self, training_data, validation_data)

                # ---------------------
                # Model Training and Validation Testing
                # ---------------------
                val_perfs = net.train_and_validate(self, train_dataloader, validation_dataloader)

                # Track validation performance for the set of hyperparameters used
                self.model_perf_list.append(val_perfs[-1])
                # Save the model if it is the best performing so far
                if grid_ind == np.nanargmin(self.model_perf_list):
                    net.save_model()

            # Print some results, determine whether to perform another layer of grid search, and if so, refine the mesh
            self.next_hp_layer(net,tune_layer)

        # After grid search finishes, plot results
        if len(self.hyper_tuning_table) > 0 and self.plot_tuning_results:
            plot_grid_search_results(self.save_file,self.param_set,variables=('learning_rate','lmbda'))

        # Set the model back to the highest performing config
        net.model = net.load_model(net.save_file,print_loaded_model=False)
