from dataclasses import dataclass
import numpy as np
from .plot_grid_search_results import plot_grid_search_results
from .hyper_tuner import HyperParamTuner

@dataclass
class GridSearchTuner(HyperParamTuner):
    hyper_tuner_layers: int = 1
    hyper_tuner_steps_per_dim: int = 3
    scale_epochs_over_layers: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Adjust variables if not optimizing hyper-parameters
        if not self.optimize_hypers:
            self.hyper_tuner_layers = 1


    # PUBLIC METHODS

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

                train_dataloader, validation_dataloader = net.configure_for_training(self.param_set, training_data, validation_data)

                # ---------------------
                # Model Training and Validation Testing
                # ---------------------
                val_perfs = net.train_and_validate(self.param_set, train_dataloader, validation_dataloader)

                # Track validation performance for the set of hyperparameters used
                self.model_perf_list.append(val_perfs[-1])
                # Save the model if it is the best performing so far
                if grid_ind == np.nanargmin(self.model_perf_list):
                    net.save()

            # Print some results, determine whether to perform another layer of grid search, and if so, refine the mesh
            self.__next_hp_layer(net,tune_layer)

        # After grid search finishes, plot results
        if len(self.hyper_tuning_table) > 0 and self.plot_tuning_results:
            plot_grid_search_results(self.save_file,self.param_set,variables=('learning_rate','lmbda'))

        # Set the model back to the highest performing config
        net.model = net.load(net.save_file,print_loaded_model=False)


    # PRIVATE METHODS

    def __next_hp_layer(self,neural_net,tune_layer):
        min_grid_index = np.nanargmin(self.model_perf_list)
        if self.optimize_hypers:
            self._save_hp_tuning_results(addl_columns={'Grid Search Layer': tune_layer}, filename=self.save_file)
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
