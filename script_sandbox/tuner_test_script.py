# Make source code importable from sandbox folder
import sys
import os
sys.path.append(os.getcwd()+'/FantasyProjections')

import random
import numpy as np
import matplotlib.pyplot as plt
from neural_net import HyperParameterSet
from tuners import GridSearchTuner, RandomSearchTuner, RecursiveRandomSearchTuner
from plot_cost_vs_computation import plot_cost_vs_computation, plot_continuous_cost
from sandbox_helpers import get_min_by_layer, add_artificial_layers
import optimization_test_functions as opt_fnc

SAVE_FOLDER = 'script_sandbox/sandbox_files/'
SAVE = False
NUM_RUNS = 10

# Hyper-Parameters
NUM_HPS = 8
hp_names = ['a','b','c','d','e','f','g','h','i','j']
linear_hp = {
    'value': 0,
    'optimizable': True,
    'val_range': [-10,10],
    'val_scale': 'linear'}

# Hyper-Parameter Tuning Settings
all_hp_tuner_settings = {
    'optimize_hypers': True,
    'plot_tuning_results': False,
    'plot_variables': ('a','b')
}
grid_tuner_settings1 = {
    'hyper_tuner_layers': 2,
    'hyper_tuner_steps_per_dim': 3,
}
total_evals = grid_tuner_settings1['hyper_tuner_layers']*grid_tuner_settings1['hyper_tuner_steps_per_dim']**NUM_HPS
grid_tuner_settings2 = {
    'hyper_tuner_layers': int(total_evals/(2**NUM_HPS)),
    'hyper_tuner_steps_per_dim': 2,
}
random_tuner_settings = {
    'n_value_combinations': grid_tuner_settings1['hyper_tuner_layers']*grid_tuner_settings1['hyper_tuner_steps_per_dim']**NUM_HPS
}
rec_random_tuner_settings = {
    'max_samples': random_tuner_settings['n_value_combinations'],
    'r_percentile': 0.1,
    'v_expect_imp': 0.1,
    's_shrink_thresh': 0
}

hp_dict = {name:linear_hp for name in hp_names[:NUM_HPS]}

hp_set = HyperParameterSet(hp_dict=hp_dict)

eval_functions = [opt_fnc.spherical, opt_fnc.rastrigin, opt_fnc. rosenbrock]
func_names = ['Spherical', 'Rastrigin', 'Rosenbrock']


for eval_function, func_name in zip(eval_functions, func_names):
    print(func_name)
    tuner_min_vals = [[],[],[],[]]
    for _ in range(NUM_RUNS):
        zero_point = tuple(random.uniform(*hp_set.get(hp).val_range) for hp in hp_dict)
        print(f'Root={zero_point}')

        # Tuning algorithm for Neural Net Hyper-Parameters
        if SAVE:
            grid_tuner = GridSearchTuner(hp_set, save_file=SAVE_FOLDER+'grid.csv', **all_hp_tuner_settings, **grid_tuner_settings1)
            grid_tuner2 = GridSearchTuner(hp_set, save_file=SAVE_FOLDER+'grid2.csv', **all_hp_tuner_settings, **grid_tuner_settings2)
            random_tuner = RandomSearchTuner(hp_set, save_file=SAVE_FOLDER+'rand.csv', **all_hp_tuner_settings, **random_tuner_settings)
            rec_random_tuner = RecursiveRandomSearchTuner(hp_set, save_file=SAVE_FOLDER+'rec_rand.csv', **all_hp_tuner_settings, **rec_random_tuner_settings)
        else:
            grid_tuner = GridSearchTuner(hp_set, **all_hp_tuner_settings, **grid_tuner_settings1)
            grid_tuner2 = GridSearchTuner(hp_set, **all_hp_tuner_settings, **grid_tuner_settings2)
            random_tuner = RandomSearchTuner(hp_set, **all_hp_tuner_settings, **random_tuner_settings)
            rec_random_tuner = RecursiveRandomSearchTuner(hp_set, **all_hp_tuner_settings, **rec_random_tuner_settings)

        tuners = [grid_tuner, grid_tuner2, random_tuner, rec_random_tuner]
        tuner_names = ['Grid Search (Breadth)','Grid Search (Depth)','Random Search', 'Recursive Random Search']

        for ind, (tuner, name) in enumerate(zip(tuners, tuner_names)):
            tuner.tune_hyper_parameters(eval_function=eval_function, eval_kwargs={'root':zero_point, 'n_dimensions': len(hp_dict)},
                                                    plot_variables=all_hp_tuner_settings['plot_variables'])
            if isinstance(tuner, RandomSearchTuner) or isinstance(tuner, RecursiveRandomSearchTuner):
                add_artificial_layers(tuner,layer_size=grid_tuner.n_value_combinations)
            mins = get_min_by_layer(tuner, NUM_HPS)
            tuner_min_vals[ind].append(mins)

    for ind, (tuner, name) in enumerate(zip(tuners, tuner_names)):
        tuner.min_vals_by_group = np.nanmean(np.array(tuner_min_vals[ind]),axis=0)
        print(f'{name} min(s): {tuner.min_vals_by_group}, n={len(tuner.hyper_tuning_table)}')

    # plot_cost_vs_computation(tuners, tuner_names, NUM_RUNS, func_name, num_hps)
    plot_continuous_cost(tuners, tuner_names, NUM_RUNS, func_name, NUM_HPS)

plt.show()
