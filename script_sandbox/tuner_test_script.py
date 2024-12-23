# Make source code importable from sandbox folder
import sys
import os
sys.path.append(os.getcwd()+'/FantasyProjections')

import random
import numpy as np
import matplotlib.pyplot as plt
from neural_net import HyperParameterSet
from tuners import GridSearchTuner, RandomSearchTuner
from plot_cost_vs_computation import plot_cost_vs_computation
from sandbox_helpers import get_min_by_layer, add_artificial_layers

SAVE_FOLDER = 'script_sandbox/sandbox_files/'
SAVE = False
NUM_RUNS = 10

# Hyper-Parameters
num_hps = 6
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
    'hyper_tuner_layers': 3,
    'hyper_tuner_steps_per_dim': 3,
}
grid_tuner_settings2 = {
    'hyper_tuner_layers': 5,
    'hyper_tuner_steps_per_dim': 2,
}
random_tuner_settings = {
    'population_size': grid_tuner_settings1['hyper_tuner_layers']*grid_tuner_settings1['hyper_tuner_steps_per_dim']**num_hps
}

hp_dict = {name:linear_hp for name in hp_names[:num_hps]}

hp_set = HyperParameterSet(hp_dict=hp_dict)

def spherical(param_set, n_dimensions=len(hp_dict), root=(0,0)):
    # It's spherical! SPHERICAL!
    val = 0
    for i, hp_name in zip(range(n_dimensions),hp_dict):
        hp_value = param_set.get(hp_name).value
        val += (hp_value - root[i])**2

    return val

def rosenbrock(param_set, n_dimensions=len(hp_dict), root=(0,0)):
    val = 0
    for i in range(n_dimensions-1):
        x_i = param_set.hyper_parameters[i].value
        x_ip1 = param_set.hyper_parameters[i+1].value
        
        val += 100*(x_ip1 - x_i**2)**2 + (1-x_i)**2

    return val

eval_function = spherical
func_name = 'Spherical'
tuner_min_vals = [[],[],[]]
for _ in range(NUM_RUNS):
    zero_point = tuple(random.uniform(*hp_set.get(hp).val_range) for hp in hp_dict)
    print(f'Root={zero_point}')

    # Tuning algorithm for Neural Net Hyper-Parameters
    if SAVE:
        grid_tuner = GridSearchTuner(hp_set, save_file=SAVE_FOLDER+'grid.csv', **all_hp_tuner_settings, **grid_tuner_settings1)
        grid_tuner2 = GridSearchTuner(hp_set, save_file=SAVE_FOLDER+'grid2.csv', **all_hp_tuner_settings, **grid_tuner_settings2)
        random_tuner = RandomSearchTuner(hp_set, save_file=SAVE_FOLDER+'rand.csv', **all_hp_tuner_settings, **random_tuner_settings)
    else:
        grid_tuner = GridSearchTuner(hp_set, **all_hp_tuner_settings, **grid_tuner_settings1)
        grid_tuner2 = GridSearchTuner(hp_set, **all_hp_tuner_settings, **grid_tuner_settings2)
        random_tuner = RandomSearchTuner(hp_set, **all_hp_tuner_settings, **random_tuner_settings)

    tuners = [grid_tuner, grid_tuner2, random_tuner]
    tuner_names = ['Grid Search (m=3)','Grid Search (m=2)','Random Search']

    for ind, (tuner, name) in enumerate(zip(tuners, tuner_names)):
        tuner.tune_hyper_parameters(eval_function=eval_function, eval_kwargs={'root':zero_point},
                                                plot_variables=all_hp_tuner_settings['plot_variables'])
        if isinstance(tuner, RandomSearchTuner):
            add_artificial_layers(tuner,layer_size=grid_tuner.total_combinations)
        mins = get_min_by_layer(tuner, num_hps)
        tuner_min_vals[ind].append(mins)

for ind, (tuner, name) in enumerate(zip(tuners, tuner_names)):
    tuner.min_vals_by_group = np.nanmean(np.array(tuner_min_vals[ind]),axis=0)
    print(f'{name} min(s): {tuner.min_vals_by_group}, n={len(tuner.hyper_tuning_table)}')

plot_cost_vs_computation(tuners, tuner_names, NUM_RUNS, func_name, num_hps)

plt.show()
