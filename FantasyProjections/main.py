"""Creates multiple Fantasy Football stat predictors (using various prediction algorithms), evaluates their predictions for a user-configurable
	set of past NFL games, and logs/visualizes the results in multiple user-configurable plots.
	Contains many user settings for the Neural Network Predictor, which is the most custom (and most important) type of predictor in the project.

	This module is a script to be run alone. Before running, the packages in requirements.txt must be installed. Use the following terminal command:
	> pip install -u requirements.txt
"""  # fmt: skip

import logging
import logging.config

import matplotlib.pyplot as plt
from config.log_config import LOGGING_CONFIG
from managers.create_objs import create_datasets, create_gamblers, create_plots, create_predictors, create_tuners
from managers.parse_inputs import parse_inputs
from managers.perform_actions import perform_evaluations, perform_trainings, perform_tunings
from managers.read_data import read_data_into_dataset
from managers.scenario_objects import ScenarioObjects
from misc.manage_files import move_logfile, name_save_folder
from neural_net import HyperParameterSet

# Read input file into parameters
input_filename = "inputs/test_scenario_input.yaml"
input_params = parse_inputs(input_filename)

# Output folder/options
save_folder = name_save_folder(input_params.save_options)
save_plots = input_params.save_options["save_plots"]  # Not currently being used

# Set up logger
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")

# Start
logger.info("Starting Program")

# Initialize a collector for all scenario objects
scenario = ScenarioObjects()

# Read data from csv files and create all datasets
all_data = read_data_into_dataset(log_datafiles=True)
scenario.datasets = create_datasets(input_params.datasets, all_data)

# Hyper-Parameters affecting the behavior of the Neural Net
scenario.hyperparameters = HyperParameterSet(hp_dict=input_params.hyperparameters)

# Initialize all predictors
scenario.predictors = create_predictors(input_params.predictors, save_folder)

# Initialize all hyper-parameter tuners
scenario.tuners = create_tuners(input_params.tuners, save_folder, scenario.hyperparameters)

# Training/tuning processes
perform_tunings(input_params.tunings, scenario, save_folder)
perform_trainings(input_params.trainings, scenario)

# Evaluate Model(s) against Test Data
scenario.evaluations = perform_evaluations(input_params.evaluations, scenario)

# Gamble based on prediction results, and evaluate success
create_gamblers(input_params.gamblers, scenario.evaluations)

# Plot evaluation results
create_plots(input_params.plot_groups, scenario.evaluations)

# Move logfile to the correct folder
logging.shutdown()
if input_params.save_options["save_log"]:
    move_logfile("logfile.log", save_folder)

# Display plots
plt.show()
