"""Executes a simulation of Fantasy Football player stat prediction (including model training/tuning) and evaluates the results.

	Creates multiple Fantasy Football stat predictors (using various prediction algorithms), evaluates their predictions for a user-configurable
	set of past NFL games, and logs/visualizes the results in multiple user-configurable plots.
	Contains many user settings for the Neural Network Predictor, which is the most custom (and most important) type of predictor in the project.
"""  # fmt: skip

import argparse
import logging
import logging.config

import managers
from config.log_config import LOGGING_CONFIG
from misc.manage_files import move_logfile, save_plots
from neural_net import HyperParameterSet

# Read command line argument: input parameter file
parser = argparse.ArgumentParser("main")
parser.add_argument("parameter_file", help="YAML file containing all scenario parameters - see example input files.")
args = parser.parse_args()
input_params = managers.parse_inputs(args.parameter_file)

# Output folder
save_folder = input_params.save_options["save_directory"]

# Set up logger
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")

# Start
logger.info(f"Starting Program. Input parameters from file: {args.parameter_file}")

# Initialize a collector for all scenario objects
scenario = managers.ScenarioObjects()

# Read data from csv files and create all datasets
all_data = managers.read_data_into_dataset(input_params.features, log_datafiles=True)
scenario.datasets = managers.create_datasets(input_params.datasets, all_data)

# Update some scenario input parameters based on features/data
input_params.update_params_based_on_features(all_data)

# Hyper-Parameters affecting the behavior of the Neural Net
scenario.hyperparameters = HyperParameterSet(hp_dict=input_params.hyperparameters)

# Initialize all predictors
scenario.predictors = managers.create_predictors(input_params.predictors, save_folder)

# Initialize all hyper-parameter tuners
scenario.tuners = managers.create_tuners(input_params.tuners, save_folder, scenario.hyperparameters)

# Training/tuning processes
managers.perform_tunings(input_params.tunings, scenario, save_folder)
managers.perform_trainings(input_params.trainings, scenario)

# Evaluate Model(s) against Test Data
scenario.evaluations = managers.perform_evaluations(input_params.evaluations, scenario)

# Gamble based on prediction results, and evaluate success
managers.create_gamblers(input_params.gamblers, scenario.evaluations)

# Plot evaluation results
managers.create_plots(input_params.plot_groups, scenario.evaluations)

# Save plots
if input_params.save_options["save_plots"]:
    logger.info("Saving plots")
    save_plots(save_folder)


# End logging and move logfile to the correct folder
logger.info("Complete!")
logging.shutdown()
if input_params.save_options["save_log"]:
    move_logfile("logfile.log", save_folder)
