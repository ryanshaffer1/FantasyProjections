"""Creates multiple Fantasy Football stat predictors (using various prediction algorithms), evaluates their predictions for a user-configurable
	set of past NFL games, and logs/visualizes the results in multiple user-configurable plots.
	Contains many user settings for the Neural Network Predictor, which is the most custom (and most important) type of predictor in the project.

	This module is a script to be run alone. Before running, the packages in requirements.txt must be installed. Use the following terminal command:
	> pip install -u requirements.txt
"""  # fmt: skip

import logging
import logging.config
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from config import data_files_config
from config.log_config import LOGGING_CONFIG
from gamblers import BasicGambler
from misc.dataset import StatsDataset
from misc.manage_files import move_logfile
from neural_net import HyperParameterSet
from parsers.parse_inputs import parse_datasets, parse_plots, parse_predictors, parse_tuners

# Read input file
with open("inputs/test_scenario_input.yaml") as stream:
    ipts = yaml.safe_load(stream)

# Parse inputs
# Output files/folders
if ipts.get("save_folder_timestamp"):
    save_folder_name = ipts.get("save_folder_prefix", "") + datetime.strftime(datetime.now().astimezone(), "%Y%m%d_%H%M%S")
else:
    save_folder_name = ipts.get("save_folder_prefix", "")
save_folder = os.path.join(ipts.get("file_save_directory", "/"), save_folder_name)
# Plotting - not currently used
save_plots = ipts.get("save_plots", True)

# Neural Net Data files
PBP_DATAFILE = data_files_config.PRE_PROCESS_FOLDER + data_files_config.NN_STAT_FILES["midgame"]
BOXSCORE_DATAFILE = data_files_config.PRE_PROCESS_FOLDER + data_files_config.NN_STAT_FILES["final"]
ID_DATAFILE = data_files_config.PRE_PROCESS_FOLDER + data_files_config.NN_STAT_FILES["id"]

# Set up logger
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")

# Start
logger.info("Starting Program")

# Read data files
pbp_df = pd.read_csv(PBP_DATAFILE, engine="pyarrow")
boxscore_df = pd.read_csv(BOXSCORE_DATAFILE, engine="pyarrow")
id_df = pd.read_csv(ID_DATAFILE, engine="pyarrow")
logger.info("Input files read")
for name, file in zip(["pbp", "boxscore", "IDs"], [PBP_DATAFILE, BOXSCORE_DATAFILE, ID_DATAFILE]):
    logger.debug(f"{name}: {file}")

# Training, validation, and test datasets
all_data = StatsDataset("All", id_df=id_df, pbp_df=pbp_df, boxscore_df=boxscore_df)
datasets = parse_datasets(ipts, all_data)
for dataset in datasets.values():
    logger.info(f"{dataset.name} Dataset size: {dataset.x_data.shape[0]}")


# Hyper-Parameters affecting the behavior of the Neural Net
param_set = HyperParameterSet(hp_dict=ipts.get("hyperparameters", {}))

# Initialize all predictors
predictors_dict = parse_predictors(ipts, save_folder)

# Initialize all hyper-parameter tuners
tuners_dict = parse_tuners(ipts, save_folder, param_set)

# Training/tuning processes
for tuning_process in ipts.get("tunings", {}):
    param_tuner = tuners_dict[tuning_process["hp_tuner"]]
    predictor = predictors_dict[tuning_process["predictor"]]
    predictor_funcs = {"train_and_validate": predictor.train_and_validate, "save": predictor.save, "load": predictor.load}
    eval_arguments = tuning_process.get("eval_arguments", {})
    param_tuner.tune_hyper_parameters(
        eval_function=predictor_funcs[tuning_process.get("eval_fn", "")],
        save_function=predictor_funcs[tuning_process.get("save_fn", "")],
        reset_function=predictor_funcs[tuning_process.get("reset_fn", "")],
        eval_kwargs={
            "training_data": datasets[eval_arguments.get("training_data")],
            "validation_data": datasets[eval_arguments.get("validation_data")],
        },
        reset_kwargs={"model_folder": save_folder},
        **tuning_process.get("kwargs", {}),
    )

for training_process in ipts.get("trainings", {}):
    predictor = predictors_dict[training_process["predictor"]]
    eval_arguments = training_process.get("eval_arguments", {})
    predictor.train_and_validate(
        training_data=datasets[eval_arguments.get("training_data")],
        validation_data=datasets[eval_arguments.get("validation_data")],
        param_set=param_set,
    )

# Evaluate Model(s) against Test Data
eval_results = {}
for evaluation in ipts.get("evaluations", {}):
    name = evaluation.get("name")
    predictor = predictors_dict[evaluation["predictor"]]
    eval_data = datasets[evaluation["dataset"]]
    prediction_result = predictor.eval_model(eval_data=eval_data)
    eval_results[name] = prediction_result

# Gamble based on prediction results, and evaluate success
type_map = {"BasicGambler": BasicGambler}
for gambler_input in ipts.get("gamblers", {}):
    name = gambler_input["name"]
    prediction_result = eval_results[gambler_input["evaluation"]]
    gambler = type_map[gambler_input["type"]](prediction_result)
    if gambler_input.get("plot_earnings", True):
        gambler.plot_earnings()
    logger.info(f"Earnings from {name}: {gambler.earnings: 0.2f} units ({gambler.accuracy * 100:0.2f}% accurate)")

# Plot evaluation results
parse_plots(ipts, eval_results)

# Move logfile to the correct folder
logging.shutdown()
if ipts.get("save_log", True):
    move_logfile("logfile.log", save_folder)

# Display plots
plt.show()
