"""Set of functions used to build/initialize various objects used in the prediction of Fantasy Football stats.

    Functions:
        create_datasets : Generates datasets using a subset of all available data, based on the parameters specified for each dataset.
        create_predictors : Generates all FantasyPredictor objects used in the scenario, based on input parameters.
        create_tuners : Generates all HyperParamTuner objects used in the scenario, based on input parameters.
        create_gamblers : Generates all Gambler objects used in the scenario, based on input parameters.
        create_plots : Generates all plots of evaluation results, based on input parameters.
"""  # fmt:skip

import logging

from gamblers import BasicGambler
from predictors import LastNPredictor, NeuralNetPredictor, PerfectPredictor, SleeperPredictor
from results import PredictionResult, PredictionResultGroup
from tuners import GridSearchTuner, RandomSearchTuner, RecursiveRandomSearchTuner

# Set up logger
logger = logging.getLogger("log")


def create_datasets(dataset_params, all_data):
    """Generates datasets using a subset of all available data, based on the parameters specified for each dataset.

        Args:
            dataset_params (list[dict]): List where each element specifies the StatsDataset to create, including name and configuration parameters (e.g. criteria to slice data on).
            all_data (StatsDataset): StatsDataset containing all available data points.

        Returns:
            dict: Maps name of each dataset to the corresponding StatsDataset object.

    """  # fmt:skip

    # Initialize dict of datasets
    datasets = {}

    # Create datasets one-by-one
    for dataset_ipt in dataset_params:
        name = dataset_ipt.get("name", "")
        for i, configuration in enumerate(dataset_ipt.get("config", {})):
            if i == 0:
                dataset = all_data.slice_by_criteria(inplace=False, **configuration)
            else:
                dataset.concat(all_data.slice_by_criteria(inplace=False, **configuration))
        dataset.name = name
        datasets[name] = dataset

    for dataset in datasets.values():
        logger.info(f"{dataset.name} Dataset size: {dataset.x_data.shape[0]}")

    return datasets


def create_predictors(predictor_params, save_folder):
    """Generates all FantasyPredictor objects used in the scenario, based on input parameters.

        Args:
            predictor_params (list[dict]): List where each element specifies the FantasyPredictor to create, including name, type, and configuration parameters.
            save_folder (str): Directory to save any models which are updated over the course of the scenario (such as NeuralNetPredictors).

        Returns:
            dict: Maps name of each predictor to the corresponding FantasyPredictor object.

    """  # fmt:skip

    # Valid types that may be entered as strings and used as FantasyPredictor classes
    type_map = {
        "NeuralNetPredictor": NeuralNetPredictor,
        "SleeperPredictor": SleeperPredictor,
        "LastNPredictor": LastNPredictor,
        "PerfectPredictor": PerfectPredictor,
    }

    # Initialize dict of predictors
    predictors_dict = {}

    # Create predictors one-by-one
    for predictor_ipts in predictor_params:
        name = predictor_ipts.get("name", "")

        # Get all config settings for the predictor from the inputs
        config = predictor_ipts.get("config", {})

        # Special case for a Neural Net setting. TODO: should change the NeuralNetPredictor constructor to make this OBE
        if predictor_ipts.get("type") == "NeuralNetPredictor":
            config["save_folder"] = save_folder if predictor_ipts.get("save_model", False) else None

        # Create the FantasyPredictor of the correct type with the config settings input
        predictor = type_map[predictor_ipts.get("type")](name=name, **config)

        # Add to list of predictors
        predictors_dict[name] = predictor

    return predictors_dict


def create_tuners(tuner_params, save_folder, param_set):
    """Generates all HyperParamTuner objects used in the scenario, based on input parameters.

        Args:
            tuner_params (list[dict]): List where each element specifies the HyperParamTuner to create, including name, type, and settings.
            save_folder (str): Directory to save any tuning results or models which are updated over the course of the scenario.
            param_set (HyperParameterSet): Hyper-parameters controllable by each tuner.

        Returns:
            dict: Maps name of each tuner to the corresponding HyperParamTuner object.

    """  # fmt:skip

    # Valid types that may be entered as strings and used as FantasyPredictor classes
    type_map = {
        "GridSearchTuner": GridSearchTuner,
        "RandomSearchTuner": RandomSearchTuner,
        "RecursiveRandomSearchTuner": RecursiveRandomSearchTuner,
    }

    # Initialize dict of tuners
    tuners_dict = {}

    # Create tuners one-by-one
    for tuner_ipts in tuner_params:
        name = tuner_ipts.get("name", "")

        # Get all settings for the predictor from the inputs
        settings = tuner_ipts.get("settings", {})

        # Modify save file
        if settings.get("save_file"):
            settings["save_file"] = save_folder + settings["save_file"]

        # Create the FantasyPredictor of the correct type with the config settings input
        tuner = type_map[tuner_ipts.get("type")](param_set, **settings)

        # Add to list of predictors
        tuners_dict[name] = tuner

    return tuners_dict


def create_gamblers(gambler_params, eval_results):
    """Generates all Gambler objects used in the scenario, based on input parameters.

        Args:
            gambler_params (list[dict]): List where each element specifies the Gambler to create, including name, type, and settings.
            eval_results (dict): Maps names of prediction evaluations to PredictionResult objects (containing the stats predictions over a dataset).

    """  # fmt:skip

    # Valid types that may be entered as strings and used as Gambler classes
    type_map = {"BasicGambler": BasicGambler}

    # Create tuners one-by-one
    for gambler_input in gambler_params:
        name = gambler_input["name"]

        # Find the evaluation results corresponding to this gambler
        prediction_result = eval_results[gambler_input["evaluation"]]

        # Create the Gambler of the correct type with the config settings input
        gambler = type_map[gambler_input["type"]](prediction_result)

        # Optionally plot outputs
        if gambler_input.get("plot_earnings", True):
            gambler.plot_earnings()

        # Log performance
        logger.info(f"Earnings from {name}: {gambler.earnings: 0.2f} units ({gambler.accuracy * 100:0.2f}% accurate)")


def create_plots(plot_group_params, eval_results):
    """Generates all plots of evaluation results, based on input parameters.

        Args:
            plot_group_params (list[dict]): List where each element specifies the plot group to create, including PredictionResults to include and plot types/settings.
            eval_results (dict): Maps names of prediction evaluations to PredictionResult objects (containing the stats predictions over a dataset).

    """  # fmt:skip

    # Valid options that may be entered as strings and used as plot functions
    type_map = {
        "error_dist": PredictionResult.plot_error_dist,
        "single_games": PredictionResult.plot_single_games,
        "scatters": PredictionResult.plot_scatters,
    }

    # Create plot groups one-by-one
    for plot_group in plot_group_params:
        # Generate PredictionResultGroup for the evaluation results associated with this plot group
        results = [eval_results[name] for name in plot_group.get("results", [])]
        result_group = PredictionResultGroup(results)

        # Generate plots one-by-one
        for plot in plot_group.get("plots", []):
            # Create plots with the input settings
            settings = plot.get("settings", {})
            result_group.plot_all(type_map[plot.get("type")], **settings)
