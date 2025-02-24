from predictors import LastNPredictor, NeuralNetPredictor, PerfectPredictor, SleeperPredictor
from results import PredictionResult, PredictionResultGroup
from tuners import GridSearchTuner, RandomSearchTuner, RecursiveRandomSearchTuner


def parse_datasets(ipts, all_data):
    datasets = {}
    for dataset_ipt in ipts.get("datasets", {}):
        name = dataset_ipt.get("name", "")
        for i, configuration in enumerate(dataset_ipt.get("config", {})):
            if i == 0:
                dataset = all_data.slice_by_criteria(inplace=False, **configuration)
            else:
                dataset.concat(all_data.slice_by_criteria(inplace=False, **configuration))
        dataset.name = name
        datasets[name] = dataset

    return datasets


def parse_predictors(ipts, save_folder):
    # Valid types that may be entered as strings and used as FantasyPredictor classes
    type_map = {
        "NeuralNetPredictor": NeuralNetPredictor,
        "SleeperPredictor": SleeperPredictor,
        "LastNPredictor": LastNPredictor,
        "PerfectPredictor": PerfectPredictor,
    }

    # Initialize dict of predictors
    predictors_dict = {}

    # Read input predictors one-by-one
    for predictor_ipts in ipts.get("predictors"):
        name = predictor_ipts.get("name", "")

        # Get all config settings for the predictor from the inputs
        config = predictor_ipts.get("config", {})

        # Special case for a Neural Net setting, should change the NeuralNetPredictor constructor to make this OBE
        if predictor_ipts.get("type") == "NeuralNetPredictor":
            config["save_folder"] = save_folder if predictor_ipts.get("save_model", False) else None

        # Create the FantasyPredictor of the correct type with the config settings input
        predictor = type_map[predictor_ipts.get("type")](name=name, **config)

        # Add to list of predictors
        predictors_dict[name] = predictor

    return predictors_dict


def parse_tuners(ipts, save_folder, param_set):
    # Valid types that may be entered as strings and used as FantasyPredictor classes
    type_map = {
        "GridSearchTuner": GridSearchTuner,
        "RandomSearchTuner": RandomSearchTuner,
        "RecursiveRandomSearchTuner": RecursiveRandomSearchTuner,
    }

    # Initialize dict of tuners
    tuners_dict = {}

    # Read input tuners one-by-one
    for tuner_ipts in ipts.get("tuners"):
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


def parse_plots(ipts, eval_results):
    type_map = {
        "error_dist": PredictionResult.plot_error_dist,
        "single_games": PredictionResult.plot_single_games,
        "scatters": PredictionResult.plot_scatters,
    }
    for plot_group in ipts.get("plot_groups", []):
        results = [eval_results[name] for name in plot_group.get("results", [])]
        result_group = PredictionResultGroup(results)
        for plot in plot_group.get("plots", []):
            settings = plot.get("settings", {})
            result_group.plot_all(type_map[plot.get("type")], **settings)
