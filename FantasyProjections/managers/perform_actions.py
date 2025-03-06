def perform_tunings(tuning_params, scenario, save_folder):
    # Unpack the inputs needed from ScenarioObjects collector
    tuners = scenario.tuners
    predictors = scenario.predictors

    # Perform hyper-parameters tunings one-by-one
    for tuning_process in tuning_params:
        # Get the tuner associated with this tuning process
        param_tuner = tuners[tuning_process["hp_tuner"]]
        # Get the predictor associated with this tuning process
        predictor = predictors[tuning_process["predictor"]]

        # Function names that may be entered as strings and used as predictor methods
        predictor_funcs = {"train_and_validate": predictor.train_and_validate, "save": predictor.save, "load": predictor.load}
        # Replace names of functions with actual handles for eval, save, and reset functions
        functions = {k: predictor_funcs[tuning_process[k]] for k in ["eval_function", "save_function", "reset_function"]}

        # Evaluation function arguments
        eval_args = tuning_process.get("eval_arguments", {})
        # Create dict mapping evaluation keywords to associated scenario objects (like datasets), where applicable
        eval_args = {k: scenario.get_obj_by_name(v) for k, v in eval_args.items()}

        # Save function arguments
        save_args = tuning_process.get("save_arguments", {})

        # Reset function arguments
        reset_args = tuning_process.get("reset_arguments", {})
        # Replace string references to variables with the variable values themselves
        var_map = {"save_folder": save_folder}
        reset_args = {k: var_map.get(v, v) for k, v in reset_args.items()}

        # Perform hyper-parameter tuning
        param_tuner.tune_hyper_parameters(
            **functions,
            eval_kwargs=eval_args,
            save_kwargs=save_args,
            reset_kwargs=reset_args,
            **tuning_process.get("kwargs", {}),
        )


def perform_trainings(training_params, scenario):
    # Unpack the inputs needed from ScenarioObjects collector
    predictors = scenario.predictors
    hyperparameters = scenario.hyperparameters

    # Perform trainings one-by-one
    for training_process in training_params:
        # Get the predictor associated with this training process
        predictor = predictors[training_process["predictor"]]

        # Evaluation function arguments
        eval_args = training_process.get("eval_arguments", {})
        # Create dict mapping evaluation keywords to associated scenario objects (like datasets), where applicable
        eval_args = {k: scenario.get_obj_by_name(v) for k, v in eval_args.items()}

        # Perform training process
        predictor.train_and_validate(
            **eval_args,
            param_set=hyperparameters,
        )


def perform_evaluations(evaluation_params, scenario):
    # Unpack the inputs needed from ScenarioObjects collector
    predictors = scenario.predictors
    datasets = scenario.datasets

    eval_results = {}
    for evaluation in evaluation_params:
        name = evaluation.get("name")
        predictor = predictors[evaluation["predictor"]]
        eval_data = datasets[evaluation["dataset"]]
        prediction_result = predictor.eval_model(eval_data=eval_data)
        eval_results[name] = prediction_result

    return eval_results
