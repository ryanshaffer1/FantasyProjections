import yaml


def parse_inputs(input_filename):
    with open(input_filename) as stream:
        ipts = yaml.safe_load(stream)

    return InputParameters(ipts)


class InputParameters:
    def __init__(self, input_dict):
        self.save_options = input_dict.get("save_options", {})
        self.datasets = input_dict.get("datasets", {})
        self.hyperparameters = input_dict.get("hyperparameters", {})
        self.predictors = input_dict.get("predictors", {})
        self.tuners = input_dict.get("tuners", {})
        self.tunings = input_dict.get("tunings", {})
        self.trainings = input_dict.get("trainings", {})
        self.evaluations = input_dict.get("evaluations", {})
        self.gamblers = input_dict.get("gamblers", {})
        self.plot_groups = input_dict.get("plot_groups", {})
