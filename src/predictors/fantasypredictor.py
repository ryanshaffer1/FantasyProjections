import numpy as np
from misc.nn_helper_functions import stats_to_fantasy_points
from misc.prediction_result import PredictionResult

class FantasyPredictor():
    def __init__(self,name):
        # # Assign training data, validation data, and test data
        # # If fewer than three datasets are in the list "datasets", the variables within
        # # "datasets" will be assigned to training_data, validation_data, then test_data in
        # # that order. All unassigned variables will be assigned None.
        # (self.training_data,self.validation_data,self.test_data) = datasets + type(datasets)([None])*(3-len(datasets))
        self.name = name

    def gen_prediction_result(self, stat_predicts, stat_truths, eval_data):
        # Generate PredictorResult object
        result = PredictionResult(self, stat_predicts, stat_truths, eval_data)

        # Compute/display average absolute error
        print(f"{self.name} Test Error: Avg. Abs. Fantasy Points Different = {(np.mean(result.avg_diff(absolute=True))):>0.2f}")

        return result

    def eval_truth(self, eval_data):
        stat_truths = stats_to_fantasy_points(
            eval_data.y_data, stat_indices='default', normalized=True)
        return stat_truths
