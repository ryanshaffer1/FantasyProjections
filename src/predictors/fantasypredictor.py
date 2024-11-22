from dataclasses import dataclass
import numpy as np
from misc.nn_helper_functions import stats_to_fantasy_points
from misc.prediction_result import PredictionResult

@dataclass
class FantasyPredictor():
    name: str = ''

# class FantasyPredictor():
#     # CONSTRUCTOR

#     def __init__(self,name):
#         self.name = name


    # PUBLIC METHODS

    def eval_truth(self, eval_data):
        stat_truths = stats_to_fantasy_points(
            eval_data.y_data, stat_indices='default', normalized=True)
        return stat_truths


    # PROTECTED METHODS
    def _gen_prediction_result(self, stat_predicts, stat_truths, eval_data):
        # Generate PredictorResult object
        result = PredictionResult(self, stat_predicts, stat_truths, eval_data)

        # Compute/display average absolute error
        print(f"{self.name} Test Error: Avg. Abs. Fantasy Points Different = {(np.mean(result.avg_diff(absolute=True))):>0.2f}")

        return result
