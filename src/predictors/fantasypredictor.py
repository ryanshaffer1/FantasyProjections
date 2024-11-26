from dataclasses import dataclass
import logging
import numpy as np
from misc.nn_helper_functions import stats_to_fantasy_points
from misc.prediction_result import PredictionResult

# Set up logger
logger = logging.getLogger('log')

@dataclass
class FantasyPredictor():
    name: str = ''

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
        logger.info(f'{self.name} {eval_data.name} Error: Avg. Abs. Fantasy Points Different = {(np.mean(result.avg_diff(absolute=True))):>0.2f}')

        return result
