"""Creates and exports a base class that supports predicting NFL stats and Fantasy Football scores.
    This class cannot generate predictions on its own, but has multiple sub-classes (children) that 
    make predictions according to various algorithms.

    Classes:
        FantasyPredictor : Base class of Predictor objects that predict NFL player stats based on some algorithm.
"""

from dataclasses import dataclass
import logging
import numpy as np
from results import PredictionResult
from misc.stat_utils import stats_to_fantasy_points

# Set up logger
logger = logging.getLogger('log')

@dataclass
class FantasyPredictor():
    """Base class of Predictor objects that predict NFL player stats based on some algorithm.
    
        Sub-classes implement specific prediction algorithms, including:
            NeuralNetPredictor: predictions using a trained Neural Net
            SleeperPredictor: predictions obtained from the Sleeper API
            LastNPredictor: predictions based on an average of a player's recent stats
            PerfectPredictor: predictions that match reality (using real stats as an input)
    
        Args:
            name (str): name of the predictor object, used for logging/display purposes.

        Public Methods:
            eval_truth : Evaluates the true NFL players' stats based on a provided dataset

        Methods Available to Sub-Classes:
            _gen_prediction_result : Packages predicted stats and true stats into a PredictionResult object and logs accuracy
    """

    name: str = ''

    # PUBLIC METHODS

    def eval_truth(self, eval_data, **kwargs):
        """Evaluates the true NFL players' stats based on a provided dataset.

            Args:
                eval_data (StatsDataset): Dataset containing NFL players' final stats across multiple games.
                    Only the Dataset attribute .y_data is accessed.

            Keyword-Args:
                All keyword arguments are passed to stats_to_fantasy_points. See that function's documentation for descriptions and valid inputs.
                Keyword arguments are optional, with defaults set in stats_to_fantasy_points, EXCEPT:
                - normalized defaults to True in this implementation.

            Returns:
                pandas.DataFrame: Contains columns for each statistic (e.g. Pass Yds) as well as a column for Fantasy Points.
                    Each row corresponds to the final results for a single player/game.
        """
        # Override any necessary keyword argument values
        kwargs['normalized'] = kwargs.get('normalized',True)

        stat_truths = stats_to_fantasy_points(eval_data.y_df, **kwargs)
        return stat_truths


    # PROTECTED METHODS

    def _gen_prediction_result(self, stat_predicts, stat_truths, eval_data, **kwargs):
        """Packages predicted stats and true stats into a PredictionResult object and logs accuracy.

            Args:
                stat_predicts (pandas.DataFrame): Contains the final statistics (including Fantasy Points) 
                    for a set of players/games, as predicted by the Predictor object.
                stat_truths (pandas.DataFrame): Contains the true, final statistics (including Fantasy Points) 
                    for a set of players/games. The set of players/games must match the set in stat_predicts.
                eval_data (StatsDataset): Dataset containing NFL players' final stats across multiple games.

            Keyword-Args:
                All keyword arguments are passed to the PredictionResult constructor. See that class's documentation for descriptions and valid inputs.
                All keyword arguments are optional.

            Returns:
                PredictionResult: Object packaging the predicted and true stats together, which can be used for plotting, 
                    performance assessments, etc.
        """

        # Generate PredictorResult object
        result = PredictionResult(dataset=eval_data, predicts=stat_predicts, truths=stat_truths, predictor_name=self.name, **kwargs)

        # Compute/display average absolute error
        logger.info(f'{self.name} {eval_data.name} Error: Avg. Abs. Fantasy Points Different = {(np.mean(result.diff_pred_vs_truth(absolute=True))):>0.2f}')

        return result
