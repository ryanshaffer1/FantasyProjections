"""Creates and exports classes to be used as various approaches to predicting NFL stats and Fantasy Football scores.

    Classes:
        PerfectPredictor : child of FantasyPredictor. Predicts NFL player stats using the true NFL stats, giving perfect predictions.
"""  # fmt:skip

from dataclasses import dataclass

from predictors import FantasyPredictor


@dataclass
class PerfectPredictor(FantasyPredictor):
    """Predictor of NFL players' stats in games, using the true NFL stats, giving perfect predictions.

        Sub-class of FantasyPredictor.

        Args:
            name (str): name of the predictor object, used for logging/display purposes.

        Public Methods:
            eval_model : Generates predicted stats for an input evaluation dataset, using the true stats for the same dataset.

    """  # fmt:skip

    # CONSTRUCTOR
    # N/A - Fully constructed by parent __init__()

    # PUBLIC METHODS

    def eval_model(self, eval_data, **kwargs):
        """Generates predicted stats for an input evaluation dataset, using the true stats for the same dataset.

            Args:
                eval_data (StatsDataset): data to use for Predictor evaluation (e.g. validation or test data).
                kwargs:
                    All keyword arguments are passed to the function stats_to_fantasy_points and to the PredictionResult constructor.
                    See the related documentation for descriptions and valid inputs. All keyword arguments are optional.

            Returns:
                PredictionResult: Object packaging the predicted and true stats together, which can be used for plotting,
                    performance assessments, etc.

        """  # fmt:skip

        # True stats from eval data
        stat_truths = self.eval_truth(eval_data, **kwargs)
        # Predicts equal truth
        stat_predicts = stat_truths

        # Create result object
        result = self._gen_prediction_result(stat_predicts, stat_truths, eval_data, **kwargs)

        return result
