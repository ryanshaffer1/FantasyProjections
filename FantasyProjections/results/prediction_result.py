"""Creates and exports classes to be used in processing the results of Fantasy Football score/stat predictors.

    Classes:
        PredictionResult : Class containing a set of NFL games/players being evaluated, a prediction of their stats, and their true stats.
"""

from config import stats_config
from misc.stat_utils import stats_to_fantasy_points, gen_random_games
from .result_plots import gen_scatterplots, plot_game_timeline, plot_error_histogram

class PredictionResult():
    """Class containing a set of NFL games/players being evaluated, a prediction of their stats, and their true stats.
        
        Attributes:
            dataset (StatsDataset): set of NFL games/players evaluated.
            predicts (pandas.DataFrame): stat lines (with Fantasy Points) predicted for the NFL games/players in dataset
            truths (pandas.DataFrame): true stat lines (with Fantasy Points) scored in the dataset entries
            predictor_name (str): name of the predictor which originated the predicted stats. Used for logging/displaying results
            id_df (pandas.DataFrame): IDs of the players/games from dataset
            pbp_df (pandas.DataFrame): DataFrame containing mid-game stats (with Fantasy Points) for all players/games in the dataset
        
        Public Methods:
            diff_pred_vs_truth : Calculates the average (signed or absolute) difference between predicted and true Fantasy Points in dataset
            plot_error_dist : Generates histogram of (signed or absolute) differences between predicted and true Fantasy Points in dataset
            plot_scatters : Generates scatterplots comparing predicted and true stats, optionally with subsections ("slices) of the dataset
            plot_single_games : Generates a line graph of predicted and true Fantasy Points over the course of a game for a single player (or multiple players)
    """

    # CONSTRUCTOR
    def __init__(self, dataset, predicts, truths, predictor_name=None, **kwargs):
        """Constructor for PredictionResult

            Args:
                dataset (StatsDataset): set of NFL games/players evaluated.
                predicts (pandas.DataFrame): stat lines (with Fantasy Points) predicted for the NFL games/players in dataset
                truths (pandas.DataFrame): true stat lines (with Fantasy Points) scored in the dataset entries
                predictor_name (str): name of the predictor which originated the predicted stats. Used for logging/displaying results
            
            Keyword-Args:
                All keyword arguments are passed to stats_to_fantasy_points. See that function's documentation for descriptions and valid inputs.
                Keyword arguments are optional, with defaults set in stats_to_fantasy_points, EXCEPT:
                - normalized defaults to True in this implementation.

            Additional Class Attributes (generated, not passed as inputs):
                id_df (pandas.DataFrame): IDs of the players/games from dataset
                pbp_df (pandas.DataFrame): DataFrame containing mid-game stats (with Fantasy Points) for all players/games in the dataset
        """
        # Optional input
        if not predictor_name:
            predictor_name = 'Unknown Predictor'

        # Assign inputs to class attributes
        self.predicts = predicts
        self.truths = truths
        self.dataset = dataset
        self.predictor_name = predictor_name

        # ID (player, week, year, team, position, etc) for each data point
        self.id_df = self.dataset.id_data.reset_index(drop=True)
        # Play-by-play data with fantasy score (used in plot_single_game)
        self.pbp_df = self.__pbp_with_fantasy_points(**kwargs)

    # PUBLIC METHODS

    def diff_pred_vs_truth(self,absolute=False):
        """Calculates the average (signed or absolute) difference between predicted and true Fantasy Points in dataset.

            Args:
                absolute (bool, optional): whether to compute absolute value of error before computing the average. Defaults to False.

            Returns:
                list: list of average Fantasy Point differences (between predict and truth) for all games/players in PredictorResult's dataset
        """

        # Calculate average difference in fantasy points between prediction and truth
        fp_predicts = self.predicts['Fantasy Points']
        fp_truths = self.truths['Fantasy Points']
        if absolute:
            fp_diffs = [abs(predict - truth)
                    for predict, truth in zip(fp_predicts, fp_truths)]
        else:
            fp_diffs = [predict - truth
                    for predict, truth in zip(fp_predicts, fp_truths)]
        return fp_diffs


    def plot_error_dist(self, absolute=False, fig=None):
        """Generates histogram of (signed or absolute) differences between predicted and true Fantasy Points in dataset

            Args:
                absolute (bool, optional): whether to compute absolute value of error before computing the average. Defaults to False.
                fig (matplotlib.figure.Figure, optional): handle to the figure to add the histogram to. Defaults to None.
        """

        # Call plotting function
        plot_error_histogram(self, absolute=absolute, fig=fig)


    def plot_scatters(self, all_plot_settings, fig=None):
        """Generates scatterplots comparing predicted and true stats, optionally with subsections ("slices) of the dataset

            Args:
                all_plot_settings (list): list of dicts, where each element of the list specifies a new figure. 
                    Each element of the list is a dict which may contain the following fields:
                    - columns (list, optional): list of the stats (e.g. 'Pass Yds') to plot in the figure. Each element is plotted on a separate subplot.
                    - slice (dict, optional): subset of the evaluated dataset to include in the figure. Keys may be 'Position', 'Team', or 'Player'.
                    - legend_slice (dict, optional): subsets of the evaluated dataset to split into separate entities in the plot legend. Same keys as slice.
                    - subtitle (str, optional): text to include as a subtitle on the figure.
                    - histograms (bool, optional): whether to include histograms on the axes of each subplot. Defaults to False.
                fig (matplotlib.figure.Figure, optional): handle to the figure to add the plots to. Defaults to None.
        """

        # Generate all scatter plots
        for plot_settings in all_plot_settings:
            # If a pre-determined figure has been passed, pass it through to the plot function
            if fig:
                plot_settings['fig'] = fig
            # Call larger function to plot one figure at a time
            gen_scatterplots(self, **plot_settings)


    def plot_single_games(self, **kwargs):
        """Generates a line graph of predicted and true Fantasy Points over the course of a game for a single player (or multiple players)
        
            Keyword-Args:
                game_ids (list, optional): list of pre-determined games/players to visualize. Each element of list is a dict with keys:
                    - "Player" : value -> str
                    - "Year" : value -> int
                    - "Week" : value -> int
                n_random (int, optional): number of games/players to randomly generate and display (besides pre-determined games in game_ids). Defaults to 0.
                fig (matplotlib.figure.Figure, optional): handle to the figure to add the plots to. Defaults to None.
        """

        # kwargs
        game_ids = kwargs.get('game_ids', None)
        n_random = kwargs.get('n_random', 0)
        fig = kwargs.get('fig', None)

        # Handle optional input of specific games/players
        if game_ids is None:
            game_ids = []

        # Generate random games to display (as well as any pre-defined games in game_ids)
        game_ids = gen_random_games(self.id_df, n_random, game_ids=game_ids)

        for game_id in game_ids:
            plot_game_timeline(self, game_id, fig=fig)


    # PRIVATE METHODS

    def __pbp_with_fantasy_points(self, **kwargs):
        # Extracts play-by-play (midgame) data for each game in the PredictionResult's dataset
        # and computes the Fantasy Points for each midgame stat line

        # All keyword arguments are passed to stats_to_fantasy_points. See that function's documentation for descriptions and valid inputs.
        # Keyword arguments are optional, with defaults set in stats_to_fantasy_points, EXCEPT:
        # - normalized defaults to True in this implementation.

        # Make copy of dataset
        pbp_df = self.dataset.x_df.copy().reset_index(drop=True)

        # Extract any necessary keyword argument values
        kwargs['normalized'] = kwargs.get('normalized',True) # Note this defaults to True instead of the standard False
        norm_thresholds = kwargs.get('norm_thresholds',stats_config.default_norm_thresholds)

        # Remove any columns that do not need to be unnormalized
        if kwargs['normalized']:
            set_normalizable_columns = frozenset(norm_thresholds)
            columns = [x for x in pbp_df.columns if x in set_normalizable_columns]
            # columns = list(set(pbp_df.columns.tolist()) & set(stats_config.default_norm_thresholds))
            pbp_df = pbp_df[columns]

        # Compute Fantasy Points
        pbp_df = stats_to_fantasy_points(pbp_df, **kwargs)
        return pbp_df
