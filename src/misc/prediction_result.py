from dataclasses import dataclass, InitVar
import matplotlib.pyplot as plt
import pandas as pd

from misc.dataset import CustomDataset
from misc.nn_helper_functions import stats_to_fantasy_points, gen_random_games
from .result_plots import gen_scatterplots, plot_game_timeline, plot_error_histogram

@dataclass
class PredictionResultGroup():
    # CONSTRUCTOR
    results: list

    def __post_init__(self):
        self.names = [result.predictor_name for result in self.results]


    # PUBLIC METHODS

    def plot_all(self, plot_func, *args, **kwargs):
        # kwargs consumed by the plot_all wrapper
        together = kwargs.get('together',False)
        kwargs.pop('together',None)

        # If plotting together, create figure and add it to arguments passed into plot function
        if together:
            fig = plt.subplots()[0]
            kwargs['fig'] = fig

        # Call plot function on all results in the group (passing unused kwargs to plot function)
        for result in self.results:
            plot_func(result, *args, **kwargs)

        # Add a legend if plotting together
        if together:
            # fig.axes[0].legend(self.names)
            fig.legend(self.names)

@dataclass
class PredictionResult():
    # CONSTRUCTOR
    predictor: InitVar[object]
    predicts: pd.DataFrame
    truths: pd.DataFrame
    dataset: CustomDataset

    def __post_init__(self, predictor):
        self.predictor_name = predictor.name
        # ID (player, week, year, team, position, etc) for each data point
        self.id_df = self.dataset.__getids__().reset_index(drop=True)
        # Play-by-play data with fantasy score (used in plot_single_game)
        self.pbp_df = self.__pbp_with_fantasy_points()


    # PUBLIC METHODS

    def avg_diff(self,absolute=False):
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
        # Call plotting function
        plot_error_histogram(self, absolute=absolute, fig=fig)


    def plot_scatters(self, all_plot_settings, fig=None):
        # Generate all scatter plots
        for plot_settings in all_plot_settings:
            # If a pre-determined figure has been passed, pass it through to the plot function
            if fig:
                plot_settings['fig'] = fig
            # Call larger function to plot one figure at a time
            gen_scatterplots(self, **plot_settings)


    def plot_single_games(self, **kwargs):
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

    def __pbp_with_fantasy_points(self):
        # Add fantasy points to play-by-play
        cols_to_keep = [
            'Elapsed Time',
            'Pass Att',
            'Pass Cmp',
            'Pass Yds',
            'Pass TD',
            'Int',
            'Rush Att',
            'Rush Yds',
            'Rush TD',
            'Rec',
            'Rec Yds',
            'Rec TD',
            'Fmb']
        pbp_data = self.dataset.x_data[:,[self.dataset.x_data_labels.index(col) for col in cols_to_keep]]
        pbp_df = stats_to_fantasy_points(pbp_data, stat_indices=cols_to_keep, normalized=True)
        return pbp_df
