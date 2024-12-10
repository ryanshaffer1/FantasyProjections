"""Creates and exports classes to be used as one approach to predicting NFL stats and Fantasy Football scores.

    Classes:
        SleeperPredictor : child of FantasyPredictor. Predicts NFL player stats using the Sleeper Fantasy web app/API.
"""

from dataclasses import dataclass
import json
import logging
import torch
from sleeper_wrapper import Stats, Players
from misc.nn_helper_functions import stats_to_fantasy_points, remove_game_duplicates
from .fantasypredictor import FantasyPredictor

# Set up logger
logger = logging.getLogger('log')

@dataclass
class SleeperPredictor(FantasyPredictor):
    """Predictor of NFL players' stats in games, using the Sleeper Fantasy web app/API to generate predictions.
    
        Sub-class of FantasyPredictor.
        Pulls stat predictions from Sleeper Fantasy. How they compute their predictions is unclear.
        Sleeper API documentation: https://github.com/SwapnikKatkoori/sleeper-api-wrapper

        Args:
            name (str): name of the predictor object, used for logging/display purposes.
            player_dict_file (str): filepath (including filename) to .json file storing all player/roster information from Sleeper. Required input.
            proj_dict_file (str, optional): filepath (including filename) to .json file storing all stat projections made by Sleeper.
                Defaults to None. If a file is not entered, or the file does not contain all the necessary data, updated information will
                automatically be requested from Sleeper.
            update_players (bool, optional): whether to request updated information on NFL players/rosters from Sleeper. Defaults to False.
        
        Additional Class Attributes:
            player_to_sleeper_id (dict): dictionary containing the names of all players in Sleeper's database, 
                mapped to their Sleeper ID numbers (used in projections)
            all_proj_dict (dict): dictionary mapping NFL weeks (in "year-week" format) to the pre-game stat predictions from Sleeper for that week.
                Each set of pre-game stat predictions is a dict mapping a player's Sleeper ID to their predicted stat line.
        
        Public Methods:
            eval_model : Generates predicted stats for an input evaluation dataset, as provided by Sleeper.
            refresh_players : Updates player dictionary (player names to ID numbers) using Sleeper API
    """

    # CONSTRUCTOR
    player_dict_file: str = None
    proj_dict_file: str = None
    update_players: bool = False

    def __post_init__(self):
        # Evaluates as part of the Constructor.
        # Generates attributes that are not simple data copies of inputs.

        # If no player dict file is input, player list must be updated from Sleeper API
        if self.player_dict_file is None:
            self.update_players = True

        # Generate dictionary mapping player names to IDs
        if self.update_players:
            self.player_to_sleeper_id = self.refresh_players()
        else:
            self.player_to_sleeper_id = self.__load_players()
        # Initialize attributes defined later (dependent on eval data used)
        self.all_proj_dict = {}


    # PUBLIC METHODS

    def eval_model(self, eval_data, **kwargs):
        """Generates predicted stats for an input evaluation dataset, as provided by Sleeper.

            Note that only pre-game predictions will be included in the evaluation result. If multiple game times in each game
            are present in eval_data, only one prediction per game will be made, with the other rows automatically dropped.

            Args:
                eval_data (StatsDataset): data to use for Neural Net evaluation (e.g. validation or test data).

            Keyword-Args:
                All keyword arguments are passed to the function stats_to_fantasy_points and to the PredictionResult constructor. 
                See the related documentation for descriptions and valid inputs. All keyword arguments are optional.

            Returns:
                PredictionResult: Object packaging the predicted and true stats together, which can be used for plotting, 
                    performance assessments, etc.
        """

        # List of stats being used to compute fantasy score
        stat_columns = eval_data.y_df.columns.tolist()
        num_stats = len(stat_columns)

        # Remove duplicated games from eval data (only one projection per game from Sleeper)
        eval_data = remove_game_duplicates(eval_data)

        # Gather projections data from Sleeper API
        self.all_proj_dict = self.__gather_sleeper_proj(eval_data)

        # Build up array of predicted stats for all eval_data cases based on
        # sleeper projections dictionary
        stat_predicts = torch.empty(0)
        for row in range(eval_data.id_data.shape[0]):
            id_row = eval_data.id_data.iloc[row]
            year_week = id_row['Year-Week']
            player = id_row['Player']
            if player in self.player_to_sleeper_id:
                proj_stats = self.all_proj_dict[year_week][self.player_to_sleeper_id[player]]
                stat_line = torch.tensor(self.__reformat_sleeper_stats(proj_stats, stat_columns))
            else:
                stat_line = torch.zeros([num_stats])
            stat_predicts = torch.cat((stat_predicts, stat_line))


        # Compute fantasy points using stat lines (note that this ignores the
        # built-in fantasy points projection in the Sleeper API, which differs
        # from the sum of the stats)
        stat_predicts = stats_to_fantasy_points(torch.reshape(stat_predicts, [-1, num_stats]),
                                                stat_indices=stat_columns, **kwargs)

        # True stats from eval data
        stat_truths = self.eval_truth(eval_data, **kwargs)

        # Create result object
        result = self._gen_prediction_result(stat_predicts, stat_truths, eval_data, **kwargs)

        return result


    def refresh_players(self):
        """Updates player dictionary (player names to ID numbers) using Sleeper API.

            Returns:
                dict: dictionary containing the names of all players in Sleeper's database, 
                    mapped to their Sleeper ID numbers (used in projections)
        """

        players = Players()
        player_dict = players.get_all_players()

        # Re-organize player dict into dictionary mapping full names to Sleeper player IDs
        # TODO: THIS DOESN'T WORK --- MULTIPLE PLAYERS WITH SAME NAME AND DIFFERENT IDS.
        # EX: MIKE WILLIAMS
        player_to_sleeper_id = {}
        for player in player_dict:
            sleeper_id = player
            player_name = player_dict[player].get('full_name', None)
            if player_name:
                player_to_sleeper_id[player_name] = sleeper_id
            else:
                logger.warning(f'Warning: {player} not added to player dictionary')

        # Save player dictionary to JSON file for use next time
        try:
            with open(self.player_dict_file, 'w', encoding='utf-8') as file:
                json.dump(player_to_sleeper_id, file)
        except (FileNotFoundError, TypeError):
            logger.warning('Sleeper player dictionary file not found during save process.')

        return player_to_sleeper_id


    # PRIVATE METHODS

    def __gather_sleeper_proj(self, eval_data):
        # Loads all_proj_dict from file (filename is an attribute of SleeperPredictor)
        # and checks if all the necessary data to evaluate against eval_data is present.
        # (i.e. are all the weeks in eval_data also present in all_proj_dict). If not,
        # updates all_proj_dict by requesting predictions for the missing weeks from Sleeper.

        # Unique year-week combinations in evaluation dataset
        eval_data.id_data['Year-Week'] = eval_data.id_data[['Year',
                                                    'Week']].astype(str).agg('-'.join, axis=1)
        unique_year_weeks = list(eval_data.id_data['Year-Week'].unique())

        # Gather all stats from Sleeper
        try:
            with open(self.proj_dict_file, 'r', encoding='utf-8') as file:
                all_proj_dict = json.load(file)
        except (FileNotFoundError, TypeError):
            logger.warning('Sleeper projection dictionary file not found during load process.')
            all_proj_dict = {}

        if not all(year_week in all_proj_dict for year_week in unique_year_weeks):
            # Gather any unsaved stats from Sleeper
            stats = Stats()
            for year_week in unique_year_weeks:
                if year_week not in all_proj_dict:
                    [year, week] = year_week.split('-')
                    week_proj = stats.get_week_projections(
                        'regular', int(year), int(week))
                    all_proj_dict[year_week] = week_proj
                    logger.info(
                        f'Adding Year-Week {year_week} to Sleeper projections dictionary: {self.proj_dict_file}')
            # Save projection dictionary to JSON file for use next time
            try:
                with open(self.proj_dict_file, 'w', encoding='utf-8') as file:
                    json.dump(all_proj_dict, file)
            except (FileNotFoundError, TypeError):
                logger.warning('Sleeper projection dictionary file not found during save process.')

        return all_proj_dict


    def __load_players(self):
        # Loads the player_to_sleeper_id dictionary from a local .json file.
        try:
            with open(self.player_dict_file, 'r', encoding='utf-8') as file:
                player_to_sleeper_id = json.load(file)
        except (FileNotFoundError, TypeError):
            logger.warning('Sleeper player dictionary file not found during load. Refreshing from Sleeper API.')
            player_to_sleeper_id = self.refresh_players()

        return player_to_sleeper_id


    def __reformat_sleeper_stats(self, stat_dict, stat_columns):
        # Re-names stats from Sleeper's format to the common names used across this project
        # and lists into the common stat line format.
        # TODO: this is kinda janky, no?

        labels_df_to_sleeper = {
            'Pass Att': 'pass_att',
            'Pass Cmp': 'pass_cmp',
            'Pass Yds': 'pass_yd',
            'Pass TD': 'pass_td',
            'Int': 'pass_int',
            'Rush Att': 'rush_att',
            'Rush Yds': 'rush_yd',
            'Rush TD': 'rush_td',
            'Rec': 'rec',
            'Rec Yds': 'rec_yd',
            'Rec TD': 'rec_td',
            'Fmb': 'fum_lost'
        }

        # stat_line = []
        # for sleeper_stat_label in labels_df_to_sleeper.values():
        #     stat_value = stat_dict.get(sleeper_stat_label, 0)
        #     stat_line.append(stat_value)

        stat_line = []
        for stat in stat_columns:
            stat_value = stat_dict.get(labels_df_to_sleeper[stat], 0)
            stat_line.append(stat_value)

        return stat_line
