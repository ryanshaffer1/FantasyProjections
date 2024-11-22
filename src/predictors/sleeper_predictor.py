from dataclasses import dataclass
import json
import torch
from sleeper_wrapper import Stats, Players
from misc.nn_helper_functions import stats_to_fantasy_points, remove_game_duplicates
from .fantasypredictor import FantasyPredictor

@dataclass
class SleeperPredictor(FantasyPredictor):
    # Prediction Algorithm: Pulls projections from Sleeper Fantasy. Not sure how they compute their projections!
    # Sleeper API calls documented here:
    # https://github.com/SwapnikKatkoori/sleeper-api-wrapper

    # CONSTRUCTOR
    player_dict_file: str = None
    proj_dict_file: str = None
    update_players: bool = False

    def __post_init__(self):
        # Generate dictionary mapping player names to IDs
        if self.update_players:
            self.player_to_sleeper_id = self.refresh_players()
        else:
            self.player_to_sleeper_id = self.__load_players()
        # Initialize attributes defined later (dependent on eval data used)
        self.all_proj_dict = {}

    # def __init__(self, name, player_dict_file, proj_dict_file, update_players=False):
    #     # Initialize FantasyPredictor
    #     super().__init__(name)
    #     # Files with data from Sleeper
    #     self.player_dict_file = player_dict_file
    #     self.proj_dict_file = proj_dict_file
    #     # Generate dictionary mapping player names to IDs
    #     if update_players:
    #         self.player_to_sleeper_id = self.refresh_players()
    #     else:
    #         self.player_to_sleeper_id = self.__load_players()
    #     # Initialize attributes defined later (dependent on eval data used)
    #     self.all_proj_dict = {}


    # PUBLIC METHODS

    def eval_model(self, eval_data):
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
                stat_line = torch.tensor(self.__reformat_sleeper_stats(proj_stats))
            else:
                stat_line = torch.zeros([12])
            stat_predicts = torch.cat((stat_predicts, stat_line))

        # Compute fantasy points using stat lines (note that this ignores the
        # built-in fantasy points projection in the Sleeper API, which differs
        # from the sum of the stats)
        stat_predicts = stats_to_fantasy_points(torch.reshape(
            stat_predicts, [-1, 12]), stat_indices='default', normalized=False)

        # True stats from eval data
        stat_truths = self.eval_truth(eval_data)

        # Create result object
        result = self._gen_prediction_result(stat_predicts, stat_truths, eval_data)

        return result


    def refresh_players(self):
        players = Players()
        player_dict = players.get_all_players()

        # Re-organize player dict into dictionary mapping full names to Sleeper player IDs
        # THIS DOESN'T WORK --- MULTIPLE PLAYERS WITH SAME NAME AND DIFFERENT IDS.
        # EX: MIKE WILLIAMS
        player_to_sleeper_id = {}
        for player in player_dict:
            sleeper_id = player
            player_name = player_dict[player].get('full_name', None)
            if player_name:
                player_to_sleeper_id[player_name] = sleeper_id
            else:
                print(f'Warning: {player} not added to player dictionary')

        # Save player dictionary to JSON file for use next time
        with open(self.player_dict_file, 'w', encoding='utf-8') as file:
            json.dump(player_to_sleeper_id, file)

        return player_to_sleeper_id


    # PRIVATE METHODS

    def __gather_sleeper_proj(self, eval_data):
        # Unique year-week combinations in evaluation dataset
        eval_data.id_data['Year-Week'] = eval_data.id_data[['Year',
                                                    'Week']].astype(str).agg('-'.join, axis=1)
        unique_year_weeks = list(eval_data.id_data['Year-Week'].unique())

        # Gather all stats from Sleeper
        with open(self.proj_dict_file, 'r', encoding='utf-8') as file:
            all_proj_dict = json.load(file)
        if not all(year_week in all_proj_dict for year_week in unique_year_weeks):
            # Gather any unsaved stats from Sleeper
            stats = Stats()
            for year_week in unique_year_weeks:
                if year_week not in all_proj_dict:
                    [year, week] = year_week.split('-')
                    week_proj = stats.get_week_projections(
                        'regular', int(year), int(week))
                    all_proj_dict[year_week] = week_proj
                    print(
                        f'Adding Year-Week {year_week} to Sleeper projections dictionary: {self.proj_dict_file}')
            # Save player dictionary to JSON file for use next time
            with open(self.proj_dict_file, 'w', encoding='utf-8') as file:
                json.dump(all_proj_dict, file)

        return all_proj_dict


    def __load_players(self):
        with open(self.player_dict_file, 'r', encoding='utf-8') as file:
            player_to_sleeper_id = json.load(file)

        return player_to_sleeper_id


    def __reformat_sleeper_stats(self, stat_dict):
        stat_indices_df_to_sleeper = {
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
        stat_line = []
        for sleeper_stat_label in stat_indices_df_to_sleeper.values():
            stat_value = stat_dict.get(sleeper_stat_label, 0)
            stat_line.append(stat_value)

        return stat_line
