"""Creates and exports class to be used in NFL player statistics data collection.

    Classes:
        SeasonalOddsCollector : Collects all player stats for all games in an NFL season. Automatically pulls data from nfl-verse and processes upon initialization.
"""  # fmt: skip

import json
import logging
from datetime import datetime

import requests

from data_pipeline.odds_pipeline.odds_api_helper_functions import DATE_FMT, SPORT_KEY, get_odds_api_key, log_api_usage
from data_pipeline.odds_pipeline.single_game_odds_gatherer import SingleGameOddsGatherer
from data_pipeline.seasonal_data_collector import SeasonalDataCollector
from data_pipeline.utils import team_abbreviations as team_abbrs
from data_pipeline.utils.data_helper_functions import construct_game_id
from data_pipeline.utils.time_helper_functions import date_to_nfl_week, week_to_date_range

# Set up logger
logger = logging.getLogger("log")


class SeasonalOddsCollector(SeasonalDataCollector):
    """Collects all player props odds for all games in an NFL season. Automatically pulls data from The Odds API and processes upon initialization.

        Args:
            year (int): Year for season (e.g. 2023).
            team_names (str | list, optional): Either "all" or a list of full team names (e.g. ["Arizona Cardinals", "Baltimore Ravens", ...]). Defaults to "all".
            weeks (list, optional): Weeks in the NFL season to collect data for. Defaults to range(1,19).

        Keyword Arguments:
            filter_df (pandas.DataFrame): Filter for roster, i.e. the list of players to collect data for. Defaults to None (collect data for every player).
                Not stored as an object attribute.
            player_props (list, optional): List of stats to gather gambling odds for. Defaults to None.
            odds_file (str, optional): File path to csv containing all previously-obtained player prop odds.

        Additional Attributes Created during Initialization:
            pbp_df (pandas.DataFrame): Play-by-play data for all plays in an NFL season, taken from nfl-verse.
            raw_rosters_df (pandas.DataFrame): Weekly roster information for all teams in an NFL season, taken from nfl-verse.
            all_rosters_df (pandas.DataFrame): Filtered roster dataframe to include only players of interest (skill positions, in the optional filter_df, etc.)
            api_key (str): Key to use in requests to The Odds API.
            games (list): List of SingleGamePbpParser objects containing data for every game in the NFL season.
            odds_df (pandas.DataFrame): All collected (including previously-collected) player prop lines for all games in the season.

        Objects Created:
            List of SingleGameOddsGatherer objects

        Public Methods:
            generate_games : Creates a SingleGameOddsGatherer object for each unique game included in the SeasonalOddsCollector.

    """  # fmt: skip

    def __init__(self, year, team_names="all", weeks=None, **kwargs):
        """Constructor for SeasonalDataCollector class.

            Args:
                year (int): Year for season (e.g. 2023).
                team_names (str | list, optional): Either "all" or a list of full team names (e.g. ["Arizona Cardinals", "Baltimore Ravens", ...]). Defaults to "all".
                weeks (list, optional): Weeks in the NFL season to collect data for. Defaults to range(1,19).
                kwargs:
                    filter_df (pandas.DataFrame): Filter for roster, i.e. the list of players to collect data for. Defaults to None (collect data for every player).
                        Not stored as an object attribute.
                    player_props (list, optional): List of stats to gather gambling odds for. Defaults to None.
                    odds_file (str, optional): File path to csv containing all previously-obtained player prop odds.

            Additional Attributes Created during Initialization:
                pbp_df (pandas.DataFrame): Play-by-play data for all plays in an NFL season, taken from nfl-verse.
                raw_rosters_df (pandas.DataFrame): Weekly roster information for all teams in an NFL season, taken from nfl-verse.
                all_rosters_df (pandas.DataFrame): Filtered roster dataframe to include only players of interest (skill positions, in the optional filter_df, etc.)
                api_key (str): Key to use in requests to The Odds API.
                games (list): List of SingleGamePbpParser objects containing data for every game in the NFL season.
                odds_df (pandas.DataFrame): All collected (including previously-collected) player prop lines for all games in the season.

        """  # fmt: skip

        # Initialize SeasonalDataCollector
        super().__init__(year=year, team_names=team_names, weeks=weeks, **kwargs)

        # Optional keyword arguments
        player_props = kwargs.get("player_props")
        odds_file = kwargs.get("odds_file")

        # API Key
        self.api_key = get_odds_api_key()

        # List of SingleGameData objects
        self.games = self.generate_games(odds_file=odds_file, player_props=player_props)

        # Gather all stats (midgame and final) from the individual teams
        self.odds_df, *_ = self.gather_all_game_data(["odds_df"])

    # PUBLIC METHODS

    def generate_games(self, odds_file=None, player_props=None):
        """Creates a SingleGameOddsGatherer object for each unique game included in the SeasonalOddsCollector.

            Args:
                odds_file (str, optional): File path to csv containing all previously-obtained player prop odds. Defaults to None.
                player_props (list, optional): List of stats to gather gambling odds for. Defaults to None.

            Returns:
                list(SingleGameOddsGatherer): List of SingleGameOddsGatherer objects corresponding to each game included in the SeasonalOddsCollector.

        """  # fmt: skip

        team_abbrevs_to_process = [team_abbrs.pbp_abbrevs[name] for name in self.team_names]

        year_start_date, _ = week_to_date_range(self.year, week=1)
        year_start_date = datetime.strftime(year_start_date, DATE_FMT)
        # Get all events in week
        response = requests.get(
            f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/events",
            params={"api_key": self.api_key, "date": year_start_date},
            timeout=10,
        )
        log_api_usage(response)
        events_list = json.loads(response.text)["data"]

        # Extract IDs from each event
        event_ids = []
        game_ids = []
        for event in events_list:
            event_ids.append(event["id"])
            year, week = date_to_nfl_week(event["commence_time"])
            home_team = team_abbrs.pbp_abbrevs[event["home_team"]]
            away_team = team_abbrs.pbp_abbrevs[event["away_team"]]
            game_data = {"Year": year, "Week": week, "Home Team": home_team, "Away Team": away_team}
            if week in self.weeks and (home_team in team_abbrevs_to_process or away_team in team_abbrevs_to_process):
                game_ids.append(construct_game_id(game_data))

        # Create objects that process odds for each game of interest
        games = []
        n_games = len(game_ids)
        logger.info(f"Processing {n_games} games:")
        for i, (event_id, game_id) in enumerate(zip(event_ids, game_ids)):
            logger.info(f"({i + 1} of {n_games}): {game_id}")
            # Process data/stats for single game
            game = SingleGameOddsGatherer(self, event_id, game_id, odds_file=odds_file, player_props=player_props)
            # Add to list of games
            games.append(game)

        return games
