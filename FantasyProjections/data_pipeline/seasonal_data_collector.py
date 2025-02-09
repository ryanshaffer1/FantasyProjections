"""Creates and exports class to be used in NFL player statistics data collection.

Classes:
    SeasonalDataCollector : Collects data (e.g. player stats) for all games in an NFL season. Automatically processes data upon initialization.
"""

import logging

import dateutil.parser as dateparse
import pandas as pd
from config import data_files_config, player_id_config
from config.player_id_config import fill_blank_player_ids
from data_pipeline.utils import team_abbreviations as team_abbrs
from data_pipeline.utils.data_helper_functions import clean_team_names
from misc.manage_files import collect_input_dfs

# Set up logger
logger = logging.getLogger('log')


class SeasonalDataCollector:
    """Collects data (e.g. player stats) for all games in an NFL season. Automatically processes data upon initialization.
    
        Specific data processing steps are carried out in sub-classes.
    
        Args:
            year (int): Year for season (e.g. 2023).
            team_names (str | list, optional): Either "all" or a list of full team names (e.g. ["Arizona Cardinals", "Baltimore Ravens", ...]). Defaults to "all".
            weeks (list, optional): Weeks in the NFL season to collect data for. Defaults to range(1,19).
        Keyword Arguments: 
            filter_df (pandas.DataFrame, optional): Filter for roster, i.e. the list of players to collect data for. Defaults to None (collect data for every player).
                Not stored as an object attribute.

        Additional Attributes Created during Initialization:
            pbp_df (pandas.DataFrame): Play-by-play data for all plays in an NFL season, taken from nfl-verse.
            raw_rosters_df (pandas.DataFrame): Weekly roster information for all teams in an NFL season, taken from nfl-verse.
            all_rosters_df (pandas.DataFrame): Filtered roster dataframe to include only players of interest (skill positions, in the optional filter_df, etc.)
            games (list): List of SingleGameDataWorker (or sub-class) objects containing data for every game in the NFL season.


        Objects Created:
            List of SingleGameDataWorker (or sub-class) objects
        
        Public Methods: 
            gather_all_game_data : Concatenates all relevant data from individual games in self.games into larger DataFrames for the full season.
            process_rosters : Trims DataFrame of all NFL week-by-week rosters in a given year to include only players of interest and data columns of interest.
    """  # fmt: skip

    def __init__(self, year, team_names='all', weeks=None, **kwargs):
        """Constructor for SeasonalDataCollector class.

            Args:
                year (int): Year for season (e.g. 2023).
                team_names (str | list, optional): Either "all" or a list of full team names (e.g. ["Arizona Cardinals", "Baltimore Ravens", ...]). Defaults to "all".
                weeks (list, optional): Weeks in the NFL season to collect data for. Defaults to range(1,19).
            Keyword Arguments: 
                filter_df (pandas.DataFrame, optional): Filter for roster, i.e. the list of players to collect data for. Defaults to None (collect data for every player).
                    Not stored as an object attribute.

            Additional Attributes Created during Initialization:
                pbp_df (pandas.DataFrame): Play-by-play data for all plays in an NFL season, taken from nfl-verse.
                raw_rosters_df (pandas.DataFrame): Weekly roster information for all teams in an NFL season, taken from nfl-verse.
                all_rosters_df (pandas.DataFrame): Filtered roster dataframe to include only players of interest (skill positions, in the optional filter_df, etc.)
                games (list): List of SingleGameDataWorker (or sub-class) objects containing data for every game in the NFL season.
        """  # fmt: skip

        # Handle unspecified weeks: all weeks
        if weeks is None:
            weeks = range(1, 19)

        # Optional keyword arguments
        filter_df = kwargs.get('filter_df', None)

        # Basic attributes
        self.year = year
        self.weeks = weeks
        self.team_names = clean_team_names(team_names, self.year)

        # Collect input data
        self.pbp_df, self.raw_rosters_df, *_ = collect_input_dfs(
            self.year, self.weeks, data_files_config.local_file_paths, data_files_config.online_file_paths, online_avail=True
        )

        # Gather team roster for all teams, all weeks of input year
        self.all_rosters_df = self.process_rosters(filter_df)

        # Initialize necessary attributes to be modified later
        self.games = []

    # PUBLIC METHODS

    def gather_all_game_data(self, df_fields):
        """Concatenates all relevant data from individual games in self.games into larger DataFrames for the full season.

            Args:
                df_fields (list | str): Names of the DataFrame properties in each game to concatenate.

            Returns:
                tuple(pandas.DataFrame): DataFrames of data consolidated across all games, one for each element in df_fields.
        """  # fmt: skip

        # Handle single field passed as string
        if isinstance(df_fields, str):
            df_fields = [df_fields]

        dfs = [0] * len(df_fields)
        for i, df_field in enumerate(df_fields):
            # Initialize empty dataframe
            df = pd.DataFrame()
            for game in self.games:
                df = pd.concat((df, getattr(game, df_field)))

            logger.debug(f'{self.year} {df_field} rows: {df.shape[0]}')
            dfs[i] = df

        return tuple(dfs)

    def process_rosters(self, filter_df=None):
        """Trims DataFrame of all NFL week-by-week rosters in a given year to include only players of interest and data columns of interest.

            Args:
                filter_df (pandas.DataFrame, optional): Pre-determined list of players to include. Defaults to None.

            Attributes Modified:
                pandas.DataFrame: all_rosters_df filtered to players of interest, several columns removed, and indexed on Team & Week
        """  # fmt: skip

        # Copy of object attribute
        all_rosters_df = self.raw_rosters_df.copy()

        # Filter to only the desired weeks
        all_rosters_df = all_rosters_df[all_rosters_df.apply(lambda x: x['week'] in self.weeks, axis=1)]

        # Filter to only the desired teams
        all_rosters_df = all_rosters_df[
            all_rosters_df.apply(lambda x: x['team'] in [team_abbrs.pbp_abbrevs[name] for name in self.team_names], axis=1)
        ]

        # Optionally filter based on subset of desired players
        if filter_df is not None:
            all_rosters_df = all_rosters_df[all_rosters_df.apply(lambda x: x['full_name'] in filter_df['Name'].to_list(), axis=1)]

        # Filter to only skill positions
        # Positions currently being tracked for stats
        skill_positions = ['QB', 'RB', 'FB', 'HB', 'WR', 'TE']
        all_rosters_df = all_rosters_df[all_rosters_df.apply(lambda x: x['position'] in skill_positions, axis=1)]

        # Filter to only active players
        valid_statuses = ['ACT']
        all_rosters_df = all_rosters_df[all_rosters_df.apply(lambda x: x['status'] in valid_statuses, axis=1)]

        # Compute age based on birth date. Assign birth year of 2000 for anyone with missing birth date...
        all_rosters_df['Age'] = all_rosters_df['season'] - all_rosters_df['birth_date'].apply(
            lambda x: dateparse.parse(x).year if (x == x) else 2000
        )

        # Trim to just the fields that are useful
        all_rosters_df = all_rosters_df[
            player_id_config.PLAYER_IDS + ['team', 'week', 'position', 'jersey_number', 'full_name', 'Age']
        ]
        # Reformat
        all_rosters_df = (
            all_rosters_df.rename(
                columns={
                    'team': 'Team',
                    'week': 'Week',
                    'position': 'Position',
                    'jersey_number': 'Number',
                    'full_name': 'Player Name',
                }
            )
            .set_index(['Team', 'Week'])
            .sort_index()
        )

        # Update player IDs
        all_rosters_df = fill_blank_player_ids(
            players_df=all_rosters_df,
            master_id_file=data_files_config.MASTER_PLAYER_ID_FILE,
            pfr_id_filename=data_files_config.PFR_ID_FILENAME,
            add_missing_pfr=False,
            update_master=False,
        )

        return all_rosters_df
