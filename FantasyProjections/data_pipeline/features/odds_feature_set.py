from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import cast

import dateutil.parser as dateparse
import numpy as np
import pandas as pd
import requests

from config.player_id_config import PRIMARY_PLAYER_ID
from data_pipeline.features.feature_set import FeatureSet
from data_pipeline.utils import team_abbreviations as team_abbrs
from data_pipeline.utils.data_helper_functions import construct_game_id
from data_pipeline.utils.name_matching import find_matching_name_ind
from data_pipeline.utils.time_helper_functions import date_to_nfl_week, find_prev_time_index, week_to_date_range

# Set up logger
logger = logging.getLogger("log")

# Constant variables used for The Odds API
SPORT_KEY = "americanfootball_nfl"
DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"
BOOKMAKER = "fanduel"


class OddsFeatureSet(FeatureSet):
    def __init__(self, features, sources, **kwargs):
        super().__init__(features, sources)
        # Optional keyword arguments
        self.surrogate = kwargs.get("surrogate", False)
        self.game_times = kwargs.get("game_times")
        self.game_to_event_ids = {}

    def post_init(self, data_files_config: dict):
        """Post-initialization method to set up the Odds API Manager and other attributes that rely on data_files_config."""
        self.api_manager = OddsAPIManager(data_files_config=data_files_config, surrogate=self.surrogate)
        self.labels_df_to_odds = pd.read_csv(data_files_config["feature_config_file"], index_col=0)["odds"].dropna().to_dict()
        self.markets = [
            self.labels_df_to_odds[stat] for stat in list({feat.name for feat in self.features} & set(self.labels_df_to_odds))
        ]

    def collect_data(
        self,
        year: int,
        weeks: list[int] | range,
        _df_sources: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        # Initialize API Manager

        team_abbrevs_to_process = list(team_abbrs.pbp_abbrevs.values())

        year_start_date, _ = week_to_date_range(year, week=1)
        year_start_date = datetime.strftime(year_start_date, DATE_FMT)
        # Get all events in week
        endpoint = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/events"
        request_params = {"date": year_start_date}
        response, success = self.api_manager.make_api_request(endpoint, request_params)

        if success:
            events_list = json.loads(response)["data"]

            # Extract IDs from each event that matches the input criteria
            game_to_event_ids = {}
            for event in events_list:
                year, week = date_to_nfl_week(event["commence_time"])
                home_team = team_abbrs.pbp_abbrevs[event["home_team"]]
                away_team = team_abbrs.pbp_abbrevs[event["away_team"]]
                game_data = {"Year": year, "Week": week, "Home Team": home_team, "Away Team": away_team}
                if week in weeks and (home_team in team_abbrevs_to_process or away_team in team_abbrevs_to_process):
                    game_id = construct_game_id(game_data)
                    game_to_event_ids[game_id] = event["id"]
            self.game_to_event_ids = game_to_event_ids
        elif self.surrogate:
            # If surrogate is enabled, use a surrogate game ID
            logger.info(f"TheOddsAPI: Using surrogate game IDs for {year}.")
            self.game_to_event_ids = {"surrogate_game_id": "surrogate_event_id"}
        else:
            msg = f"TheOddsAPI: Failed to retrieve game IDs for {year}."
            logger.error(msg)
            raise ValueError(msg)

    def process_data(self, game_data_worker):
        #        """Collects historical odds data for the player props of interest at game times of interest.

        # Initialize output dataframe
        odds_df = pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], [], [], []], names=["Year", "Week", PRIMARY_PLAYER_ID, "Elapsed Time"]),
        )

        # Get game times from either feature set or game data worker
        game_times = self.__clean_game_times(game_data_worker)

        # Event ID for current game
        try:
            event_id = self.game_to_event_ids[game_data_worker.game_id]
        except KeyError:
            if self.surrogate:
                # If surrogate is enabled, use a surrogate event ID
                event_id = "surrogate_event_id"
            else:
                msg = f"Event ID not found for game {game_data_worker.game_id}. Aborting early."
                logger.warning(msg)
                return odds_df

        # Maps game time to UTC time
        self.all_game_times = game_data_worker.pbp_df["time_of_day"].dropna()

        # Make a separate API request for each desired game time, each market
        for game_time in game_times:
            # Determine time of the most recent play in order to request the odds from that time
            pbp_index = find_prev_time_index(game_time, self.all_game_times)
            time = datetime.strftime(dateparse.parse(self.all_game_times.iloc[pbp_index]), DATE_FMT)

            for market in self.markets:
                # Make the request to The Odds for data on the selected markets at the correct time
                endpoint = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/events/{event_id}/odds"
                request_params = {
                    "regions": "us",
                    "markets": market,
                    "date": time,
                }
                response, success = self.api_manager.make_api_request(endpoint, request_params)

                if success:
                    # Process API Response to obtain market data. Exit gracefully if response does not contain "data" or data does not contain "markets".
                    try:
                        event_data = json.loads(response)["data"]
                    except KeyError:
                        msg = f"Key Error encountered in API request. Aborting game {game_data_worker.game_id} early."
                        logger.warning(msg)
                        return odds_df
                    try:
                        market_data = cast(
                            "list",
                            pd.DataFrame(event_data["bookmakers"]).set_index("key").loc[BOOKMAKER, "markets"],
                        )
                    except KeyError:
                        msg = f"Key Error encountered: {BOOKMAKER} missing. Aborting game {game_data_worker.game_id} early."
                        logger.warning(msg)
                        return odds_df

                    # Process Odds data into a DataFrame
                    single_market_odds_df = pd.DataFrame(market_data[0]["outcomes"]).rename(
                        columns={"description": "Player Name", "name": "Line"},
                    )
                    # Add Player Prop Stat and UTC time from market data
                    player_prop_stat = team_abbrs.invert(self.labels_df_to_odds)[market_data[0]["key"]]
                    single_market_odds_df["UTC Time"] = market_data[0]["last_update"]

                # Handle unsuccessful API requests: create surrogate data or skip this market
                elif self.surrogate:
                    logger.info(
                        f"TheOddsAPI: Using surrogate odds for game {game_data_worker.game_id} at time {time} for market {market}.",
                    )
                    single_market_odds_df = self.gen_surrogate_odds(time, game_data_worker)
                    player_prop_stat = team_abbrs.invert(self.labels_df_to_odds)[market]
                else:
                    msg = f"TheOddsAPI: request failed for game {game_data_worker.game_id} at time {time} for market {market}."
                    logger.warning(msg)
                    continue

                # Format dataframe for output
                single_market_odds_df = self.reformat_odds_df(game_data_worker, single_market_odds_df, player_prop_stat)
                # Add to running list of all odds
                if single_market_odds_df is not None:
                    odds_df = odds_df.merge(single_market_odds_df, how="outer", left_index=True, right_index=True)

        return odds_df

    def gen_surrogate_odds(self, time, game_data_worker):
        single_market_odds_df = pd.DataFrame()
        single_market_odds_df["Player Name"] = game_data_worker.roster_df.set_index("Player Name").index.repeat(2)
        single_market_odds_df["Line"] = ["Over", "Under"] * len(game_data_worker.roster_df)
        single_market_odds_df["price"] = 0
        single_market_odds_df["point"] = 0
        single_market_odds_df["UTC Time"] = time

        return single_market_odds_df

    def reformat_odds_df(self, game_data_worker, df_to_format, player_prop_stat):
        """Formats data collected from The Odds into consistent style as other data products (common Player IDs, etc.) and structures as one row per player.

            Args:
                df_to_format (pandas.DataFrame): Data collected from The Odds with some minor formatting/added info already applied.

            Returns:
                pandas.DataFrame: Data in consistent structure/style as other data products.

        """  # fmt: skip

        # Modify into the desired DataFrame structure
        if "Line" in df_to_format.columns:
            df_to_format = df_to_format.set_index(["Player Name", "UTC Time"])
            for line in ["Over", "Under"]:
                df_to_format[f"{player_prop_stat} {line} Point"] = (
                    df_to_format.reset_index().set_index(["Player Name", "UTC Time", "Line"]).xs(line, level=2)["point"]
                )
                df_to_format[f"{player_prop_stat} {line} Price"] = (
                    df_to_format.reset_index().set_index(["Player Name", "UTC Time", "Line"]).xs(line, level=2)["price"]
                )
            df_to_format = df_to_format.reset_index().drop(columns=["Line", "price", "point"]).drop_duplicates(keep="first")

        # Add Year, Week, and Game ID
        df_to_format["Year"] = game_data_worker.year
        df_to_format["Week"] = game_data_worker.week
        df_to_format["Game ID"] = game_data_worker.game_id

        # Add Player ID based on roster. Remove any players not in the roster.
        roster_inds = df_to_format["Player Name"].apply(
            find_matching_name_ind,
            args=(game_data_worker.roster_df["Player Name"].unique(),),
        )
        missing_players = df_to_format[roster_inds.isna()]
        df_to_format = df_to_format.drop(missing_players.index)
        if df_to_format.empty:
            return None
        df_to_format[PRIMARY_PLAYER_ID] = roster_inds.apply(
            lambda x: game_data_worker.roster_df.reset_index().iloc[int(x)][PRIMARY_PLAYER_ID] if not np.isnan(x) else np.nan,
        )

        # Add Elapsed Time
        elapsed_time_inds = df_to_format["UTC Time"].apply(find_prev_time_index, args=(self.all_game_times,))
        df_to_format["Elapsed Time"] = self.all_game_times.index[elapsed_time_inds].to_list()

        # Drop any rows that are not outputs
        df_to_format = df_to_format.drop(columns=["Player Name", "UTC Time", "Game ID"])

        # Set index to match other feature sets
        df_to_format = df_to_format.set_index(["Year", "Week", PRIMARY_PLAYER_ID, "Elapsed Time"])

        return df_to_format

    def __clean_game_times(self, game_data_worker):
        # Get game times from either feature set or game data worker, and ensure they are in a list
        game_times = self.game_times if self.game_times is not None else game_data_worker.game_times
        if game_times is None:
            game_times = [0]  # Default to just the start of the game if not game times are provided
        elif isinstance(game_times, np.ndarray):
            game_times = game_times.tolist()
        return game_times


class OddsAPIManager:
    def __init__(self, data_files_config, enable_api_requests: bool = True, surrogate: bool = False):
        self.data_files_config = data_files_config
        self.api_key = self.get_odds_api_key(data_files_config["odds_api_key_file"])
        self.surrogate = surrogate
        self.enable_api_requests = enable_api_requests or not surrogate

        # Track previous API requests
        self.all_requests_file = self.data_files_config["odds_requests_file"]
        self.all_requests_df = self.process_previous_requests()

    def process_previous_requests(self) -> pd.DataFrame:
        # Read the JSON file
        try:
            all_requests_df = pd.read_json(self.all_requests_file, orient="records", lines=True)
        except FileNotFoundError:
            # Set up an empty dataframe
            all_requests_df = pd.DataFrame(columns=["End Point", "Params", "Response Code", "Response Text"])

        # Filter out any duplicated requests (in case they were mistaken made twice)
        all_requests_df = all_requests_df.drop_duplicates(subset=["End Point", "Params"], keep="last")

        return all_requests_df

    def get_odds_api_key(self, filename: str) -> str:
        """Reads API key for The Odds API from a local file. API key must be obtained manually and saved to this file by the user.

            The file must be a text file containing ONLY the API key and no other text.

            Args:
                filename (str): Path to text file containing API key.

            Returns:
                str: The Odds API key read from the file.

        """  # fmt: skip

        with open(filename, encoding="utf-8") as file:
            api_key = file.readline()

        return api_key

    def make_api_request(self, endpoint: str, request_params: dict) -> tuple[str, bool]:
        # Check for an identical request in the database of previous requests
        params_string = str(dict(sorted(request_params.items())))
        try:
            response = str(
                self.all_requests_df.set_index(["End Point", "Params"]).loc[
                    (endpoint, params_string),
                    "Response Text",
                ],
            )
        except KeyError:
            pass
        else:
            # Response found in the database, so return it
            success = True
            return response, success

        # If surrogate turned on, skip making the API request (surrogate data will be generated later)
        if self.surrogate:
            return "", False

        # Add API key to the request params
        request_params["api_key"] = self.api_key

        # Make the API request
        if self.enable_api_requests:
            response = requests.get(endpoint, params=request_params, timeout=10)
            response_code = response.status_code
            response_text = response.text
        else:
            msg = "API requests are disabled, but surrogate data is not enabled, so no request was made to The Odds API."
            raise ValueError(msg)

        # Log API usage
        self.log_api_usage(response)

        # Handle successful/unsuccessful response codes
        success, msg = self.handle_response_codes(response_code)
        if msg:
            logger.warning(msg)

        # Store the successful request/result in the dataframe and file
        if success:
            new_request_df = pd.DataFrame(
                columns=self.all_requests_df.columns,
                data=[[endpoint, params_string, response_code, str(response_text)]],
            )
            self.all_requests_df = pd.concat((self.all_requests_df, new_request_df)).reset_index(drop=True)
            new_request_df.to_json(self.all_requests_file, orient="records", lines=True, mode="a")

        return response_text, success

    def log_api_usage(self, response):
        """Logs the number of API requests fulfilled by The Odds with the current API key, and how many requests are remaining on the key.

            Args:
                response (requests.models.Response): Response from The Odds API, containing metadata on the number of requests made/available.

        """  # fmt: skip

        # Check for a deactivated API key
        deactivated_key_status = 401
        if response.status_code == deactivated_key_status:
            logger.info("TheOddsAPI: API key is deactivated.")
        else:
            # Check API key usage
            logger.info(
                f"TheOddsAPI: {response.headers['x-requests-remaining']} requests left ({response.headers['x-requests-used']} used)",
            )

    def handle_response_codes(self, response_code: int) -> tuple[bool, str | None]:
        match response_code:
            case 200:
                # Successful
                success = True
                msg = None
            case 422:
                # Unprocessable Entity
                success = False
                msg = "TheOddsAPI returned code: Unprocessable Entity. This may be due to an invalid market."
            case _:
                success = False
                msg = "TheOddsAPI returned unknown code!"

        return success, msg
