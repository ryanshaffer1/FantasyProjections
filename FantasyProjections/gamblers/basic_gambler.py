"""Creates and exports classes to be used in gambling on player props based on predicted stat lines.

    Classes:
        BasicGambler : Class containing an algorithm to place bets on NFL Player Props, based on predicted performance and odds/lines set by a sportsbook. Also contains the results of these bets.
"""  # fmt: skip

import numpy as np
import pandas as pd

from gamblers.gambling_plots import plot_earnings_by_week


class BasicGambler:
    """Class containing an algorithm to place bets on NFL Player Props, based on predicted performance and odds/lines set by a sportsbook. Also contains the results of these bets.

        The BasicGambler class implements a simple betting algorithm, where one unit is bet on the over if the predicted player performance is greater than
        the over/under line set by the sports book, and one unit is bet on the under if the predicted performance is less than the line.

        Attributes:
            prediction_result (PredictionResult): object created by a FantasyPredictor that contains predicted player stats, and the true outcomes of past games.
            odds_df (pandas.DataFrame): Odds data, processed to be used for placing and evaluating bets.
            bets (pandas.DataFrame): DataFrame containing all bets placed on player props in the dataset, as well as their payouts for winning and losing.
            bet_results (pandas.DataFrame): DataFrame containing the results of all bets placed on player props: whether the bet hit, and the earnings (positive or negative).
            earnings (float): Total gambling earnings (or losses, if negative) over the dataset
            accuracy (float): Accuracy, as a fraction of correct bets divided by all bets

        Public Methods:
            process_odds_df : Read master sheet of player prop odds, and format into DataFrame where each row matches the corresponding player/game in the prediction result dataset.
            place_bets : Uses predicted stats and odds over/under lines to decide whether to place a bet on each over or under.
            score_bets : Determines the success and payout from each bet placed on a player prop.
            compute_performance : Outputs basic gambling performance metrics: total earnings, and accuracy percentage.
            plot_earnings : Plots line graph tracking cumulative gambling earnings over each NFL week in the bet results.

    """  # fmt: skip

    def __init__(self, prediction_result):
        """Constructor for BasicGambler.

            Args:
                prediction_result (PredictionResult): object created by a FantasyPredictor that contains predicted player stats, and the true outcomes of past games.

            Additional Class Attributes (generated, not passed as inputs):
                odds_df (pandas.DataFrame): Odds data, processed to be used for placing and evaluating bets.
                bets (pandas.DataFrame): DataFrame containing all bets placed on player props in the dataset, as well as their payouts for winning and losing.
                bet_results (pandas.DataFrame): DataFrame containing the results of all bets placed on player props: whether the bet hit, and the earnings (positive or negative).
                earnings (float): Total gambling earnings (or losses, if negative) over the dataset
                accuracy (float): Accuracy, as a fraction of correct bets divided by all bets

        """  # fmt: skip
        self.prediction_result = prediction_result

        # Read and format odds df
        self.odds_df = self.process_odds_df()
        self.player_props = self.odds_df.columns.get_level_values(0).unique().to_list()

        # Compare predicted stats against odds to determine bets to place
        self.bets = self.place_bets()

        # Compare true stats against odds to determine results of bets
        self.bet_results = self.score_bets()

        self.earnings, self.accuracy = self.compute_performance()

    def process_odds_df(self):
        """Read master sheet of player prop odds, and format into DataFrame where each row matches the corresponding player/game in the prediction result dataset.

            Returns:
                pandas.DataFrame: DataFrame of odds (player prop) data for each player/game in the dataset being processed.

        """  # fmt: skip

        # Player/game IDs in dataset
        ids = self.prediction_result.dataset.id_data.set_index(["Player ID", "Year", "Week"]).index

        # Get odds from dataset misc data
        odds_df = self.prediction_result.dataset.misc_df.filter(like="odds_")

        # Set index and column labels
        odds_df.index = ids
        odds_df.columns = odds_df.columns.str.replace("odds_", "")

        # Configure multi-level columns
        odds_df.columns = pd.MultiIndex.from_tuples(odds_df.columns.str.split(r"\s+(?=[Over|Under])", regex=True).to_list())

        # Replace any 0 price values with NaN, since these are not valid lines (likely represent missing data)
        zero_over_price = odds_df.loc[:, (slice(None), "Over Price")] == 0
        odds_df[zero_over_price] = np.nan
        odds_df[zero_over_price.rename(columns={"Over Price": "Over Point"})] = np.nan
        zero_under_price = odds_df.loc[:, (slice(None), "Under Price")] == 0
        odds_df[zero_under_price] = np.nan
        odds_df[zero_under_price.rename(columns={"Under Price": "Under Point"})] = np.nan

        return odds_df

    def place_bets(self):
        """Uses predicted stats and odds over/under lines to decide whether to place a bet on each over or under.

            The BasicGambler class simply bets one unit on the over if the predicted stat > odds line, and one unit on the under if predicted stat < odds line.

            Note that the DataFrame returned from this method has a column for each player prop named "Over Units". This quantifies the bet being placed. A positive
            number is the amount of units bet on the over. A negative number is the amount of units bet on the under. Zero means no bet is placed on this player/player prop.

            Returns:
                pandas.DataFrame: DataFrame containing all bets placed on player props in the dataset, as well as their payouts for winning and losing.
                    The DataFrame columns are a multiindex, where the first level is the player prop (e.g. "Pass Yds")
                    and the second level contains "Over Units", "Win Earns", and "Loss Earns".

        """  # fmt: skip

        # Initialize dataframe holding index values from odds (players/games)
        bets = self.odds_df.index.to_frame(index=False)

        for player_prop in self.player_props:
            # Compute difference between prediction and line(s) (allowing for different lines to be set for over vs under, just in case)
            predict_diff_over_line = (
                self.prediction_result.predicts[player_prop] - self.odds_df.reset_index()[player_prop, "Over Point"]
            )
            predict_diff_under_line = (
                self.prediction_result.predicts[player_prop] - self.odds_df.reset_index()[player_prop, "Under Point"]
            )

            # Compute respective payouts for over and under
            over_earns = self.odds_df.reset_index()[player_prop, "Over Price"] - 1
            under_earns = self.odds_df.reset_index()[player_prop, "Under Price"] - 1

            # Betting algorithm: Bet one unit on over if predict > line, bet one unit on under if predict < line
            predict_over = (predict_diff_over_line > 0).astype(int)
            predict_under = -1 * (predict_diff_under_line < 0).astype(int)

            # Track bets placed and their outcomes for winning/losing
            units_bet = predict_over + predict_under
            bets[player_prop, "Over Units"] = units_bet
            bets[player_prop, "Win Earns"] = np.abs(units_bet) * (
                over_earns.mask(units_bet <= 0, 0) + under_earns.mask(units_bet >= 0, 0)
            )
            bets[player_prop, "Loss Earns"] = -1 * np.abs(units_bet)

        # Format bets dataframe
        bets = bets.set_index(self.odds_df.index.names)
        bets.columns = pd.MultiIndex.from_product(
            [self.player_props, ["Over Units", "Win Earns", "Loss Earns"]],
        )

        return bets

    def score_bets(self):
        """Determines the success and payout from each bet placed on a player prop.

            Returns:
                pandas.DataFrame: DataFrame containing the results of all bets placed on player props: whether the bet hit, and the earnings (positive or negative).
                    There is one row per bet placed.

        """  # fmt: skip

        bet_results = self.bets.index.to_frame(index=False)

        for player_prop in self.player_props:
            # Compare line to actual stat result to determine whether the over or under hit
            over_hit = (self.prediction_result.truths[player_prop] - self.odds_df.reset_index()[player_prop, "Over Point"]) > 0
            under_hit = (self.prediction_result.truths[player_prop] - self.odds_df.reset_index()[player_prop, "Under Point"]) < 0

            # Determine whether each bet placed was a success or not
            bet_hit = np.logical_or(
                np.logical_and(self.bets.reset_index()[player_prop, "Over Units"] > 0, over_hit),
                np.logical_and(self.bets.reset_index()[player_prop, "Over Units"] < 0, under_hit),
            )
            bet_results[(player_prop, "Hit")] = bet_hit

            # Compute earnings/losses based on whether bet hit
            bet_results[(player_prop, "Earnings")] = self.bets.reset_index()[player_prop, "Win Earns"].mask(
                np.logical_not(bet_hit),
                0,
            ) + self.bets.reset_index()[player_prop, "Loss Earns"].mask(bet_hit, 0)

        # Format bets dataframe
        bet_results = bet_results.set_index(self.bets.index.names)
        bet_results.columns = pd.MultiIndex.from_product([self.player_props, ["Hit", "Earnings"]])

        # Flatten to a row per bet, and keep only the bets placed (filtering out zeros)
        bet_results = self.__process_bet_results(bet_results_df=bet_results)

        return bet_results

    def compute_performance(self):
        """Outputs basic gambling performance metrics: total earnings, and accuracy percentage.

            Returns:
                float: Total gambling earnings (or losses, if negative) over the dataset
                float: Accuracy, as a fraction of correct bets divided by all bets

        """  # fmt: skip
        earnings = self.bet_results["Earnings"].sum()
        accuracy = self.bet_results["Hit"].sum() / self.bet_results.shape[0]

        return earnings, accuracy

    def plot_earnings(self):
        """Plots line graph tracking cumulative gambling earnings over each NFL week in the bet results."""  # fmt: skip
        plot_earnings_by_week(self.bet_results)

    def __process_bet_results(self, bet_results_df=None):
        # Handle optional input: default value is self.bet_results_df
        if bet_results_df is None:
            bet_results_df = self.bet_results
        # Flatten the dataframe, so that each individual bet has its own row, not a row per player/game
        flattened_df = pd.DataFrame()
        for player_prop in self.player_props:
            prop_df = bet_results_df[player_prop].copy()
            prop_df["Player Prop Stat"] = player_prop
            flattened_df = pd.concat((flattened_df, prop_df))

        # Remove rows that correspond to empty bets
        flattened_df = flattened_df[flattened_df["Earnings"] != 0]

        return flattened_df
