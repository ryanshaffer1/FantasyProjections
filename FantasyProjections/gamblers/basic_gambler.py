import numpy as np
import pandas as pd
from config import data_files_config
from gamblers.gambling_plots import plot_earnings_by_week

class BasicGambler():
    def __init__(self, prediction_result, odds_file=None):

        # Optional input
        if  odds_file is None:
            odds_file = data_files_config.ODDS_FILE

        self.prediction_result = prediction_result
        self.odds_file = odds_file

        # Read and format odds df
        self.odds_df = self.process_odds_df()

        # Compare predicted stats against odds to determine bets to place
        self.bets = self.place_bets()

        # Compare true stats against odds to determine results of bets
        self.bet_results = self.score_bets()
        self.bet_results = self.__process_bet_results()

        self.earnings, self.accuracy = self.compute_performance()

    def process_odds_df(self):
        """Read master sheet of player prop odds, and format into DataFrame where each row matches the corresponding player/game in the prediction result dataset.

            Returns:
                pandas.DataFrame: DataFrame of odds (player prop) data for each player/game in the dataset being processed.
        """
        # Player/game IDs in dataset
        ids = self.prediction_result.dataset.id_data.copy().set_index(['Player ID','Year','Week'])

        # Read odds CSV
        odds_df = pd.read_csv(self.odds_file)

        # Configure dataset of odds that aligns with players/games in prediction result
        odds_df = odds_df.pivot(index=['Player ID','Year','Week'], columns='Player Prop Stat', values=['Over Point','Over Price','Under Point','Under Price','Elapsed Time'])
        odds_df = odds_df.swaplevel(axis=1).sort_index(axis=1)

        # Add NaNs in odds for any players/games in the dataset that are missing in odds
        missing_indices = ids.index.difference(odds_df.index)
        odds_df = pd.concat((odds_df,pd.DataFrame(index=missing_indices))).loc[ids.index]

        return odds_df


    def place_bets(self):
        # Initialize dataframe holding index values from odds (players/games)
        bets = self.odds_df.index.to_frame(index=False)

        for player_prop in ['Pass Yds', 'Rush Yds', 'Rec Yds']:
            # Compute difference between prediction and line(s) (allowing for different lines to be set for over vs under, just in case)
            predict_diff_over_line = self.prediction_result.predicts[player_prop] - self.odds_df.reset_index()[player_prop, 'Over Point']
            predict_diff_under_line = self.prediction_result.predicts[player_prop] - self.odds_df.reset_index()[player_prop, 'Under Point']

            # Compute respective payouts for over and under
            over_earns = (self.odds_df.reset_index()[player_prop, 'Over Price'] - 1)
            under_earns = (self.odds_df.reset_index()[player_prop, 'Under Price'] - 1)

            # Betting algorithm: Bet one unit on over if predict > line, bet one unit on under if predict < line
            predict_over = (predict_diff_over_line > 0).astype(int)
            predict_under = -1 * (predict_diff_under_line < 0).astype(int)

            # Track bets placed and their outcomes for winning/losing
            units_bet = predict_over + predict_under
            bets[player_prop, 'Over Units'] = units_bet
            bets[player_prop, 'Win Earns'] = np.abs(units_bet) * (over_earns.mask(units_bet <= 0, 0) + under_earns.mask(units_bet >= 0, 0))
            bets[player_prop, 'Loss Earns'] = -1 * np.abs(units_bet)

        # Format bets dataframe
        bets = bets.set_index(self.odds_df.index.names)
        bets.columns = pd.MultiIndex.from_product([['Pass Yds','Rush Yds','Rec Yds'], ['Over Units','Win Earns','Loss Earns']])

        return bets


    def score_bets(self):
        bet_results = self.bets.index.to_frame(index=False)

        for player_prop in ['Pass Yds', 'Rush Yds', 'Rec Yds']:
            # Compare line to actual stat result to determine whether the over or under hit
            over_hit = (self.prediction_result.truths[player_prop] - self.odds_df.reset_index()[player_prop, 'Over Point']) > 0
            under_hit = (self.prediction_result.truths[player_prop] - self.odds_df.reset_index()[player_prop, 'Under Point']) < 0

            # Determine whether each bet placed was a success or not
            bet_hit = np.logical_or(np.logical_and(self.bets.reset_index()[player_prop, 'Over Units'] > 0, over_hit),
                                    np.logical_and(self.bets.reset_index()[player_prop, 'Over Units'] < 0, under_hit))
            bet_results[(player_prop, 'Hit')] = bet_hit

            # Compute earnings/losses based on whether bet hit
            bet_results[(player_prop, 'Earnings')] = (self.bets.reset_index()[player_prop, 'Win Earns'].mask(np.logical_not(bet_hit), 0) +
                                                      self.bets.reset_index()[player_prop, 'Loss Earns'].mask(bet_hit, 0))

        # Format bets dataframe
        bet_results = bet_results.set_index(self.bets.index.names)
        bet_results.columns = pd.MultiIndex.from_product([['Pass Yds','Rush Yds','Rec Yds'], ['Hit','Earnings']])

        # Add column tracking all earnings across this player/game
        bet_results['Total Earnings'] = bet_results[[(player_prop, 'Earnings') for player_prop in ['Pass Yds','Rush Yds','Rec Yds']]].sum(axis=1)

        return bet_results


    def compute_performance(self):
        earnings = self.bet_results['Earnings'].sum()
        accuracy = self.bet_results['Hit'].sum() / self.bet_results.shape[0]

        return earnings, accuracy

    def plot_earnings(self):
        plot_earnings_by_week(self.bet_results)


    def __process_bet_results(self, bet_results_df = None):
        # Handle optional input: default value is self.bet_results_df
        if bet_results_df is None:
            bet_results_df = self.bet_results
        # Flatten the dataframe, so that each individual bet has its own row, not a row per player/game
        flattened_df = pd.DataFrame()
        for player_prop in ['Pass Yds', 'Rush Yds', 'Rec Yds']:
            prop_df = bet_results_df[player_prop].copy()
            prop_df['Player Prop Stat'] = player_prop
            flattened_df = pd.concat((flattened_df, prop_df))
        # flattened_df = flattened_df.reset_index().set_index(bet_results_df.index.names + ['Player Prop Stat'])

        # Remove rows that correspond to empty bets
        flattened_df = flattened_df[flattened_df['Earnings'] != 0]

        return flattened_df
