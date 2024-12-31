from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from config import data_files_config, stats_config
from data_pipeline.scrape_pro_football_reference import scrape_box_score
from data_pipeline import team_abbreviations

def validate_parsed_data(final_stats_df, urls_df, scrape=False, save_data=False):
    last_req_time = datetime.now()

    # Pre-process input
    urls_df = urls_df.drop_duplicates(keep='first')

    # Read saved truth data
    true_df = pd.read_csv(data_files_config.TRUE_STATS_FILE).set_index(['Player','Year','Week'])

    # Compare saved truth data to input data to determine whether any saved truth data is missing
    missing_players = final_stats_df.index.difference(true_df.index)

    if scrape and len(missing_players) > 0:
        # Copy df before manipulating it, then filter to only the missing players
        missing_games = final_stats_df.copy().loc[missing_players].reset_index()

        # Gather unique game IDs that are missing
        missing_games['Game ID'] = missing_games.apply(construct_game_id, axis=1)
        missing_games = missing_games.drop_duplicates(subset=['Game ID'],keep='first')
        # Scrape the corresponding stat URLs
        print(f'Missing {len(missing_players)} players from {len(missing_games)} games')
        for i, (_, row) in enumerate(missing_games.iterrows()):
            team_abbrevs = team_abbreviations.convert_abbrev(row[['Team','Opponent']].to_list(),
                                                            team_abbreviations.pbp_abbrevs,
                                                            team_abbreviations.boxscore_website_abbrevs)
            game_url = urls_df.loc[row['Game ID'],'PFR URL']
            print(f'({i+1} of {len(missing_games)}) Scraping game {row['Game ID']} from {game_url}')
            boxscore, last_req_time = scrape_box_score(game_url, team_abbrevs, last_req_time)

            boxscore[['Year', 'Week']] = row.loc[['Year', 'Week']].to_list()
            boxscore = boxscore.reset_index().set_index(['Player','Year','Week'])
            true_df = pd.concat((true_df, boxscore))

        # Remove any unwanted duplicates from the resulting dataframe
        true_df = true_df.reset_index().drop_duplicates(keep='first').set_index(['Player','Year','Week'])
        # Save updated dataframe for use next time
        if save_data:
            true_df.to_csv(data_files_config.TRUE_STATS_FILE)

    # List of statistics to compute differences between truth and parsed data
    stats = stats_config.default_stat_list
    # Generate df of comparison (difference between truth and parsed)
    diff_df = __compare_dfs(true_df, final_stats_df, stats)

    # Optionally save dataframe to a csv
    if save_data:
        diff_df.to_csv(data_files_config.PARSING_VALIDATION_FILE)

    # Plot differences between truth and parsed data
    __plot_validation_comparison(diff_df, stats)
    plt.show()


def construct_game_id(final_stats_row):
    # Gather/format data from Series
    year = final_stats_row['Year']
    week = str(final_stats_row['Week']).rjust(2,'0')
    home_team = final_stats_row['Team'] if final_stats_row['Site'] == 'Home' else final_stats_row['Opponent']
    away_team = final_stats_row['Team'] if final_stats_row['Site'] == 'Away' else final_stats_row['Opponent']

    # Construct and return game_id string
    game_id = f'{year}_{week}_{away_team}_{home_team}'
    return game_id


def __compare_dfs(true_df, final_stats_df, stats):
    # Merge dataframes and compute differences in statistics
    merged_df = pd.merge(
        true_df.reset_index(), final_stats_df.reset_index(), how='inner', on=[
            'Player', 'Year', 'Week'])
    for stat in stats:
        merged_df[f'{stat}_diff'] = merged_df[f'{stat}_y'] - \
            merged_df[f'{stat}_x']  # Estimated minus truth
    diff_df = merged_df[['Player', 'Year', 'Week'] + [s + '_diff' for s in stats]]

    return diff_df


def __plot_validation_comparison(diff_df, stats):
    x = []
    for _ in range(diff_df.shape[0]):
        for i in range(len(stats)):
            x.append(i)
    y = diff_df[[s + '_diff' for s in stats]].stack()

    _, ax = plt.subplots(1, 1)
    ax.scatter(x, y)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats)
    for label in ax.get_xticklabels():
        label.set(rotation=45, horizontalalignment='right')
    plt.show(block=False)
