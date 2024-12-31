"""Creates and exports functions to gather data from pro-football-reference.com.

    Functions:
        scrape_box_score : Obtains box-score statistics (final player stats for a game) from pro-football-reference.com for all players on a given team in a given game.
"""

from datetime import datetime
from time import sleep
import requests
from bs4 import BeautifulSoup
import pandas as pd
from config import stats_config

# Timing input (global variable) used to prevent network overload/shutoff
REQ_WAIT_TIME = 2  # seconds between web scraper HTTP requests


def scrape_box_score(stats_html, team_abbrevs, last_req_time):
    """Obtains box-score statistics (final player stats for a game) from pro-football-reference.com for all players on a given team in a given game.

        Args:
            stats_html (str): HTML to page containing game stats on pro-football-reference.com.
            team_abbrevs (str | list of strs): Abbreviation(s) of the team(s) being processed (as used on pro-football-reference.com)

        Returns:
            pandas.DataFrame: Final stats for each player on the team who played in the game.
    """

    # Handle optional non-iterable type for team_abbrevs
    if not hasattr(team_abbrevs, '__iter__'):
        team_abbrevs = [team_abbrevs]

    # Gathers offensive stats from a game

    box_score_df = pd.DataFrame()

    # Make HTTP request (after waiting for the cooldown period)
    sleep_time = max(0, REQ_WAIT_TIME - (datetime.now() -
                     last_req_time).total_seconds())
    print(f'\t sleeping {sleep_time} seconds')
    sleep(sleep_time)  # Wait until enough time has passed
    r = requests.get(stats_html, timeout=10)  # Make the GET request
    last_req_time = datetime.now()  # Keep track of last time a GET request was made

    soup = BeautifulSoup(r.content, 'html.parser')
    stats_table = soup.find('table', {'id': 'player_offense'})
    stats_table_body = stats_table.find('tbody')
    rows = stats_table_body.find_all('tr')
    for row in rows:
        if row.find('th',
                    {'data-stat': 'player'}) is None or row.find('th',
                        {'data-stat': 'player'}).find('a') is None:
            continue
        if row.find('td', {'data-stat': 'team'}).text not in team_abbrevs:
            continue
        box_score = pd.Series()
        box_score['Player'] = row.find(
            'th', {'data-stat': 'player'}).find('a').text
        box_score['Team'] = row.find('td', {'data-stat': 'team'}).text
        # Collect all stats
        for (key, val) in stats_config.labels_df_to_pfr.items():
            box_score[key] = int(row.find('td', {'data-stat': val}).text)

        box_score_df = pd.concat([box_score_df, box_score], axis=1)

    # Format output df and add Fantasy Points
    box_score_df = box_score_df.transpose().set_index('Player')

    return box_score_df, last_req_time
