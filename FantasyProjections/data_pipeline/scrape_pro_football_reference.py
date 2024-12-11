"""Creates and exports functions to gather data from pro-football-reference.com.

    Functions:
        scrape_box_score : Obtains box-score statistics (final player stats for a game) from pro-football-reference.com for all players on a given team in a given game.
"""

from datetime import datetime
from time import sleep
import requests
from bs4 import BeautifulSoup
import pandas as pd
import team_abbreviations
from misc.stat_utils import stats_to_fantasy_points

# Timing inputs (global variables) used to prevent network overload/shutoff
last_req_time = datetime.now()
REQ_WAIT_TIME = 2  # seconds between web scraper HTTP requests


def scrape_box_score(stats_html, full_team_name):
    """Obtains box-score statistics (final player stats for a game) from pro-football-reference.com for all players on a given team in a given game.

        Args:
            stats_html (str): HTML to page containing game stats on pro-football-reference.com.
            full_team_name (str): Name of the team to process.

        Returns:
            pandas.DataFrame: Final stats for each player on the team who played in the game.
    """

    # Gathers offensive stats from a game

    box_score_df = pd.DataFrame()

    # Make HTTP request (after waiting for the cooldown period)
    global last_req_time
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
        if row.find('td', {'data-stat': 'team'}
                    ).text != team_abbreviations.boxscore_website_abbrevs[full_team_name]:
            continue
        box_score = pd.Series()
        box_score['Player'] = row.find(
            'th', {'data-stat': 'player'}).find('a').text
        box_score['Team'] = row.find('td', {'data-stat': 'team'}).text
        box_score['Pass Att'] = int(
            row.find('td', {'data-stat': 'pass_att'}).text)
        box_score['Pass Cmp'] = int(
            row.find('td', {'data-stat': 'pass_cmp'}).text)
        box_score['Pass Yds'] = int(
            row.find('td', {'data-stat': 'pass_yds'}).text)
        box_score['Pass TD'] = int(
            row.find('td', {'data-stat': 'pass_td'}).text)
        box_score['Int'] = int(row.find('td', {'data-stat': 'pass_int'}).text)
        box_score['Rush Att'] = int(
            row.find('td', {'data-stat': 'rush_att'}).text)
        box_score['Rush Yds'] = int(
            row.find('td', {'data-stat': 'rush_yds'}).text)
        box_score['Rush TD'] = int(
            row.find('td', {'data-stat': 'rush_td'}).text)
        box_score['Rec'] = int(row.find('td', {'data-stat': 'rec'}).text)
        box_score['Rec Yds'] = int(
            row.find('td', {'data-stat': 'rec_yds'}).text)
        box_score['Rec TD'] = int(row.find('td', {'data-stat': 'rec_td'}).text)
        box_score['Fmb'] = int(
            row.find(
                'td', {
                    'data-stat': 'fumbles_lost'}).text)

        box_score_df = pd.concat([box_score_df, box_score], axis=1)

    # Format output df and add Fantasy Points
    box_score_df = stats_to_fantasy_points(box_score_df.transpose().set_index('Player'))

    return box_score_df
