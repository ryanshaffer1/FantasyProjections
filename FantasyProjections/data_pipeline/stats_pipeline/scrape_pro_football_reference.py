"""Creates and exports functions to gather data from pro-football-reference.com.

    Functions:
        scrape_box_score : Obtains box-score statistics (final player stats for a game) from pro-football-reference.com for all players on a given team in a given game.
        search_for_missing_pfr_id : Checks pro-football-reference.com for a player ID which corresponds to the input player name.
        scrape_player_page_for_name : Given a URL to a pro-football-reference.com player profile, returns the player's name.
"""  # fmt: skip

import logging
from datetime import datetime
from time import sleep

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import stats_config
from config.data_files_config import PFR_PLAYER_URL_INTRO
from data_pipeline.utils.name_matching import drop_name_frills, fuzzy_match

# Set up logger
logger = logging.getLogger("log")

REQ_WAIT_TIME = 2  # seconds between web scraper HTTP requests to avoid rate-limiting lockout by pro-football-reference


def scrape_box_score(stats_html, team_abbrevs, last_req_time):
    """Obtains box-score statistics (final player stats for a game) from pro-football-reference.com for all players on a given team in a given game.

        Args:
            stats_html (str): HTML to page containing game stats on pro-football-reference.com.
            team_abbrevs (str | list of strs): Abbreviation(s) of the team(s) being processed (as used on pro-football-reference.com)
            last_req_time (datetime.datetime): Timestamp of the last HTTP request made.
                Included so that REQ_WAIT_TIME is not exceeded (tripping rate limiting thresholds).

        Returns:
            pandas.DataFrame: Final stats for each player on the team who played in the game.
            datetime.datetime: Timestamp of the HTTP request made during the function call.

    """  # fmt: skip

    # Handle optional non-iterable type for team_abbrevs
    if not hasattr(team_abbrevs, "__iter__"):
        team_abbrevs = [team_abbrevs]

    # Gathers offensive stats from a game

    box_score_df = pd.DataFrame()

    # Make HTTP request (after waiting for the cooldown period)
    sleep_time = max(0, REQ_WAIT_TIME - (datetime.now().astimezone() - last_req_time).total_seconds())
    sleep(sleep_time)  # Wait until enough time has passed
    try:
        r = requests.get(stats_html, timeout=10)  # Make the GET request
    except requests.exceptions.ReadTimeout:
        logger.warning(f"Read Timeout on {stats_html}. No data obtained.")
        last_req_time = datetime.now().astimezone()  # Keep track of last time a GET request was made
        success = False
        return box_score_df, last_req_time, success

    last_req_time = datetime.now().astimezone()  # Keep track of last time a GET request was made
    success = True

    soup = BeautifulSoup(r.content, "html.parser")
    stats_table = soup.find("table", {"id": "player_offense"})
    stats_table_body = stats_table.find("tbody")
    rows = stats_table_body.find_all("tr")
    for row in rows:
        if row.find("th", {"data-stat": "player"}) is None or row.find("th", {"data-stat": "player"}).find("a") is None:
            continue
        if row.find("td", {"data-stat": "team"}).text not in team_abbrevs:
            continue
        box_score = pd.Series()
        box_score["Player Name"] = row.find("th", {"data-stat": "player"}).find("a").text
        box_score["pfr_id"] = row.find("th", {"data-stat": "player"})["data-append-csv"]
        box_score["Team"] = row.find("td", {"data-stat": "team"}).text
        # Collect all stats
        for key, val in stats_config.labels_df_to_pfr.items():
            box_score[key] = int(row.find("td", {"data-stat": val}).text)

        box_score_df = pd.concat([box_score_df, box_score], axis=1)

    # Format output df and add Fantasy Points
    box_score_df = box_score_df.transpose().reset_index(drop=True)

    return box_score_df, last_req_time, success


def search_for_missing_pfr_id(player_name, max_attempts=20, continue_on_404=False):
    """Checks pro-football-reference.com for a player ID which corresponds to the input player name.

        Generates a dict mapping each ID it checks to the corresponding player name,
        so that these pairs can be "cached" for future searches to reduce the number of HTTP requests needed.

        Args:
            player_name (str): Player name.
            max_attempts (int, optional): Max number of player IDs to try. Defaults to 20.
                Note that the search will also abort if it encounters an HTTPError (like 404/page not found).
            continue_on_404 (bool, optional): Whether to keep trying new pages after encountering a 404 error. Defaults to False.

        Returns:
            str | None: Player ID in PFR format, if a match was found. If no matching player name was found, returns None.
            dict: Dictionary mapping player names to player IDs in PFR format for all the pairs checked during the search.

    """  # fmt: skip

    last_req_time = datetime.now().astimezone()  # Keep track of last time a GET request was made

    # All PFR IDs have same format: LastFiXX where XX are numbers (e.g. Jayden Daniels -> DaniJa00)
    # Special characters like ' must be removed (performed by drop_name_frills)
    cleaned_name = drop_name_frills(player_name, expand_nicknames=False, lowercase=False)
    pfr_id_base = cleaned_name.split(" ")[1][:4].ljust(4, "x") + player_name.split(" ")[0][:2]

    # Try multiple PFR IDs to find one that has a matching player name
    i = 0
    names_to_pfr_ids = {}  # Running dict pairing names to IDs
    while i < max_attempts:
        # Construct current attempted player ID and associated URL
        curr_pfr_id = pfr_id_base + str(i).rjust(2, "0")
        player_url = f"{PFR_PLAYER_URL_INTRO}{curr_pfr_id[0]}/{curr_pfr_id}.htm"

        # Scrape Pro Football Reference for player name associated with current pfr_id
        scraped_name, last_req_time = scrape_player_page_for_name(player_url, last_req_time)
        if scraped_name is None:
            # Reached a 404 error before finding the name
            if continue_on_404:
                i += 1
                continue
            # If not continue on 404, return None
            return None, names_to_pfr_ids

        names_to_pfr_ids[scraped_name] = curr_pfr_id  # Add pair of name and ID to running dict
        if fuzzy_match(scraped_name, player_name):
            # Match!
            return curr_pfr_id, names_to_pfr_ids

        # Try the next ID
        i += 1

    # Exited loop without finding correct ID
    return None, names_to_pfr_ids


def scrape_player_page_for_name(player_url, last_req_time):
    """Given a URL to a pro-football-reference.com player profile, returns the player's name.

        Args:
            player_url (str): URL to parse for a player name.
            last_req_time (datetime.datetime): Timestamp of the last HTTP request made.
                Included so that REQ_WAIT_TIME is not exceeded (tripping rate limiting thresholds).

        Returns:
            str | None: Player name, if found. If HTTPError is raised (like 404 error), return None.
            datetime.datetime: Timestamp of the HTML request made during the function call.

    """  # fmt: skip

    sleep_time = max(0, REQ_WAIT_TIME - (datetime.now().astimezone() - last_req_time).total_seconds())
    sleep(sleep_time)  # Wait until enough time has passed
    last_req_time = datetime.now().astimezone()  # Keep track of last time a GET request was made

    try:
        r = requests.get(player_url, timeout=10)  # Make the GET request
    except requests.exceptions.ReadTimeout:
        return None, last_req_time

    # Check if HTTPError
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        scraped_name = None
        return scraped_name, last_req_time

    # Parse HTML to find player name
    soup = BeautifulSoup(r.content, "html.parser")
    try:
        scraped_name = soup.find("div", {"id": "meta"}).find("h1").find("span").text
    except AttributeError:
        # In some cases there is no span inside the h1
        scraped_name = soup.find("div", {"id": "meta"}).find("h1").text

    return scraped_name, last_req_time
