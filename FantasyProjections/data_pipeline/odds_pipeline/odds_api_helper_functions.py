"""Defines convenience functions for working with The Odds API.

    Functions:
        get_odds_api_key : Reads API key for The Odds API from a local file. API key must be obtained manually and saved to this file by the user.
        log_api_usage : Logs the number of API requests fulfilled by The Odds with the current API key, and how many requests are remaining on the key.
"""

import logging

from config import data_files_config

# Set up logger
logger = logging.getLogger("log")

# Constant variables used for The Odds API
SPORT_KEY = "americanfootball_nfl"
DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"
BOOKMAKER = "fanduel"

def get_odds_api_key(filename=None):
    """Reads API key for The Odds API from a local file. API key must be obtained manually and saved to this file by the user.

        The file must be a text file containing ONLY the API key and no other text.

        Args:
            filename (str, optional): Path to text file containing API key. Defaults to using the filepath defined in data_files_config.

        Returns:
            str: The Odds API key read from the file.
    """

    # Handle default for optional input
    if filename is None:
        filename = data_files_config.ODDS_API_KEY_FILE

    with open(filename, encoding="utf-8") as file:
        api_key = file.readline()

    return api_key


def log_api_usage(response):
    """Logs the number of API requests fulfilled by The Odds with the current API key, and how many requests are remaining on the key.

        Args:
            response (requests.models.Response): Response from The Odds API, containing metadata on the number of requests made/available.
    """
    # Check your usage
    logger.info(f"{response.headers['x-requests-remaining']} requests left ({response.headers['x-requests-used']} used)")
