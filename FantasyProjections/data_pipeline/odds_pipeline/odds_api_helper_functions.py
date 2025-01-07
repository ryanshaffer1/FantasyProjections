
import logging

# Set up logger
logger = logging.getLogger('log')

# Constant variables used for The Odds API
SPORT_KEY = 'americanfootball_nfl'
DATE_FMT = '%Y-%m-%dT%H:%M:%SZ'
BOOKMAKER = 'fanduel'

def get_odds_api_key(filename=None):

    # Handle default for optional input
    if filename is None:
        filename = 'FantasyProjections/config/private/odds_api_key.txt'

    with open(filename, encoding='utf-8') as file:
        api_key = file.readline()

    return api_key


def log_api_usage(response):
    # Check your usage
    logger.info(f'\n{response.headers['x-requests-remaining']} requests left ({response.headers['x-requests-used']} used)')
