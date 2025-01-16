
import logging
import numpy as np

# Set up logger
logger = logging.getLogger('log')


def find_matching_name_ind(name, others):

    if isinstance(others, str):
        others = [others]

    exact_matches = [name==other for other in others]

    try:
        index = exact_matches.index(True)
        return index
    except ValueError:
        pass

    fuzzy_matches = [fuzzy_match(name, other) for other in others]

    try:
        index = fuzzy_matches.index(True)
        return index
    except ValueError:
        return np.nan

def fuzzy_match(name1, name2):

    if name1 == name2:
        return True

    if drop_name_frills(name1) == drop_name_frills(name2):
        logger.debug(f'Fuzzy Name Match: {name1} == {name2}')
        return True

    return False

def drop_name_frills(name, expand_nicknames=True, lowercase=True):

    special_chars = ['.','-',"'"]
    unicode_quirks = {'Ã©':'é'}
    suffixes = ['Sr', 'Jr', 'III', 'II']
    nicknames_to_full_names = {'Mike ': 'Michael ',
                               'Rob ': 'Robert ',
                               'Tim ': 'Timothy ',
                               'Josh ': 'Joshua ',
                               'Chig ': 'Chigoziem ',
                               }

    for char in special_chars:
        name = name.replace(char, '')

    for quirk, correct in unicode_quirks.items():
        name = name.replace(quirk, correct)

    for suffix in suffixes:
        name = name.removesuffix(suffix)

    if expand_nicknames:
        for nickname, full_name in nicknames_to_full_names.items():
            name = name.replace(nickname, full_name)

    if lowercase:
        name = name.lower()

    name = name.strip()

    return name
