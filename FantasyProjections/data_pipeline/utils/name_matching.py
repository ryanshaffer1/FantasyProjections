"""Creates and exports helper functions related to identifying exact and approximate ("fuzzy") matches to names.

    Functions:
        find_matching_name_ind : Finds the index in a list of names that exactly, or approximately, matches the given name.
        fuzzy_match : Returns whether two names are roughly identical to each other, based on common name variations.
        drop_name_frills : Modifies an input name to just the "base" name with no modifiers included.
"""

import logging
import numpy as np

# Set up logger
logger = logging.getLogger('log')


def find_matching_name_ind(name, others):
    """Finds the index in a list of names that exactly, or approximately, matches the given name.

        Args:
            name (str): Name to look for in the list.
            others (list | str): Set of names to compare against input name (or optionally, one name)

        Returns:
            int | np.nan: Index in others of the exact or apprximate match to name. np.nan if no match was found.
    """

    # Handle only one name input for others
    if isinstance(others, str):
        others = [others]

    # Look for an exact match first
    exact_matches = [name==other for other in others]
    try:
        index = exact_matches.index(True)
        return index
    except ValueError:
        pass

    # Look for a "fuzzy match" next
    fuzzy_matches = [fuzzy_match(name, other) for other in others]
    try:
        index = fuzzy_matches.index(True)
        return index
    except ValueError:
        return np.nan


def fuzzy_match(name1, name2, log=True):
    """Returns whether two names are roughly identical to each other, based on common name variations.
    
        The variations checked are set within the function "drop_name_frills".

        Args:
            name1 (str): First name
            name2 (str): Second name
            log (bool, optional): Whether to log a debug message when a fuzzy match is found.

        Returns:
            bool: True if the names are exactly or approximately identical; else, False.
    """

    # Check for exact match first
    if name1 == name2:
        return True

    # Remove modifiers from both names and then check if they are exact matches
    if drop_name_frills(name1) == drop_name_frills(name2):
        # Optionally log that a fuzzy match was found
        if log:
            logger.debug(f'Fuzzy Name Match: {name1} == {name2}')
        return True

    return False


def drop_name_frills(name, expand_nicknames=True, lowercase=True):
    """Modifies an input name to just the "base" name with no modifiers included.
    
        Modifiers that can be removed include:
        - Nicknames (e.g. replace "Mike" with "Michael")
        - Special characters/punctuation (e.g. "B.J." becomes "BJ")
        - Suffixes (e.g. "Bob Smith Jr" becomes "Bob Smith")
        This function also fixes common unicode encoding issues (e.g. "Ã©"),
        and optionally returns the lowercase version of the name.

        Args:
            name (str): Name in raw format
            expand_nicknames (bool, optional): Whether to replace nicknames with common forms. Defaults to True.
            lowercase (bool, optional): Whether to return the name in all lowercase. Defaults to True.

        Returns:
            str: Name in "cleaned" format with modifiers removed (useful for fuzzy name matching).
    """

    # Changes to be made
    special_chars = ['.','-',"'"]
    unicode_quirks = {'Ã©':'é'}
    suffixes = ['Sr', 'Jr', 'III', 'II']
    nicknames_to_full_names = {'Mike ': 'Michael ',
                               'Rob ': 'Robert ',
                               'Tim ': 'Timothy ',
                               'Josh ': 'Joshua ',
                               'Chig ': 'Chigoziem ',
                               }

    # Remove special characters
    for char in special_chars:
        name = name.replace(char, '')

    # Replace unicode encoding errors with the correct unicode characters
    for quirk, correct in unicode_quirks.items():
        name = name.replace(quirk, correct)

    # Remove suffixes from name
    for suffix in suffixes:
        name = name.removesuffix(suffix)

    # Optionally replace nicknames with the corresponding full names
    if expand_nicknames:
        for nickname, full_name in nicknames_to_full_names.items():
            name = name.replace(nickname, full_name)

    # Optionally make the name lowercase
    if lowercase:
        name = name.lower()

    # Remove any extraneous whitespace left behind from other operations
    name = name.strip()

    return name
