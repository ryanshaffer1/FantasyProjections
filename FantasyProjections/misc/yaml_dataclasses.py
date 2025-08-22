"""Classes used to store and process data input via YAML files.

    Classes:
        Flags : Class containing build_dataset flags to configure a run from a YAML.
        DatasetOptions : Class specifying what data to collect in a build_dataset run from a YAML.
        DataFilesConfig : Class pointing to a data files configuration YAML file and collecting its definitions.

"""  # fmt: skip

from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class Flags:
    """Class containing build_dataset flags to configure a run from a YAML.

        Args:
            save_data (bool, optional): Whether to save generated data to files.
            process_to_nn (bool, optional): Whether to complete neural net pre-processing.
            filter_roster (bool, optional): Whether to attempt data filtering from a pre-defined player list.
            update_filter (bool, optional): Whether to re-generate the roster filter based on data being collected.
            validate_parsing (bool, optional): Whether to perform independent verification of select data (depending on features collected).
            scrape_missing (bool, optional): Whether to collect any missing data required for validation from the internet.

    """  # fmt: skip

    save_data: bool = True
    process_to_nn: bool = True
    filter_roster: bool = True
    update_filter: bool = False
    validate_parsing: bool = False
    scrape_missing: bool = False


@dataclass
class DatasetOptions:
    """Class specifying what data to collect in a build_dataset run from a YAML.

        Args:
            team_names (str | list[str], optional): Team names/abbreviations to collect data for. Defaults to "all".
            years (list[int], optional): Seasons to collect data from. Defaults to None (no data collection occurs).
            weeks (list[int], optional): Weeks to collect data from in each season. Defaults to None (no data collection occurs).
            game_times (str | list[int]): List of times, in minutes of elapsed game time, to collect data for each game. Defaults to "all".

    """  # fmt: skip

    team_names: str | list[str] = "all"
    years: list[int] | None = None
    weeks: list[int] | None = None
    game_times: str | list[int] = "all"

    def __post_init__(self):
        if self.years is None:
            self.years = [2024]
        if self.weeks is None:
            self.weeks = list(range(1, 18))


@dataclass
class DataFilesConfig:
    """Class pointing to a data files configuration YAML file and collecting its definitions.

        Args:
            config_file (str): Path to the YAML file containing data files configuration settings.

        Additional Public Attributes:
            config (dict): All configuration settings loaded from the config_file.

    """  # fmt: skip

    config_file: str

    def __post_init__(self):
        with open(self.config_file) as file:
            self.config = yaml.safe_load(file)
