from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class Flags:
    save_data: bool = True
    process_to_nn: bool = True
    filter_roster: bool = True
    update_filter: bool = False
    validate_parsing: bool = False
    scrape_missing: bool = False


@dataclass
class DatasetOptions:
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
    config_file: str

    def __post_init__(self):
        with open(self.config_file) as file:
            self.config = yaml.safe_load(file)
