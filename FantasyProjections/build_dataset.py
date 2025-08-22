"""Gathers data to be used in Fantasy Football stat prediction.

Pulls data from local files where possible, and from online repository when necessary (saving locally for next time). Parses inputs to determine the
play-by-play performance of all NFL players or a filtered subset (filtered by highest-scoring in Fantasy Football) over all games within a specified
time-frame. Optionally normalizes/pre-processes the data for use in a Neural Net predictor.

This module is a script to be run alone. Before running, the packages in requirements.txt must be installed. Use the following terminal command:
> pip install -u requirements.txt
"""  # fmt: skip

import argparse
import logging
import logging.config
from datetime import datetime

import pandas as pd
import yaml

from config.log_config import LOGGING_CONFIG
from data_pipeline.dataset_processor import DatasetProcessor
from data_pipeline.seasonal_data_collector import SeasonalDataCollector
from misc.manage_files import collect_roster_filter, create_folders, move_logfile, save_plots
from misc.yaml_constructor import add_yaml_constructors

# Set up logger
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")


def main(parameter_file: str):
    """Runs the build_dataset process, collecting, editing, and saving NFL stats/information for players before, during, and after their games.

        Args:
            parameter_file (str): File path to the configuration parameters (YAML file) for this run.

        Raises:
            ValueError: Invalid parameters entered, such as no year being specified.

    """  # fmt: skip

    add_yaml_constructors()
    with open(parameter_file) as file:
        inputs = yaml.safe_load(file)
    # Unpack inputs into variables
    flags = inputs["flags"]
    dataset_opts = inputs["dataset_options"]
    feature_sets = inputs["feature_sets"]
    data_files_config = inputs["data_files_config"]

    # Start
    start_time = datetime.now().astimezone()
    logger.info("Starting Program")

    # Log user inputs
    logger.debug(
        f"""FLAGS: \nSAVE_DATA={flags.save_data} \nPROCESS_TO_NN={flags.process_to_nn} \nFILTER_ROSTER={flags.filter_roster} \nUPDATE_FILTER={flags.update_filter}\
                \nVALIDATE_PARSING={flags.validate_parsing} \nSCRAPE_MISSING={flags.scrape_missing},
        DATA INPUTS: \nTEAM_NAMES={dataset_opts.team_names} \nYEARS={dataset_opts.years} \nWEEKS={dataset_opts.weeks} \nGAME_TIMES={dataset_opts.game_times}""",
    )

    # Check inputs
    if len(dataset_opts.years) == 0:
        msg = "No years specified. Specify at least one year to process."
        logger.error(msg)
        raise ValueError(msg)

    # Huge output data arrays
    midgame_df = pd.DataFrame()
    final_stats_df = pd.DataFrame()
    aux_data_df = pd.DataFrame()

    # Load/prepare optional roster filter
    roster_filter_file = data_files_config["roster_filter_file"] if flags.filter_roster else None
    roster_save_file = roster_filter_file if flags.save_data else None
    filter_df, filter_load_success = collect_roster_filter(flags.filter_roster, flags.update_filter, roster_filter_file)
    logger.info(f"Filter Load Success: {filter_load_success}; Filter file: {roster_filter_file}")

    # Finish initializing feature sets
    for feature in feature_sets:
        feature.post_init(data_files_config=data_files_config)

    # Process NFL data one year at a time
    for year in dataset_opts.years:
        logger.info(f"--------------- {year} ---------------")

        # Create SeasonalData object, which automatically processes all data for that year
        seasonal_data = SeasonalDataCollector(
            data_files_config=data_files_config,
            year=year,
            feature_sets=feature_sets,
            team_names=dataset_opts.team_names,
            weeks=dataset_opts.weeks,
            game_times=dataset_opts.game_times,
            filter_df=filter_df,
        )

        # Concatenate results from current year to remaining years
        midgame_df = pd.concat((midgame_df, seasonal_data.midgame_df))
        final_stats_df = pd.concat((final_stats_df, seasonal_data.final_stats_df))
        aux_data_df = pd.concat((aux_data_df, seasonal_data.all_game_info_df))

        logger.info(f"{year} processing complete.")
    logger.info(f"""Data collection complete.
                Total midgame data rows: {midgame_df.shape[0]}""")

    # Create dataset processor with all collected data
    processor = DatasetProcessor(
        data_files_config=data_files_config,
        feature_sets=feature_sets,
        midgame_df=midgame_df,
        final_stats_df=final_stats_df,
        aux_data_df=aux_data_df,
        **inputs.get("processor_config", {}),
    )

    # If roster filter could not be found/applied before processing,
    # generate a roster filter file now and apply it to the data
    if flags.filter_roster and ((not filter_load_success) or flags.update_filter):
        logger.info("Generating new filter.")
        processor.generate_roster_filter(seasonal_data.raw_rosters_df, save_file=roster_save_file)
        midgame_df, final_stats_df = processor.apply_roster_filter()

    # Optionally validate that parsed data matches data found from secondary sources (e.g. Pro-Football-Reference.com)
    if flags.validate_parsing:
        processor.validate_final_df(scrape_missing=flags.scrape_missing, save_data=flags.save_data)

    # Save collected data
    if flags.save_data:
        create_folders(data_files_config["output_folder"])
        midgame_df.to_csv(data_files_config["output_file_midgame"])
        logger.info(f"Saved midgame stats to {data_files_config['output_file_midgame']}.")
        final_stats_df.to_csv(data_files_config["output_file_final_stats"])
        logger.info(f"Saved final stats to {data_files_config['output_file_final_stats']}.")

    # Save plots
    if flags.save_data:
        logger.info("Saving plots")
        save_plots(data_files_config["validation_folder"])

    # End logging and move logfile to the correct folder
    logger.info(f"Program complete. Elapsed Time: {(datetime.now().astimezone() - start_time).total_seconds()} seconds")
    logging.shutdown()
    if flags.save_data:
        move_logfile("logfile.log", data_files_config["data_folder"])


# CONFIGURATION TO RUN FROM THE COMMAND LINE
def main_cli():
    """Run the build_dataset process from the command line with input arguments."""

    # Read command line argument: input parameter file
    parser = argparse.ArgumentParser("main")
    parser.add_argument("parameter_file", help="YAML file containing all dataset parameters - see example input files.")
    args = parser.parse_args()

    # Run the main function
    main(args.parameter_file)


if __name__ == "__main__":
    main_cli()
