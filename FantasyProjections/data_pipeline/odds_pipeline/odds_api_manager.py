from __future__ import annotations

import logging

import pandas as pd
import requests

from config import data_files_config

# Set up logger
logger = logging.getLogger("log")


class OddsAPIManager:
    def __init__(self, surrogate: bool = False):
        self.api_key = self.get_odds_api_key()
        self.surrogate = surrogate

        # Track previous API requests
        self.all_requests_file = data_files_config.ODDS_REQUESTS_FILE  # .replace("requests", "test")
        self.all_requests_df = self.process_previous_requests()

    def process_previous_requests(self) -> pd.DataFrame:
        # Read the JSON file
        try:
            all_requests_df = pd.read_json(self.all_requests_file, orient="records", lines=True)
        except FileNotFoundError:
            # Set up an empty dataframe
            all_requests_df = pd.DataFrame(columns=["End Point", "Params", "Response Code", "Response Text"])

        # Filter out any duplicated requests (in case they were mistaken made twice)
        all_requests_df = all_requests_df.drop_duplicates(subset=["End Point", "Params"], keep="last")

        return all_requests_df

    def get_odds_api_key(self, filename=None):
        """Reads API key for The Odds API from a local file. API key must be obtained manually and saved to this file by the user.

            The file must be a text file containing ONLY the API key and no other text.

            Args:
                filename (str, optional): Path to text file containing API key. Defaults to using the filepath defined in data_files_config.

            Returns:
                str: The Odds API key read from the file.

        """  # fmt: skip

        # Handle default for optional input
        if filename is None:
            filename = data_files_config.ODDS_API_KEY_FILE

        with open(filename, encoding="utf-8") as file:
            api_key = file.readline()

        return api_key

    def make_api_request(self, endpoint: str, request_params: dict) -> tuple[str, bool]:
        # Check for an identical request in the database of previous requests
        params_string = str(dict(sorted(request_params.items())))
        try:
            response = str(
                self.all_requests_df.set_index(["End Point", "Params"]).loc[
                    (endpoint, params_string),
                    "Response Text",
                ],
            )
        except KeyError:
            pass
        else:
            # Response found in the database, so return it
            success = True
            return response, success

        # If surrogate turned on, skip making the API request and return surrogate data
        if self.surrogate:
            response_text = self.load_surrogate_response(endpoint, params=request_params)
            return response_text, True

        # Add API key to the request params
        request_params["api_key"] = self.api_key

        # Make the API request
        response = requests.get(endpoint, params=request_params, timeout=10)
        response_code = response.status_code
        response_text = response.text

        # Log API usage
        self.log_api_usage(response)

        # Handle successful/unsuccessful response codes
        success, msg = self.handle_response_codes(response_code)
        if msg:
            logger.warning(msg)

        # Store the successful request/result in the dataframe and file
        if success:
            new_request_df = pd.DataFrame(
                columns=self.all_requests_df.columns,
                data=[[endpoint, params_string, response_code, str(response_text)]],
            )
            self.all_requests_df = pd.concat((self.all_requests_df, new_request_df)).reset_index(drop=True)
            new_request_df.to_json(self.all_requests_file, orient="records", lines=True, mode="a")

        return response_text, success

    def load_surrogate_response(self, endpoint: str, params: dict) -> str:
        # Read file that matches the end point type (such as "events" or "odds")
        endpoint_type = endpoint.split("/")[-1]
        filename = data_files_config.ODDS_SURROGATE_DATA + f"{endpoint_type}.txt"
        try:
            with open(filename) as file:
                response_text = file.read()
        except FileNotFoundError as exc:
            msg = f"Surrogate data not supported for Odds API end point of type {endpoint_type}"
            raise NotImplementedError(msg) from exc

        return response_text

    def log_api_usage(self, response):
        """Logs the number of API requests fulfilled by The Odds with the current API key, and how many requests are remaining on the key.

            Args:
                response (requests.models.Response): Response from The Odds API, containing metadata on the number of requests made/available.

        """  # fmt: skip

        # Check API key usage
        logger.info(
            f"TheOddsAPI: {response.headers['x-requests-remaining']} requests left ({response.headers['x-requests-used']} used)",
        )

    def handle_response_codes(self, response_code: int) -> tuple[bool, str | None]:
        match response_code:
            case 200:
                # Successful
                success = True
                msg = None
            case 422:
                # Unprocessable Entity
                success = False
                msg = "TheOddsAPI returned code: Unprocessable Entity. This may be due to an invalid market."
            case _:
                success = False
                msg = "TheOddsAPI returned unknown code!"

        return success, msg
