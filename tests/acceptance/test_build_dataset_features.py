import os

import pytest

# Module under test
import build_dataset

CONFIG_FILES_FOLDER = "tests/_test_files/config_files"


@pytest.mark.skip(reason="Feature not yet implemented")
def test_missing_features():
    parameter_file = os.path.join(CONFIG_FILES_FOLDER, "01_build_dataset_full.yaml")
    parameter_file = CONFIG_FILES_FOLDER + "/01_build_dataset_full.yaml"
    parameter_file = "inputs/build_dataset.yaml"
    build_dataset.main(parameter_file)
