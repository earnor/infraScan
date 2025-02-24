# import packages
import os
import sys
import math
import time
import logging
import json

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# Get the parent directory of GUI (i.e., InfraScan)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)  # Add InfraScan to Python's module search path
from logging_config import logger  # Import central logger

# Ensure this File is in `sys.path`
sys.path.insert(0, os.path.dirname(__file__))
import data_import
import voronoi_tiling
import scenarios
import plots
import generate_infrastructure
import scoring
import OSM_network
import traveltime_delay


# Main Class
class Road:
    # Class Variable
    # Constructor
    def __init__(self, config: dict):
        # Instance Variable
        self.config = config # Configuration JSON
        self.wd = os.path.join(self.config["General"].get("working_directory", ""), "InfraScanRoad") # Working Directory
        os.chdir(self.wd) # Change working directory

    def run(self):
        logger.road("Running InfraScanRoad")


def has_stdin_input():
    """Check if there's data available in sys.stdin (to detect GUI mode)."""
    return not sys.stdin.isatty()  # True if input is piped (GUI mode)

if __name__ == "__main__":
    logger.road("Starting InfraScanRoad")
    logger.road("sys.argv: %s", sys.argv)

    try:
        if has_stdin_input():
            logger.road("Reading configuration from GUI (stdin)...")
            config_data = json.load(sys.stdin)
        else:
            logger.road("No valid JSON received, using default configuration.")
            cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            with open(os.path.join(cwd, "GUI", "base_config.json"), "r") as f:
                config_data = json.load(f)
    except json.JSONDecodeError:
        logger.road("Failed to parse JSON. Using default configuration.")
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(cwd, "GUI", "base_config.json"), "r") as f:
            config_data = json.load(f)

    scanner = Road(config_data)
    scanner.run()