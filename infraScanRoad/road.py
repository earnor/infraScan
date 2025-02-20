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


def has_stdin_input():
    """Check if there's data available in sys.stdin (to detect GUI mode)."""
    return not sys.stdin.isatty()  # True if input is piped (GUI mode)

if __name__ == "__main__":
    logging.info("Starting InfraScanRoad")
    logging.debug("sys.argv: %s", sys.argv)

    try:
        if has_stdin_input():
            logging.info("Reading configuration from GUI (stdin)...")
            config_data = json.load(sys.stdin)
        else:
            logging.warning("No valid JSON received, using default configuration.")
            cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            with open(os.path.join(cwd, "GUI", "base_config.json"), "r") as f:
                config_data = json.load(f)
    except json.JSONDecodeError:
        logging.error("Failed to parse JSON. Using default configuration.")
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(cwd, "GUI", "base_config.json"), "r") as f:
            config_data = json.load(f)

    scanner = Road(config_data)
    scanner.run()