import logging
import os
import sys

# Ensure InfraScan is in sys.path for imports due to complex directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Define ANSI escape codes for colors
COLORS = {
    "TRACE": "\033[94m",    # Blue
    "VERBOSE": "\033[90m",  # Light Gray
    "NOTICE": "\033[93m",   # Yellow
    "WARNING": "\033[33m",  # Orange
    "ALERT": "\033[91m",    # Red
    "ERROR": "\033[31m",    # Bright Red
    "CRITICAL": "\033[95m", # Magenta
    "RAIL": "\033[96m",     # Cyan 🚆
    "ROAD": "\033[35m",     # Purple 🚗
    "RESET": "\033[0m"      # Reset color
}
# Define custom log levels
TRACE = 5
VERBOSE = 15
NOTICE = 25
ALERT = 35
RAIL = 45
ROAD = 46

logging.addLevelName(TRACE, "TRACE")
logging.addLevelName(VERBOSE, "VERBOSE")
logging.addLevelName(NOTICE, "NOTICE")
logging.addLevelName(ALERT, "ALERT")
logging.addLevelName(RAIL, "RAIL")
logging.addLevelName(ROAD, "ROAD")

# Extend the Logger class to add custom log methods
class CustomLogger(logging.Logger):
    def trace(self, msg, *args, **kwargs):
        msg = self._format_message(msg, args)
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, (), **kwargs)  # Pass empty tuple to avoid TypeError

    def verbose(self, msg, *args, **kwargs):
        msg = self._format_message(msg, args)
        if self.isEnabledFor(VERBOSE):
            self._log(VERBOSE, msg, (), **kwargs)

    def notice(self, msg, *args, **kwargs):
        msg = self._format_message(msg, args)
        if self.isEnabledFor(NOTICE):
            self._log(NOTICE, msg, (), **kwargs)

    def alert(self, msg, *args, **kwargs):
        msg = self._format_message(msg, args)
        if self.isEnabledFor(ALERT):
            self._log(ALERT, msg, (), **kwargs)

    def rail(self, msg, *args, **kwargs):
        msg = self._format_message(msg, args)
        if self.isEnabledFor(RAIL):
            self._log(RAIL, msg, (), **kwargs)

    def road(self, msg, *args, **kwargs):
        msg = self._format_message(msg, args)
        if self.isEnabledFor(ROAD):
            self._log(ROAD, msg, (), **kwargs)

    def _format_message(self, msg, args):
        """Ensures multiple arguments are joined as a single string before logging."""
        if args:
            return f"{msg} " + " ".join(map(str, args))
        return str(msg)

# Set the custom logger
logging.setLoggerClass(CustomLogger)
logger = logging.getLogger(__name__)

# Custom log formatter with colors
class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLORS.get(record.levelname, COLORS["RESET"])
        log_message = super().format(record)
        return f"{log_color}{log_message}{COLORS['RESET']}"

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Define log format
log_format = "%(levelname)s: %(message)s"
formatter = CustomFormatter(log_format)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)  # Capture all logs

# Export the logger for other scripts
__all__ = ["logger"]

def show_all_loggers():
    logger.trace("This is a TRACE message (5) - Lower-level debugging")
    logger.debug("This is a DEBUG message (10) - Standard debugging")
    logger.verbose("This is a VERBOSE message (15) - Extra debugging details")
    logger.info("This is an INFO message (20) - Everything is working")
    logger.notice("This is a NOTICE message (25) - Something worth noting")
    logger.warning("This is a WARNING message (30) - Potential issue")
    logger.alert("This is an ALERT message (35) - Important attention needed")
    logger.error("This is an ERROR message (40) - Serious issue")
    logger.critical("This is a CRITICAL message (50) - System failure")
    logger.rail("This is a RAIL message (45) - Rail system activity 🚆")
    logger.road("This is a ROAD message (46) - Road system activity 🚗")


if __name__ == "__main__":
    show_all_loggers()
