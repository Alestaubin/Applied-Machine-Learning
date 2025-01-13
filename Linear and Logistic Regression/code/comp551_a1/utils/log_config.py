import logging
import sys

# Set the defult log level here
logging_level = logging.INFO

# Change the logging format here
logging.basicConfig(format='%(levelname)s - %(filename)s - %(lineno)d - %(asctime)s - %(message)s',
                    level=logging_level,
                    stream=sys.stdout)

# Create the logger for other classes to use
logger = logging.getLogger('simulation')