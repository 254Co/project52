# File: utils/logging_setup.py
import logging

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=level
    )