# app/logging_config.py

import logging

def setup_logging():
    """
    Configures logging for the application.
    """
    logging.basicConfig(
        filename='app.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
