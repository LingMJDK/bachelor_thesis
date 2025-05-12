import os
from pathlib import Path

# This file contains the configuration for the project.
# It includes the paths to the data files and the URLs to download them.
DATA_DIR = Path("data")
UTILS_DIR = Path("utils")

# Url, filename and path for the IMDB dataset
IMDB_URL = "https://gist.githubusercontent.com/pbloem/2c3af77626d6c80f62487c35a28e3e8c/raw/5730d8f548a55a98b4415a457c5d827b33cd64d3/data_rnn.py"
IMDB_PY_SCRIPT = "data_rnn.py"

ENWIK8_PATH = Path(".data/external/enwik8")
