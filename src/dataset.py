import os
import requests


def get_data(data_dir: str, url: str, filename: str) -> str:
    """
    Downloads the data from the given URL if it is not already present in the data directory.
    Returns the path to the data file.
    Params:
    - data_dir: The directory where the data file will be stored.
    Returns:
    - The path to the data file.
    """
    # Create the data directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    # Path to the data file
    data_path = os.path.join(data_dir, filename)

    # Download the file if it does not exist
    if not os.path.exists(data_path):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(data_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {filename} to {data_path}")
    else:
        print(f"{filename} already exists. Skipping download.")
