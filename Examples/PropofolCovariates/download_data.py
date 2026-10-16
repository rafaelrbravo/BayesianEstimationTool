"""Download the three original upstream CSV files for the propofol example.

Run from any directory:
    python Examples/PropofolCovariates/download_data.py

Existing files are never overwritten. These data remain attributed to the
upstream authors and are not committed to BET.
"""
from pathlib import Path
from urllib.request import urlopen
import os
import tempfile

DATA_DIR = Path(__file__).resolve().parent / "data"
BASE_URL = "https://raw.githubusercontent.com/wahlquisty/learning-pharmacometric-covariate-structures/main/symbreg/data"
FILES = ("dataframe.csv", "eleveld_infusiondata.csv", "eleveld_modelparams.csv")


def download_data(directory=DATA_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        destination = directory / filename
        if destination.exists():
            if destination.stat().st_size == 0:
                raise ValueError(f"Existing upstream file is empty: {destination}")
            print(f"Already exists: {destination}")
            continue
        temp_path = None
        try:
            with urlopen(f"{BASE_URL}/{filename}", timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=directory, prefix=filename + ".", suffix=".part", delete=False) as output:
                    temp_path = Path(output.name)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
            if temp_path.stat().st_size == 0:
                raise ValueError(f"Empty upstream file: {filename}")
            os.replace(temp_path, destination)
            print(f"Downloaded: {destination} ({destination.stat().st_size:,} bytes)")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    download_data()
