"""Download the three original upstream CSV files for the propofol example.

Run from any directory:
    python Examples/PropofolCovariates/download_data.py

Downloads are pinned to the same upstream commit as the symbolic-model export.
Existing files are never overwritten. These data remain attributed to the
upstream authors and are not committed to BET.
"""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import os
import tempfile

DATA_DIR = Path(__file__).resolve().parent / "data"
UPSTREAM_COMMIT = "939d30699fa6157c14c601712882c51106b82a24"
BASE_URL = f"https://raw.githubusercontent.com/wahlquisty/learning-pharmacometric-covariate-structures/{UPSTREAM_COMMIT}/symbreg/data"
FILES = ("dataframe.csv", "eleveld_infusiondata.csv", "eleveld_modelparams.csv")


def download_data(directory=DATA_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        destination = directory / filename
        temp_path = None
        try:
            # Fetch the immutable upstream revision even for cached files: this
            # detects stale or modified local CSVs before inference.
            remote_hash = hashlib.sha256()
            with urlopen(f"{BASE_URL}/{filename}", timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=directory, prefix=filename + ".", suffix=".part", delete=False) as output:
                    temp_path = Path(output.name)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        remote_hash.update(chunk)
                        output.write(chunk)
            if temp_path.stat().st_size == 0:
                raise ValueError(f"Empty upstream file: {filename}")
            if destination.exists():
                local_hash = hashlib.sha256()
                with destination.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        local_hash.update(chunk)
                if local_hash.digest() != remote_hash.digest():
                    raise ValueError(f"Cached upstream file differs from pinned commit {UPSTREAM_COMMIT}: {destination}")
                print(f"Verified cached file: {destination} (SHA-256 {remote_hash.hexdigest()})")
            else:
                os.replace(temp_path, destination)
                print(f"Downloaded: {destination} ({destination.stat().st_size:,} bytes; SHA-256 {remote_hash.hexdigest()})")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

if __name__ == "__main__":
    download_data()
