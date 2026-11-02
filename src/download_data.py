"""Download the CarDekho 'car data.csv' file and verify its checksum.

Primary source (requires a free Kaggle account):
    https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
    -> file "car data.csv"

Kaggle downloads need authentication, so this script fetches a public GitHub
mirror of the same file and verifies its SHA-256 against the version this
project was developed and evaluated on. If the checksum differs, the file is
rejected so results stay reproducible.

Usage:
    python -m src.download_data            # mirror + checksum verification
    python -m src.download_data --force    # overwrite an existing file
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from src import config

MIRROR_URL = "https://raw.githubusercontent.com/krishnaik06/Car-Price-Prediction/master/car%20data.csv"
EXPECTED_SHA256 = "00b477def120548e583c638b00146982956b80646817901245ef78da9c4b3063"
TIMEOUT_SECONDS = 30


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(dest: Path = config.DATA_PATH, url: str = MIRROR_URL, force: bool = False) -> Path:
    if dest.exists() and not force:
        if sha256_of(dest) == EXPECTED_SHA256:
            print(f"Dataset already present and verified: {dest}")
            return dest
        raise RuntimeError(f"{dest} exists but its checksum does not match the expected "
                           "file. Re-run with --force to replace it.")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
        tmp.write_bytes(response.read())

    digest = sha256_of(tmp)
    if digest != EXPECTED_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch (got {digest}). The mirror may have changed; download "
            "'car data.csv' from Kaggle manually and save it as data/raw/car_data.csv."
        )
    tmp.replace(dest)
    print(f"Saved and verified {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    args = parser.parse_args()
    try:
        download(force=args.force)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report any failure cleanly
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
