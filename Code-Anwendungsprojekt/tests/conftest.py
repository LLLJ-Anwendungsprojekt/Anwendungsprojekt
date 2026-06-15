"""Gemeinsame Fixtures: lädt die aufbereiteten Daten aus data/processed/."""

from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

# Indizes, die build_dataset.py behält
KEEP_INDICES = {
    "000001.SS", "399001.SZ", "GDAXI", "GSPTSE", "HSI", "IXIC",
    "KS11", "N100", "N225", "NYA", "SSMI", "TWII",
}


def _load(name):
    path = PROCESSED / name
    if not path.is_file():
        pytest.skip(f"{path} fehlt – zuerst src/build_dataset.py ausführen")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def daily():
    return _load("stocks_gpr_daily.csv")


@pytest.fixture(scope="session")
def monthly():
    return _load("stocks_gpr_monthly.csv")


@pytest.fixture(scope="session")
def features():
    return _load("stocks_gpr_features.csv")
