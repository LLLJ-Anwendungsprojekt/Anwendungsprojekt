"""Integritätschecks: keine Lücken, keine Duplikate, erwarteter Zeitraum & Indizes."""

import pandas as pd

from conftest import KEEP_INDICES


def test_gpr_columns_have_no_nan(daily):
    assert daily[["GPRD", "GPRD_ACT", "GPRD_THREAT"]].isna().sum().sum() == 0


def test_no_duplicate_index_date(daily):
    assert daily.duplicated(["Index", "Date"]).sum() == 0


def test_date_range(daily):
    dates = pd.to_datetime(daily["Date"])
    assert dates.min() >= pd.Timestamp("2000-01-01")
    assert dates.max() <= pd.Timestamp("2021-05-31")


def test_only_expected_indices(daily):
    assert set(daily["Index"].unique()) <= KEEP_INDICES


def test_features_have_no_nan(features):
    # build_dataset.py entfernt am Ende alle NaN-Zeilen (Lags etc.)
    assert features.isna().sum().sum() == 0
