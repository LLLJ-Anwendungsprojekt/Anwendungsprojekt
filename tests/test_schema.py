"""Prüft, dass die aufbereiteten Dateien die erwarteten Spalten enthalten."""

DAILY_COLS = {"Index", "Date", "Close", "GPRD", "GPRD_ACT", "GPRD_THREAT", "YearMonth"}
MONTHLY_COLS = {"Index", "YearMonth", "Close_last", "Stock_monthly_pct",
                "GPRD_mean", "GPRD_monthly_pct"}
FEATURE_COLS = {"gpr_lag1", "stock_lag1", "stock_vol6", "gprd_zscore",
                "gpr_spike", "stock_down", "gpr_up_next"}


def test_daily_columns(daily):
    assert DAILY_COLS <= set(daily.columns)


def test_monthly_columns(monthly):
    assert MONTHLY_COLS <= set(monthly.columns)


def test_feature_columns(features):
    # Monatsspalten + Feature-Spalten
    assert (MONTHLY_COLS | FEATURE_COLS) <= set(features.columns)
