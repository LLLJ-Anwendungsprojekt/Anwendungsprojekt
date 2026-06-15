"""Sanity-Checks der Event-Study-Konfiguration in event_regression.py."""

import sys
from pathlib import Path

import pytest

# src/ importierbar machen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "lineare_regression"))

evt = pytest.importorskip("event_regression")


def test_shock_quantile_in_range():
    assert 0 < evt.SHOCK_Q < 1


def test_event_window_positive():
    assert evt.EVENT_WINDOW > 0


def test_ar_columns_match_window():
    # je ein AR-Wert pro Tag im Fenster [-W … +W]
    assert len(evt.AR_COLS) == 2 * evt.EVENT_WINDOW + 1
    assert len(evt.REL_DAYS) == len(evt.AR_COLS)


def test_period_order():
    assert evt.START < evt.END
