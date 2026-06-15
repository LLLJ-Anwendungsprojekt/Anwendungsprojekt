"""Prüft, dass die abgeleiteten Label-Spalten korrekt gesetzt sind."""

BINARY_COLS = ["gpr_spike", "stock_down", "gpr_up_next"]


def test_labels_are_binary(features):
    for col in BINARY_COLS:
        assert set(features[col].unique()) <= {0, 1}


def test_stock_down_matches_sign(features):
    # stock_down == 1 genau dann, wenn die Monatsrendite negativ ist
    expected = (features["Stock_monthly_pct"] < 0).astype(int)
    assert (features["stock_down"] == expected).all()


def test_pct_change_grouped_per_index(monthly):
    # Renditen dürfen nicht über Indexgrenzen hinweg berechnet werden:
    # je Index neu berechnet muss exakt der gespeicherten Spalte entsprechen.
    recomputed = monthly.groupby("Index")["Close_last"].pct_change() * 100
    valid = monthly["Stock_monthly_pct"].notna() & recomputed.notna()
    assert (recomputed[valid] - monthly["Stock_monthly_pct"][valid]).abs().max() < 1e-6
