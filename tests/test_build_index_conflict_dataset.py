import unittest

import pandas as pd

from src.build_index_conflict_dataset import aggregate_conflicts_by_date


class TestBuildIndexConflictDataset(unittest.TestCase):
    def test_aggregate_conflicts_by_date_filters_and_sums(self):
        ged_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "date": pd.to_datetime(["2021-01-04", "2021-01-04", "2021-01-05"]),
                "type_of_violence": [1, 2, 1],
                "number_of_sources": [2, float("nan"), 1],
                "country_id": [100, 101, 100],
                "region": ["Africa", "Africa", "Asia"],
                "event_clarity": [1, 2, 1],
                "where_prec": [1, 2, 1],
                "deaths_civilians": [3, 1, 0],
                "deaths_unknown": [0, 2, 0],
                "best": [5, 10, 1],
                "high": [7, 12, 1],
                "low": [4, 8, 1],
                "total_deaths": [5, 12, 1],
                "conflict_duration_days": [0, 2, 1],
            }
        )

        result = aggregate_conflicts_by_date(ged_df)
        day_one = result.loc[result["date"] == pd.Timestamp("2021-01-04")].iloc[0]

        self.assertEqual(day_one["conflict_count"], 2)
        self.assertEqual(day_one["fatalities_best_sum"], 15)
        self.assertEqual(day_one["countries_affected"], 2)
        self.assertEqual(day_one["type_of_violence_1_count"], 1)
        self.assertEqual(day_one["type_of_violence_2_count"], 1)
        self.assertAlmostEqual(day_one["avg_sources"], 2.0)
        self.assertAlmostEqual(day_one["share_high_clarity"], 0.5)
        self.assertAlmostEqual(day_one["share_high_where_prec"], 0.5)


if __name__ == "__main__":
    unittest.main()