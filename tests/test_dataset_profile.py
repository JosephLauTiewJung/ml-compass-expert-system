import unittest

import pandas as pd

from dataset_profile import profile_dataset


class DatasetProfileTest(unittest.TestCase):
    def test_profiles_dataset_wide_defaults(self):
        df = pd.DataFrame(
            {
                "age": [21, 22, None, 24],
                "income": [100, 120, 130, 140],
                "label": ["yes", "no", "yes", "yes"],
            }
        )

        profile, auto_fields = profile_dataset(df)

        self.assertEqual(profile["data_type"], "tabular")
        self.assertEqual(profile["dataset_rows"], 4)
        self.assertEqual(profile["missing_value_percentage"], 8)
        self.assertEqual(profile["missing_value_type"], "numerical")
        self.assertIn("dataset_rows", auto_fields)

    def test_detects_categorical_missing_values(self):
        df = pd.DataFrame({"city": ["A", None, "B"], "score": [1, 2, 3]})

        profile, _ = profile_dataset(df)

        self.assertEqual(profile["missing_value_type"], "categorical")

    def test_detects_mixed_missing_values(self):
        df = pd.DataFrame({"city": ["A", None, "B"], "score": [1, None, 3]})

        profile, _ = profile_dataset(df)

        self.assertEqual(profile["missing_value_type"], "mixed")

    def test_profiles_target_balance(self):
        df = pd.DataFrame(
            {
                "feature": [10, 11, 12, 13, 14],
                "target": ["yes", "yes", "yes", "yes", "no"],
            }
        )

        profile, auto_fields = profile_dataset(df, target_column="target")

        self.assertTrue(profile["has_target_column"])
        self.assertEqual(profile["target_type"], "categorical")
        self.assertEqual(profile["target_unique_classes"], 2)
        self.assertEqual(profile["majority_class_percentage"], 80)
        self.assertEqual(profile["class_balance"], "balanced")
        self.assertIn("majority_class_percentage", auto_fields)

    def test_numeric_target_with_few_unique_values_is_label(self):
        df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 0, 1]})

        profile, _ = profile_dataset(df, target_column="target")

        self.assertEqual(profile["target_type"], "numeric_label")
        self.assertEqual(profile["target_unique_classes"], 2)

    def test_continuous_numeric_target_keeps_class_count_unknown(self):
        df = pd.DataFrame(
            {
                "feature": range(100),
                "target": [value * 1.5 for value in range(100)],
            }
        )

        profile, _ = profile_dataset(df, target_column="target")

        self.assertEqual(profile["target_type"], "continuous_numeric")
        self.assertEqual(profile["target_unique_classes"], "unknown")

    def test_detects_important_outliers(self):
        df = pd.DataFrame({"feature": [10] * 100 + [1000, 1001, 1002], "target": [0] * 103})

        profile, _ = profile_dataset(df, target_column="target")

        self.assertTrue(profile["has_outliers"])

    def test_leaves_outliers_unknown_without_numeric_features(self):
        df = pd.DataFrame({"feature": ["a", "b", "c"], "target": ["x", "y", "x"]})

        profile, _ = profile_dataset(df, target_column="target")

        self.assertEqual(profile["has_outliers"], "unknown")

    def test_no_target_selection_leaves_target_fields_absent(self):
        df = pd.DataFrame({"feature": [1, 2, 3], "target": ["x", "y", "x"]})

        profile, _ = profile_dataset(df)

        self.assertNotIn("majority_class_percentage", profile)
        self.assertNotIn("target_type", profile)


if __name__ == "__main__":
    unittest.main()
