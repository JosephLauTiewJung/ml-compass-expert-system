import json
import unittest
from pathlib import Path

from inference_engine.inference_engine import infer_recommendation


ROOT_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_PATH = ROOT_DIR / "knowledge_base" / "knowledge_base.json"


def load_knowledge_base():
    with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


class InferenceEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge_base = load_knowledge_base()

    def infer(self, **overrides):
        payload = {
            "has_target_column": True,
            "target_type": "categorical",
            "target_unique_classes": 2,
            "can_have_multiple_labels": False,
            "data_type": "tabular",
            "text_complexity": "unknown",
            "dataset_rows": 5000,
            "class_balance": "balanced",
            "majority_class_percentage": 50,
            "missing_value_percentage": 0,
            "missing_value_type": "unknown",
            "has_outliers": False,
            "goal": "classification",
            "priority": "balanced",
            "domain": "general",
            "has_gpu": True,
            "timeline": "medium",
            "programming_skill": "intermediate",
            "output_can_be_calculated_by_formula": False,
        }
        payload.update(overrides)
        return infer_recommendation(payload, self.knowledge_base)

    def test_categorical_target_infers_classification(self):
        result = self.infer()

        self.assertEqual(result["problem_type"], "classification")
        self.assertEqual(result["classification_type"], "binary_classification")
        self.assertIn("Logistic Regression", result["recommended_baseline_model"])

    def test_numeric_label_uses_numeric_trap(self):
        result = self.infer(target_type="numeric_label", target_unique_classes=3)

        self.assertEqual(result["problem_type"], "classification")
        self.assertEqual(result["classification_type"], "multi_class_classification")
        self.assertTrue(any("Numeric labels" in warning for warning in result["warnings"]))

    def test_continuous_numeric_infers_regression(self):
        result = self.infer(
            target_type="continuous_numeric",
            target_unique_classes="unknown",
            goal="regression",
        )

        self.assertEqual(result["problem_type"], "regression")
        self.assertIn("Linear Regression", result["recommended_baseline_model"])
        self.assertIn("R2 Score", result["evaluation_metrics"])

    def test_time_series_forecasting_overrides_regression(self):
        result = self.infer(
            target_type="continuous_numeric",
            target_unique_classes="unknown",
            data_type="time_series",
            goal="forecasting",
        )

        self.assertEqual(result["problem_type"], "time_series_forecasting")
        self.assertIn("Moving Average", result["recommended_baseline_model"])

    def test_unlabeled_grouping_infers_clustering(self):
        result = self.infer(
            has_target_column=False,
            target_type="unknown",
            target_unique_classes="unknown",
            goal="grouping",
        )

        self.assertEqual(result["problem_type"], "clustering")
        self.assertIn("K-Means", result["recommended_baseline_model"])

    def test_anomaly_goal_infers_anomaly_detection(self):
        result = self.infer(
            has_target_column=False,
            target_type="unknown",
            target_unique_classes="unknown",
            goal="anomaly_detection",
        )

        self.assertEqual(result["problem_type"], "anomaly_detection")
        self.assertIn("Isolation Forest", result["recommended_advanced_model"])

    def test_small_dataset_blocks_deep_learning_first_choice(self):
        result = self.infer(
            data_type="text",
            text_complexity="complex",
            dataset_rows=300,
            goal="nlp",
            has_gpu=False,
            programming_skill="beginner",
        )

        self.assertEqual(result["problem_type"], "nlp")
        self.assertNotIn("BERT", result["recommended_advanced_model"])
        self.assertTrue(any("Small datasets" in warning for warning in result["warnings"]))

    def test_imbalanced_classification_uses_imbalance_metrics(self):
        result = self.infer(class_balance="imbalanced", majority_class_percentage=85)

        self.assertEqual(result["evaluation_metrics"], ["F1-Score", "Precision", "Recall", "PR-AUC"])

    def test_missing_dataset_size_gives_temporary_recommendation(self):
        result = self.infer(dataset_rows="unknown")

        self.assertEqual(result["confidence_level"], "medium")
        self.assertEqual(result["status"], "temporary_recommendation")


if __name__ == "__main__":
    unittest.main()
