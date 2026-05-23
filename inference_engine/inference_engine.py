from copy import deepcopy


UNKNOWN = "unknown"
SENSITIVE_DOMAINS = {"medical", "legal", "financial", "educational"}
DEEP_LEARNING_MODELS = {"CNN", "ResNet", "Vision Transformer", "LSTM", "GRU", "BERT", "Transformer", "Autoencoder"}
SCALE_REQUIRED_MARKERS = ("Logistic Regression", "Linear Regression", "SVM", "KNN", "Neural", "LSTM", "GRU")
TREE_MODEL_MARKERS = ("Decision Tree", "Random Forest", "XGBoost", "LightGBM")

STARTER_CODE_CATALOG = {
    "classification": """from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions, average="weighted"))""",
    "regression": """from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions, squared=False))
print("R2:", r2_score(y_test, predictions))""",
    "clustering": """from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

features = df.drop(columns=["id"], errors="ignore")
scaled_features = StandardScaler().fit_transform(features)

model = KMeans(n_clusters=3, random_state=42, n_init="auto")
clusters = model.fit_predict(scaled_features)

print("Silhouette Score:", silhouette_score(scaled_features, clusters))""",
    "anomaly_detection": """from sklearn.ensemble import IsolationForest

features = df.drop(columns=["id"], errors="ignore")

model = IsolationForest(contamination="auto", random_state=42)
labels = model.fit_predict(features)

df["anomaly_label"] = labels
print(df["anomaly_label"].value_counts())""",
    "time_series_forecasting": """from sklearn.metrics import mean_absolute_error

series = df.sort_values("date")["target"]
window = 7
predictions = series.shift(1).rolling(window=window).mean()

valid = predictions.notna()
print("MAE:", mean_absolute_error(series[valid], predictions[valid]))""",
    "nlp": """from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X = df["text"]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions, average="weighted"))""",
}


def normalize_input(raw_form_data):
    """Convert UI answers into the knowledge-base input schema."""
    data = dict(raw_form_data)
    normalized = {
        "has_target_column": _coerce_unknown_bool(data.get("has_target_column", UNKNOWN)),
        "target_type": data.get("target_type", UNKNOWN),
        "target_unique_classes": _coerce_optional_number(data.get("target_unique_classes")),
        "can_have_multiple_labels": _coerce_unknown_bool(data.get("can_have_multiple_labels", UNKNOWN)),
        "data_type": data.get("data_type", UNKNOWN),
        "text_complexity": data.get("text_complexity", UNKNOWN),
        "dataset_rows": _coerce_optional_number(data.get("dataset_rows")),
        "class_balance": data.get("class_balance", UNKNOWN),
        "majority_class_percentage": _coerce_optional_number(data.get("majority_class_percentage")),
        "missing_value_percentage": _coerce_optional_number(data.get("missing_value_percentage")),
        "missing_value_type": data.get("missing_value_type", UNKNOWN),
        "has_outliers": _coerce_unknown_bool(data.get("has_outliers", UNKNOWN)),
        "goal": data.get("goal", UNKNOWN),
        "priority": data.get("priority", UNKNOWN),
        "domain": data.get("domain", UNKNOWN),
        "has_gpu": _coerce_unknown_bool(data.get("has_gpu", UNKNOWN)),
        "timeline": data.get("timeline", UNKNOWN),
        "programming_skill": data.get("programming_skill", UNKNOWN),
        "output_can_be_calculated_by_formula": _coerce_unknown_bool(
            data.get("output_can_be_calculated_by_formula", UNKNOWN)
        ),
    }

    if normalized["data_type"] == "text" and normalized["goal"] == UNKNOWN:
        normalized["goal"] = "nlp"
    if normalized["data_type"] == "time_series" and normalized["goal"] == UNKNOWN:
        normalized["goal"] = "forecasting"

    return normalized


def infer_recommendation(user_input, knowledge_base):
    """Infer a structured ML recommendation from normalized user input."""
    facts = normalize_input(user_input)
    result = deepcopy(knowledge_base.get("default_output_template", {}))
    result.setdefault("recommended_baseline_model", [])
    result.setdefault("recommended_advanced_model", [])
    result.setdefault("preprocessing_steps", [])
    result.setdefault("evaluation_metrics", [])
    result.setdefault("warnings", [])

    if facts["output_can_be_calculated_by_formula"] is True:
        result.update(
            {
                "problem_type": "rule_based_logic",
                "confidence_level": "high",
                "status": "approved",
                "explanation": "Machine learning is not needed because the output follows a direct formula or fixed rule.",
                "recommended_baseline_model": ["Rule-Based Logic", "Standard Statistics"],
                "recommended_advanced_model": [],
                "preprocessing_steps": ["Validate the formula or business rule with sample cases."],
                "evaluation_metrics": ["Rule accuracy on known examples"],
                "starter_code_suggestion": "# Implement the known formula or rule directly instead of training a model.",
            }
        )
        return result

    problem_type, task_reason, warnings = _infer_problem_type(facts)
    result["problem_type"] = problem_type
    result["warnings"].extend(warnings)

    classification_type = _infer_classification_type(facts, problem_type)
    if classification_type:
        result["classification_type"] = classification_type

    baseline, advanced, model_warnings = _select_models(facts, knowledge_base, problem_type)
    result["recommended_baseline_model"] = baseline
    result["recommended_advanced_model"] = advanced
    result["warnings"].extend(model_warnings)

    result["evaluation_metrics"] = _select_metrics(facts, knowledge_base, problem_type)
    result["preprocessing_steps"] = _select_preprocessing(facts, baseline + advanced)
    result["confidence_level"] = _select_confidence(facts, problem_type, result["warnings"])
    result["status"] = _select_status(facts, result["confidence_level"])
    result["explanation"] = _build_explanation(facts, problem_type, task_reason, result["confidence_level"])
    result["starter_code_suggestion"] = STARTER_CODE_CATALOG.get(problem_type, "# Add a starter template for this task.")

    result["warnings"] = _dedupe(result["warnings"])
    result["evaluation_metrics"] = _dedupe(result["evaluation_metrics"])
    result["preprocessing_steps"] = _dedupe(result["preprocessing_steps"])
    return result


def _infer_problem_type(facts):
    warnings = []
    goal = facts["goal"]
    data_type = facts["data_type"]
    has_target = facts["has_target_column"]
    target_type = facts["target_type"]

    if goal == "anomaly_detection":
        return "anomaly_detection", "The goal is to identify rare or abnormal cases.", warnings

    if data_type == "time_series" and goal in {"forecasting", UNKNOWN}:
        return "time_series_forecasting", "The data is ordered by time and the goal is future prediction.", warnings

    if data_type == "text" or goal == "nlp":
        return "nlp", "The project depends on understanding or processing human text.", warnings

    if has_target is False:
        if goal == "grouping":
            return "clustering", "There is no target column and the goal is to group similar data.", warnings
        warnings.append("No target column was provided, so the safest unsupervised starting point is clustering.")
        return "clustering", "There is no target column, so supervised learning cannot be confirmed.", warnings

    if target_type == "numeric_label":
        warnings.append("Numeric labels are categories, not quantities. Treating this as classification.")
        return "classification", "The target numbers behave like category labels.", warnings

    if target_type == "categorical":
        return "classification", "The target output is a class or category.", warnings

    if target_type == "continuous_numeric":
        if goal == "forecasting" or data_type == "time_series":
            return "time_series_forecasting", "The numeric target is connected to chronological forecasting.", warnings
        return "regression", "The target output is a continuous numeric value.", warnings

    if goal in {"classification", "regression", "grouping", "forecasting"}:
        return {
            "classification": "classification",
            "regression": "regression",
            "grouping": "clustering",
            "forecasting": "time_series_forecasting",
        }[goal], "The task was inferred from the selected project goal.", warnings

    warnings.append("The task type is uncertain because the target and goal information are incomplete.")
    return "classification", "Using classification as a temporary beginner baseline.", warnings


def _infer_classification_type(facts, problem_type):
    if problem_type != "classification":
        return None
    if facts["can_have_multiple_labels"] is True:
        return "multi_label_classification"
    unique_classes = facts["target_unique_classes"]
    if unique_classes == UNKNOWN:
        return None
    if unique_classes == 2:
        return "binary_classification"
    if unique_classes > 2:
        return "multi_class_classification"
    return None


def _select_models(facts, knowledge_base, problem_type):
    catalog = knowledge_base.get("model_catalog", {}).get(problem_type, {})
    baseline = list(catalog.get("baseline", []))
    advanced = list(catalog.get("advanced", []))
    warnings = []

    if not baseline:
        baseline = ["Decision Tree"]

    if problem_type == "nlp" and facts["text_complexity"] == "complex":
        advanced = _dedupe(advanced + ["LSTM", "BERT", "Transformer"])

    if facts["data_type"] == "image":
        baseline = ["CNN"]
        advanced = ["ResNet", "Vision Transformer"]

    if facts["priority"] == "transparency" or facts["domain"] in SENSITIVE_DOMAINS:
        baseline = _prefer_interpretable(baseline, problem_type)
        warnings.append("Explainability is important, so interpretable baseline models are prioritized.")

    dataset_rows = facts["dataset_rows"]
    if dataset_rows == UNKNOWN:
        warnings.append("Dataset size is unknown, so advanced recommendations are tentative.")
        advanced = []
    elif dataset_rows < knowledge_base["thresholds"]["small_dataset_rows_lt"]:
        advanced = [model for model in advanced if model not in DEEP_LEARNING_MODELS]
        warnings.append("Small datasets can overfit complex models. Start with a simple baseline first.")
    elif dataset_rows > knowledge_base["thresholds"]["large_dataset_rows_gt"] and facts["priority"] == "performance":
        deep = catalog.get("deep_learning", [])
        if problem_type in {"nlp", "time_series_forecasting"}:
            advanced = _dedupe(advanced + deep)

    constrained = (
        facts["timeline"] == "short"
        or facts["programming_skill"] == "beginner"
        or facts["has_gpu"] is False
    )
    if constrained:
        advanced = [model for model in advanced if model not in DEEP_LEARNING_MODELS]
        warnings.append("Timeline, skill, or hardware constraints make lightweight models safer first choices.")

    if facts["can_have_multiple_labels"] is True and problem_type == "classification":
        baseline = ["One-vs-Rest Logistic Regression"]
        advanced = _dedupe(["Random Forest", "XGBoost"] + advanced)

    return _dedupe(baseline), _dedupe(advanced), warnings


def _prefer_interpretable(models, problem_type):
    preferred = {
        "classification": ["Logistic Regression", "Decision Tree"],
        "regression": ["Linear Regression", "Decision Tree Regressor"],
        "clustering": ["K-Means", "Hierarchical Clustering"],
        "anomaly_detection": ["Z-Score", "IQR Method"],
        "time_series_forecasting": ["Moving Average", "ARIMA", "Prophet"],
        "nlp": ["TF-IDF + Logistic Regression", "TF-IDF + Naive Bayes"],
    }.get(problem_type, [])
    return _dedupe([model for model in preferred if model in models] + models)


def _select_metrics(facts, knowledge_base, problem_type):
    metrics = knowledge_base.get("metric_catalog", {}).get(problem_type, [])

    if problem_type == "classification":
        metric_groups = knowledge_base["metric_catalog"]["classification"]
        if facts["class_balance"] == "imbalanced" or _number_gt(
            facts["majority_class_percentage"],
            knowledge_base["thresholds"]["imbalanced_majority_class_percentage_gt"],
        ):
            return metric_groups["imbalanced"]
        if facts["class_balance"] == "balanced":
            return metric_groups["balanced"] + ["F1-Score"]
        return ["Accuracy", "F1-Score"]

    if problem_type == "regression":
        metric_groups = knowledge_base["metric_catalog"]["regression"]
        if facts["has_outliers"] is True:
            return metric_groups["outliers"] + metric_groups["explain_variance"]
        return metric_groups["normal_errors"] + metric_groups["explain_variance"]

    return list(metrics)


def _select_preprocessing(facts, models):
    steps = ["Split the dataset into training and test sets."]
    data_type = facts["data_type"]

    if data_type == "text":
        steps.append("Convert text to numerical features with TF-IDF or embeddings.")
    elif data_type == "image":
        steps.extend(["Resize images to a consistent shape.", "Normalize pixel values."])
    elif data_type == "time_series":
        steps.extend(["Sort records by date or time.", "Use a time-based train/test split."])
    elif data_type in {"tabular", "multimodal", UNKNOWN}:
        steps.append("Encode categorical input features.")

    missing_percentage = facts["missing_value_percentage"]
    if missing_percentage != UNKNOWN:
        if missing_percentage < 5:
            steps.append("Remove rows with missing values if the data loss is acceptable.")
        elif missing_percentage > 40:
            steps.append("Drop features with too many missing values or confirm that they are important.")

    if facts["missing_value_type"] in {"numerical", "mixed"}:
        steps.append("Use median imputation for missing numerical values.")
    if facts["missing_value_type"] in {"categorical", "mixed"}:
        steps.append("Use mode imputation or an Unknown category for missing categorical values.")

    if any(_contains_marker(model, SCALE_REQUIRED_MARKERS) for model in models):
        steps.append("Scale numeric features for distance or gradient-based models.")
    if any(_contains_marker(model, TREE_MODEL_MARKERS) for model in models):
        steps.append("Feature scaling is optional for tree-based models.")

    if facts["class_balance"] == "imbalanced":
        steps.append("Use stratified splitting or class weights for imbalanced classes.")
    if facts["has_outliers"] is True:
        steps.append("Inspect outliers before choosing regression metrics or transformations.")

    return steps


def _select_confidence(facts, problem_type, warnings):
    if problem_type is None or facts["data_type"] == UNKNOWN:
        return "low"

    critical_unknowns = [facts["dataset_rows"], facts["class_balance"], facts["missing_value_percentage"]]
    if any(value == UNKNOWN for value in critical_unknowns):
        return "medium"
    if any("uncertain" in warning.lower() for warning in warnings):
        return "low"
    return "high"


def _select_status(facts, confidence):
    if confidence == "high":
        return "approved"
    if UNKNOWN in {facts["dataset_rows"], facts["data_type"], facts["has_target_column"]}:
        return "temporary_recommendation"
    return "approved_with_warnings"


def _build_explanation(facts, problem_type, task_reason, confidence):
    priority_text = {
        "transparency": "The ranking favors models that are easier to explain.",
        "performance": "The ranking includes stronger models when the dataset and constraints allow them.",
        "balanced": "The ranking balances accuracy, explainability, and feasibility.",
        UNKNOWN: "The ranking uses safe beginner-friendly defaults.",
    }.get(facts["priority"], "The ranking uses safe beginner-friendly defaults.")
    return f"{task_reason} Confidence is {confidence}. {priority_text}"


def _coerce_unknown_bool(value):
    if value in {True, False, UNKNOWN}:
        return value
    if value is None:
        return UNKNOWN
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "y"}:
        return True
    if normalized in {"false", "no", "n"}:
        return False
    return UNKNOWN


def _coerce_optional_number(value):
    if value in {None, "", UNKNOWN}:
        return UNKNOWN
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return UNKNOWN
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _number_gt(value, threshold):
    return value != UNKNOWN and value > threshold


def _contains_marker(value, markers):
    return any(marker in value for marker in markers)


def _dedupe(items):
    deduped = []
    for item in items:
        if item and item not in deduped:
            deduped.append(item)
    return deduped
