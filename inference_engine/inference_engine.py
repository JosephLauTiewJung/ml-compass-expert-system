"""Rule-based inference engine for ML Compass recommendations.

The engine keeps each decision traceable by pairing every recommendation update
with a rule ID, evidence, and user-facing impact statement.
"""

from copy import deepcopy
from data.starter_code_catalog import STARTER_CODE_CATALOG

UNKNOWN = "unknown"
SENSITIVE_DOMAINS = {"medical", "legal", "financial", "educational"}
DEEP_LEARNING_MODELS = {"CNN", "ResNet", "Vision Transformer", "LSTM", "GRU", "BERT", "Transformer", "Autoencoder"}
SCALE_REQUIRED_MARKERS = ("Logistic Regression", "Linear Regression", "SVM", "KNN", "Neural", "LSTM", "GRU")
TREE_MODEL_MARKERS = ("Decision Tree", "Random Forest", "XGBoost", "LightGBM")

def normalize_input(raw_form_data):
    """Convert UI answers into the knowledge-base input schema."""
    data = dict(raw_form_data)
    # Coerce free-form UI values once so the rule engine can compare a small,
    # predictable set of booleans, numbers, and UNKNOWN markers.
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
    # Copy the template so recommendations can be built without mutating the
    # shared knowledge-base object loaded by the app or tests.
    result = deepcopy(knowledge_base.get("default_output_template", {}))
    result.setdefault("recommended_baseline_model", [])
    result.setdefault("recommended_advanced_model", [])
    result.setdefault("preprocessing_steps", [])
    result.setdefault("evaluation_metrics", [])
    result.setdefault("warnings", [])
    result.setdefault("explanation_trace", [])
    trace = result["explanation_trace"]

    # Rule-based outputs do not need model selection. Exit early so downstream
    # ML-specific rules do not add contradictory recommendations.
    if facts["output_can_be_calculated_by_formula"] is True:
        _add_trace(
            trace,
            "R001",
            "Machine Learning Necessity",
            "ml_necessity",
            "use_machine_learning = false",
            "The output can be calculated directly with a formula or fixed rule.",
            {"output_can_be_calculated_by_formula": facts["output_can_be_calculated_by_formula"]},
            "Stops model selection and recommends rule-based logic instead of machine learning.",
        )
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

    # The pipeline is intentionally staged: infer task type first, then use that
    # task to choose models, metrics, preprocessing, confidence, and final status.
    problem_type, task_reason, warnings, task_trace = _infer_problem_type(facts)
    result["problem_type"] = problem_type
    result["warnings"].extend(warnings)
    trace.extend(task_trace)

    classification_type, classification_trace = _infer_classification_type(facts, problem_type)
    if classification_type:
        result["classification_type"] = classification_type
    trace.extend(classification_trace)

    baseline, advanced, model_warnings, model_trace = _select_models(facts, knowledge_base, problem_type)
    result["recommended_baseline_model"] = baseline
    result["recommended_advanced_model"] = advanced
    result["warnings"].extend(model_warnings)
    trace.extend(model_trace)

    metrics, metric_trace = _select_metrics(facts, knowledge_base, problem_type)
    result["evaluation_metrics"] = metrics
    trace.extend(metric_trace)

    preprocessing, preprocessing_trace = _select_preprocessing(facts, baseline + advanced)
    result["preprocessing_steps"] = preprocessing
    trace.extend(preprocessing_trace)

    confidence, confidence_trace = _select_confidence(facts, problem_type, result["warnings"])
    result["confidence_level"] = confidence
    trace.extend(confidence_trace)

    status, status_trace = _select_status(facts, result["confidence_level"])
    result["status"] = status
    trace.extend(status_trace)

    result["explanation"] = _build_explanation(facts, problem_type, task_reason, result["confidence_level"])
    result["starter_code_suggestion"] = STARTER_CODE_CATALOG.get(problem_type, "# Add a starter template for this task.")

    # Multiple rules can reach the same advice from different evidence. Dedupe
    # at the boundary while leaving the explanation trace complete.
    result["warnings"] = _dedupe(result["warnings"])
    result["evaluation_metrics"] = _dedupe(result["evaluation_metrics"])
    result["preprocessing_steps"] = _dedupe(result["preprocessing_steps"])
    return result


def _infer_problem_type(facts):
    warnings = []
    trace = []
    goal = facts["goal"]
    data_type = facts["data_type"]
    has_target = facts["has_target_column"]
    target_type = facts["target_type"]

    # Explicit task goals and data types take precedence over target-column
    # heuristics because they encode the user's intended workflow.
    if goal == "anomaly_detection":
        _add_trace(
            trace,
            "R008",
            "Anomaly Detection Task",
            "task_type",
            "problem_type = anomaly_detection",
            "The goal is to find rare or abnormal cases.",
            {"goal": goal},
            "Selects anomaly detection models and metrics.",
        )
        return "anomaly_detection", "The goal is to identify rare or abnormal cases.", warnings, trace

    if data_type == "time_series" and goal in {"forecasting", UNKNOWN}:
        _add_trace(
            trace,
            "R009",
            "Time Series Forecasting Task",
            "task_type",
            "problem_type = time_series_forecasting",
            "The data is ordered by time and the goal is future prediction.",
            {"data_type": data_type, "goal": goal},
            "Selects forecasting models, time-based splitting, and forecasting metrics.",
        )
        return "time_series_forecasting", "The data is ordered by time and the goal is future prediction.", warnings, trace

    if data_type == "text" or goal == "nlp":
        _add_trace(
            trace,
            "R010",
            "NLP / LLM Task",
            "task_type",
            "problem_type = nlp",
            "The task depends on understanding or processing human text.",
            {"data_type": data_type, "goal": goal},
            "Selects text preprocessing, NLP models, and text classification metrics.",
        )
        return "nlp", "The project depends on understanding or processing human text.", warnings, trace

    if has_target is False:
        if goal == "grouping":
            _add_trace(
                trace,
                "R007",
                "Clustering Task",
                "task_type",
                "problem_type = clustering",
                "There is no target column and the goal is to group similar data.",
                {"has_target_column": has_target, "goal": goal},
                "Selects unsupervised clustering models and clustering metrics.",
            )
            return "clustering", "There is no target column and the goal is to group similar data.", warnings, trace
        warnings.append("No target column was provided, so the safest unsupervised starting point is clustering.")
        _add_trace(
            trace,
            "R020",
            "Unlabeled Data Clustering",
            "learning_type",
            "problem_type = clustering",
            "No target column was provided, so supervised learning cannot be confirmed.",
            {"has_target_column": has_target, "goal": goal},
            "Uses clustering as the safest unsupervised starting point.",
        )
        return "clustering", "There is no target column, so supervised learning cannot be confirmed.", warnings, trace

    if target_type == "numeric_label":
        warnings.append("Numeric labels are categories, not quantities. Treating this as classification.")
        _add_trace(
            trace,
            "R011",
            "Numeric Category Rule",
            "numeric_trap",
            "problem_type = classification",
            "The target numbers behave like category labels, not measurable quantities.",
            {"target_type": target_type},
            "Treats numeric labels such as 0/1 or 1/2/3 as classes.",
        )
        return "classification", "The target numbers behave like category labels.", warnings, trace

    if target_type == "categorical":
        _add_trace(
            trace,
            "R005",
            "Classification Task",
            "task_type",
            "problem_type = classification",
            "The target output is a class or category.",
            {"has_target_column": has_target, "target_type": target_type},
            "Selects classification models and classification metrics.",
        )
        return "classification", "The target output is a class or category.", warnings, trace

    if target_type == "continuous_numeric":
        if goal == "forecasting" or data_type == "time_series":
            _add_trace(
                trace,
                "R009",
                "Time Series Forecasting Task",
                "task_type",
                "problem_type = time_series_forecasting",
                "The numeric target is connected to chronological forecasting.",
                {"target_type": target_type, "data_type": data_type, "goal": goal},
                "Forecasting overrides ordinary regression because order over time matters.",
            )
            return "time_series_forecasting", "The numeric target is connected to chronological forecasting.", warnings, trace
        # Continuous numeric targets default to regression only after the
        # forecasting/time-series cases have been ruled out.
        _add_trace(
            trace,
            "R006",
            "Regression Task",
            "task_type",
            "problem_type = regression",
            "The target output is a continuous numeric value.",
            {"has_target_column": has_target, "target_type": target_type},
            "Selects regression models and regression metrics.",
        )
        return "regression", "The target output is a continuous numeric value.", warnings, trace

    if goal in {"classification", "regression", "grouping", "forecasting"}:
        # When target metadata is incomplete, preserve momentum by falling back
        # to the selected goal and recording uncertainty in the trace.
        problem_type = {
            "classification": "classification",
            "regression": "regression",
            "grouping": "clustering",
            "forecasting": "time_series_forecasting",
        }[goal]
        _add_trace(
            trace,
            "R050",
            "Missing Target Column",
            "missing_input",
            f"problem_type = {problem_type}",
            "The target details are incomplete, so the task is inferred from the selected project goal.",
            {"goal": goal, "target_type": target_type, "has_target_column": has_target},
            "Allows a temporary recommendation while preserving uncertainty.",
        )
        return problem_type, "The task was inferred from the selected project goal.", warnings, trace

    warnings.append("The task type is uncertain because the target and goal information are incomplete.")
    _add_trace(
        trace,
        "R049",
        "Low Confidence Recommendation",
        "confidence",
        "problem_type = classification",
        "The task type is uncertain because the target and goal information are incomplete.",
        {"goal": goal, "target_type": target_type, "has_target_column": has_target},
        "Uses classification as a temporary beginner baseline.",
    )
    return "classification", "Using classification as a temporary beginner baseline.", warnings, trace


def _infer_classification_type(facts, problem_type):
    if problem_type != "classification":
        return None, []
    if facts["can_have_multiple_labels"] is True:
        # Multi-label classification is checked before class count because a
        # record can have many labels even when each label is binary.
        return "multi_label_classification", [
            _trace_item(
                "R058",
                "Multi-Label Classification",
                "task_type",
                "classification_type = multi_label_classification",
                "One record can belong to multiple labels at the same time.",
                {"can_have_multiple_labels": facts["can_have_multiple_labels"]},
                "Uses multi-label-compatible model recommendations.",
            )
        ]
    unique_classes = facts["target_unique_classes"]
    if unique_classes == UNKNOWN:
        return None, []
    if unique_classes == 2:
        return "binary_classification", [
            _trace_item(
                "R056",
                "Binary Classification",
                "task_type",
                "classification_type = binary_classification",
                "The classification target has exactly two classes.",
                {"target_unique_classes": unique_classes},
                "Labels the task as binary classification.",
            )
        ]
    if unique_classes > 2:
        return "multi_class_classification", [
            _trace_item(
                "R057",
                "Multi-Class Classification",
                "task_type",
                "classification_type = multi_class_classification",
                "The classification target has more than two classes.",
                {"target_unique_classes": unique_classes},
                "Labels the task as multi-class classification.",
            )
        ]
    return None, []


def _select_models(facts, knowledge_base, problem_type):
    catalog = knowledge_base.get("model_catalog", {}).get(problem_type, {})
    baseline = list(catalog.get("baseline", []))
    advanced = list(catalog.get("advanced", []))
    warnings = []
    trace = []

    if not baseline:
        baseline = ["Decision Tree"]

    # Start from the catalog, then layer domain, data size, and practical
    # constraints on top. This keeps model selection explainable and testable.
    _add_trace(
        trace,
        "R029",
        "Baseline Model Rule",
        "dataset_size",
        f"baseline_models = {baseline}",
        "A beginner project should start with a simple baseline before advanced models.",
        {"problem_type": problem_type},
        "Keeps baseline models visible even when advanced options are available.",
    )

    _add_data_type_model_trace(trace, facts, problem_type, baseline, advanced)

    if problem_type == "nlp" and facts["text_complexity"] == "complex":
        advanced = _dedupe(advanced + ["LSTM", "BERT", "Transformer"])
        _add_trace(
            trace,
            "R015",
            "Complex Text Data",
            "data_type",
            "advanced_models include LSTM, BERT, Transformer",
            "Complex text can require context, word order, or sentence meaning.",
            {"data_type": facts["data_type"], "text_complexity": facts["text_complexity"]},
            "Adds deep-learning NLP candidates before practical constraints are applied.",
        )

    if facts["data_type"] == "image":
        baseline = ["CNN"]
        advanced = ["ResNet", "Vision Transformer"]
        _add_trace(
            trace,
            "R016",
            "Image Data",
            "data_type",
            "models = CNN, ResNet, Vision Transformer",
            "Image data contains spatial pixel patterns.",
            {"data_type": facts["data_type"]},
            "Overrides generic model catalogs with image-specific model families.",
        )

    if facts["priority"] == "transparency" or facts["domain"] in SENSITIVE_DOMAINS:
        # Sensitive domains favor interpretable first recommendations even when
        # higher-capacity models remain available as advanced options.
        baseline = _prefer_interpretable(baseline, problem_type)
        warnings.append("Explainability is important, so interpretable baseline models are prioritized.")
        _add_trace(
            trace,
            "R003",
            "Transparency Priority",
            "priority_filter",
            f"baseline_models = {baseline}",
            "Sensitive or explainable projects need models that are easier to justify.",
            {"priority": facts["priority"], "domain": facts["domain"]},
            "Reorders baseline models so interpretable choices appear first.",
        )

    dataset_rows = facts["dataset_rows"]
    if dataset_rows == UNKNOWN:
        warnings.append("Dataset size is unknown, so advanced recommendations are tentative.")
        advanced = []
        _add_trace(
            trace,
            "R051",
            "Missing Dataset Size",
            "missing_input",
            "advanced_models = []",
            "Dataset size is unknown, so complex model recommendations are uncertain.",
            {"dataset_rows": dataset_rows},
            "Removes advanced recommendations until dataset size is known.",
        )
    elif dataset_rows < knowledge_base["thresholds"]["small_dataset_rows_lt"]:
        advanced = [model for model in advanced if model not in DEEP_LEARNING_MODELS]
        warnings.append("Small datasets can overfit complex models. Start with a simple baseline first.")
        _add_trace(
            trace,
            "R025",
            "Small Dataset",
            "dataset_size",
            f"advanced_models = {advanced}",
            "Small datasets have higher overfitting risk.",
            {"dataset_rows": dataset_rows, "threshold": knowledge_base["thresholds"]["small_dataset_rows_lt"]},
            "Removes deep learning as a first-choice recommendation.",
        )
    elif dataset_rows > knowledge_base["thresholds"]["large_dataset_rows_gt"] and facts["priority"] == "performance":
        deep = catalog.get("deep_learning", [])
        if problem_type in {"nlp", "time_series_forecasting"}:
            advanced = _dedupe(advanced + deep)
            _add_trace(
                trace,
                "R027",
                "Large Dataset",
                "dataset_size",
                f"advanced_models = {advanced}",
                "A large dataset with a performance priority can support stronger models for complex tasks.",
                {"dataset_rows": dataset_rows, "priority": facts["priority"], "problem_type": problem_type},
                "Adds deep learning options for suitable NLP or forecasting tasks.",
            )
    else:
        # Medium-sized datasets keep advanced options visible without promoting
        # heavyweight deep-learning models by default.
        _add_trace(
            trace,
            "R026",
            "Medium Dataset",
            "dataset_size",
            f"advanced_models = {advanced}",
            "The dataset size is large enough to consider ensemble models but does not force deep learning.",
            {"dataset_rows": dataset_rows},
            "Keeps standard advanced models as upgrade options.",
        )

    constrained = (
        facts["timeline"] == "short"
        or facts["programming_skill"] == "beginner"
        or facts["has_gpu"] is False
    )
    if constrained:
        # Practical constraints are applied after size rules so they can veto
        # deep-learning candidates added by complex text or large datasets.
        advanced = [model for model in advanced if model not in DEEP_LEARNING_MODELS]
        warnings.append("Timeline, skill, or hardware constraints make lightweight models safer first choices.")
        _add_trace(
            trace,
            "R045",
            "Resource Limitation",
            "practical_constraints",
            f"advanced_models = {advanced}",
            "Timeline, programming skill, or hardware constraints make lightweight models safer first choices.",
            {
                "timeline": facts["timeline"],
                "programming_skill": facts["programming_skill"],
                "has_gpu": facts["has_gpu"],
            },
            "Removes heavyweight deep-learning models from first recommendations.",
        )

    if facts["can_have_multiple_labels"] is True and problem_type == "classification":
        baseline = ["One-vs-Rest Logistic Regression"]
        advanced = _dedupe(["Random Forest", "XGBoost"] + advanced)
        _add_trace(
            trace,
            "R058",
            "Multi-Label Classification",
            "task_type",
            "baseline_models = ['One-vs-Rest Logistic Regression']",
            "One record can belong to multiple labels at the same time.",
            {"can_have_multiple_labels": facts["can_have_multiple_labels"]},
            "Uses a baseline model that supports independent label decisions.",
        )

    return _dedupe(baseline), _dedupe(advanced), warnings, trace


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


def _add_data_type_model_trace(trace, facts, problem_type, baseline, advanced):
    data_type = facts["data_type"]
    if data_type == "tabular":
        _add_trace(
            trace,
            "R013",
            "Tabular Data Model Family",
            "data_type",
            f"models = {baseline + advanced}",
            "Tabular datasets are usually a good fit for traditional machine learning models.",
            {"data_type": data_type, "problem_type": problem_type},
            "Uses the problem-type model catalog for structured data.",
        )
    elif data_type == "text" and facts["text_complexity"] != "complex":
        _add_trace(
            trace,
            "R014",
            "Simple Text Data",
            "data_type",
            f"models = {baseline + advanced}",
            "Simple text tasks can often be solved using TF-IDF with linear models or Naive Bayes.",
            {"data_type": data_type, "text_complexity": facts["text_complexity"]},
            "Uses beginner-friendly text models before considering deep learning.",
        )
    elif data_type == "time_series":
        _add_trace(
            trace,
            "R017",
            "Time Series Data",
            "data_type",
            f"models = {baseline + advanced}",
            "Chronological order matters for time-series tasks.",
            {"data_type": data_type, "goal": facts["goal"]},
            "Uses forecasting models from the time-series catalog.",
        )
    elif data_type == "multimodal":
        _add_trace(
            trace,
            "R018",
            "Multi-Modal Data",
            "data_type",
            f"models = {baseline + advanced}",
            "Multiple data types require identifying the dominant data type or combining extracted features.",
            {"data_type": data_type},
            "Starts with the inferred task catalog and warns the user to simplify the first version.",
        )


def _select_metrics(facts, knowledge_base, problem_type):
    metrics = knowledge_base.get("metric_catalog", {}).get(problem_type, [])
    trace = []

    if problem_type == "classification":
        metric_groups = knowledge_base["metric_catalog"]["classification"]
        # Class imbalance changes what "good" means, so metric selection checks
        # both the explicit balance answer and the numeric majority share.
        if facts["class_balance"] == "imbalanced" or _number_gt(
            facts["majority_class_percentage"],
            knowledge_base["thresholds"]["imbalanced_majority_class_percentage_gt"],
        ):
            selected = metric_groups["imbalanced"]
            _add_trace(
                trace,
                "R039",
                "Imbalanced Classification Metric",
                "evaluation_metrics",
                f"metrics = {selected}",
                "Accuracy alone may be misleading for imbalanced data.",
                {
                    "class_balance": facts["class_balance"],
                    "majority_class_percentage": facts["majority_class_percentage"],
                },
                "Selects F1, precision, recall, and PR-AUC instead of relying only on accuracy.",
            )
            return selected, trace
        if facts["class_balance"] == "balanced":
            selected = metric_groups["balanced"] + ["F1-Score"]
            _add_trace(
                trace,
                "R038",
                "Balanced Classification Metric",
                "evaluation_metrics",
                f"metrics = {selected}",
                "Balanced classes make accuracy usable, with F1-score as an additional check.",
                {"class_balance": facts["class_balance"]},
                "Selects beginner-friendly classification metrics.",
            )
            return selected, trace
        selected = ["Accuracy", "F1-Score"]
        _add_trace(
            trace,
            "R052",
            "Missing Class Balance",
            "missing_input",
            f"metrics = {selected}",
            "Class balance is unknown.",
            {"class_balance": facts["class_balance"]},
            "Uses both accuracy and F1-score while preserving uncertainty.",
        )
        return selected, trace

    if problem_type == "regression":
        metric_groups = knowledge_base["metric_catalog"]["regression"]
        if facts["has_outliers"] is True:
            selected = metric_groups["outliers"] + metric_groups["explain_variance"]
            _add_trace(
                trace,
                "R043",
                "Regression MAE",
                "evaluation_metrics",
                f"metrics = {selected}",
                "MAE is safer when outliers are present.",
                {"has_outliers": facts["has_outliers"]},
                "Selects a regression metric that is less dominated by large errors.",
            )
            return selected, trace
        selected = metric_groups["normal_errors"] + metric_groups["explain_variance"]
        _add_trace(
            trace,
            "R042",
            "Regression RMSE",
            "evaluation_metrics",
            f"metrics = {selected}",
            "RMSE is a standard regression error metric when outliers are not the main concern.",
            {"has_outliers": facts["has_outliers"]},
            "Selects RMSE plus R2 for error size and explained variance.",
        )
        return selected, trace

    selected = list(metrics)
    if selected:
        _add_trace(
            trace,
            "R054",
            "Metric Clarity Check",
            "evaluation_metrics",
            f"metrics = {selected}",
            "The selected metrics match the inferred task type.",
            {"problem_type": problem_type},
            "Uses the task-specific metric catalog.",
        )
    return selected, trace


def _select_preprocessing(facts, models):
    steps = ["Split the dataset into training and test sets."]
    trace = [
        _trace_item(
            "R059",
            "Safe Final Recommendation",
            "final_validation",
            "preprocessing includes train/test split",
            "A model recommendation needs a separate training and test set for validation.",
            {"models": models},
            "Adds a standard validation step before model training.",
        )
    ]
    data_type = facts["data_type"]

    # Data-type-specific preprocessing is chosen before generic data-quality
    # handling so the final checklist reads from broad setup to cleanup details.
    if data_type == "text":
        steps.append("Convert text to numerical features with TF-IDF or embeddings.")
        _add_trace(
            trace,
            "R014",
            "Simple Text Data",
            "data_type",
            "preprocessing includes TF-IDF or embeddings",
            "Text must be converted to numerical features before machine learning models can use it.",
            {"data_type": data_type},
            "Adds text vectorization guidance.",
        )
    elif data_type == "image":
        steps.extend(["Resize images to a consistent shape.", "Normalize pixel values."])
        _add_trace(
            trace,
            "R016",
            "Image Data",
            "data_type",
            "preprocessing includes resizing and pixel normalization",
            "Image models need consistent input dimensions and normalized pixel values.",
            {"data_type": data_type},
            "Adds image preprocessing guidance.",
        )
    elif data_type == "time_series":
        steps.extend(["Sort records by date or time.", "Use a time-based train/test split."])
        _add_trace(
            trace,
            "R017",
            "Time Series Data",
            "data_type",
            "preprocessing includes chronological sorting and time-based split",
            "Chronological order matters for forecasting tasks.",
            {"data_type": data_type},
            "Avoids random splitting that can leak future information into training.",
        )
    elif data_type in {"tabular", "multimodal", UNKNOWN}:
        steps.append("Encode categorical input features.")
        _add_trace(
            trace,
            "R033",
            "Nominal Encoding",
            "preprocessing",
            "preprocessing includes categorical feature encoding",
            "Categorical input features must be represented numerically for most ML models.",
            {"data_type": data_type},
            "Adds basic categorical encoding guidance.",
        )

    missing_percentage = facts["missing_value_percentage"]
    if missing_percentage != UNKNOWN:
        if missing_percentage < 5:
            steps.append("Remove rows with missing values if the data loss is acceptable.")
            _add_trace(
                trace,
                "R035",
                "Low Missing Values",
                "preprocessing",
                "preprocessing includes optional row removal",
                "Missing values are low enough that row removal may be acceptable.",
                {"missing_value_percentage": missing_percentage},
                "Adds a simple missing-value option for small missing percentages.",
            )
        elif missing_percentage > 40:
            steps.append("Drop features with too many missing values or confirm that they are important.")
            _add_trace(
                trace,
                "R055",
                "Too Many Missing Values in Feature",
                "data_quality",
                "preprocessing includes dropping high-missingness features",
                "Very high missingness can make a feature unreliable.",
                {"missing_value_percentage": missing_percentage},
                "Warns the user to drop or justify heavily missing features.",
            )

    if facts["missing_value_type"] in {"numerical", "mixed"}:
        steps.append("Use median imputation for missing numerical values.")
        _add_trace(
            trace,
            "R036",
            "Missing Numerical Values",
            "preprocessing",
            "preprocessing includes median imputation",
            "Median imputation is a robust beginner-friendly choice for missing numerical values.",
            {"missing_value_type": facts["missing_value_type"]},
            "Adds numerical missing-value handling.",
        )
    if facts["missing_value_type"] in {"categorical", "mixed"}:
        steps.append("Use mode imputation or an Unknown category for missing categorical values.")
        _add_trace(
            trace,
            "R037",
            "Missing Categorical Values",
            "preprocessing",
            "preprocessing includes mode imputation or Unknown category",
            "Missing categorical values need a valid category before model training.",
            {"missing_value_type": facts["missing_value_type"]},
            "Adds categorical missing-value handling.",
        )

    if any(_contains_marker(model, SCALE_REQUIRED_MARKERS) for model in models):
        # Scaling depends on the selected models, not just the data type.
        steps.append("Scale numeric features for distance or gradient-based models.")
        _add_trace(
            trace,
            "R030",
            "Scaling Required",
            "preprocessing",
            "preprocessing includes numeric feature scaling",
            "Distance and gradient-based models are affected by feature scale.",
            {"models": models},
            "Adds scaling guidance for models such as Logistic Regression, SVM, KNN, or neural models.",
        )
    if any(_contains_marker(model, TREE_MODEL_MARKERS) for model in models):
        steps.append("Feature scaling is optional for tree-based models.")
        _add_trace(
            trace,
            "R031",
            "Scaling Optional for Tree Models",
            "preprocessing",
            "preprocessing notes scaling is optional",
            "Tree models split using thresholds and are less affected by scale.",
            {"models": models},
            "Prevents unnecessary scaling work for tree-based models.",
        )

    if facts["class_balance"] == "imbalanced":
        steps.append("Use stratified splitting or class weights for imbalanced classes.")
        _add_trace(
            trace,
            "R039",
            "Imbalanced Classification Metric",
            "evaluation_metrics",
            "preprocessing includes stratified splitting or class weights",
            "Imbalanced classes need train/test splits and training settings that preserve minority classes.",
            {"class_balance": facts["class_balance"]},
            "Adds imbalance handling guidance.",
        )
    if facts["has_outliers"] is True:
        steps.append("Inspect outliers before choosing regression metrics or transformations.")
        _add_trace(
            trace,
            "R043",
            "Regression MAE",
            "evaluation_metrics",
            "preprocessing includes outlier inspection",
            "Outliers can change model choice, metric choice, and transformations.",
            {"has_outliers": facts["has_outliers"]},
            "Adds outlier inspection before modeling.",
        )

    return steps, trace


def _select_confidence(facts, problem_type, warnings):
    # Confidence reflects input completeness and rule conflicts; it does not
    # score model accuracy.
    if problem_type is None or facts["data_type"] == UNKNOWN:
        return "low", [
            _trace_item(
                "R049",
                "Low Confidence Recommendation",
                "confidence",
                "confidence_level = low",
                "Important task or data type information is missing.",
                {"problem_type": problem_type, "data_type": facts["data_type"]},
                "Marks the recommendation as uncertain.",
            )
        ]

    critical_unknowns = [facts["dataset_rows"], facts["class_balance"], facts["missing_value_percentage"]]
    if any(value == UNKNOWN for value in critical_unknowns):
        return "medium", [
            _trace_item(
                "R048",
                "Medium Confidence Recommendation",
                "confidence",
                "confidence_level = medium",
                "The task and data type are clear, but dataset size, class balance, or missing values are unknown.",
                {
                    "dataset_rows": facts["dataset_rows"],
                    "class_balance": facts["class_balance"],
                    "missing_value_percentage": facts["missing_value_percentage"],
                },
                "Allows a temporary recommendation while flagging missing inputs.",
            )
        ]
    if any("uncertain" in warning.lower() for warning in warnings):
        return "low", [
            _trace_item(
                "R049",
                "Low Confidence Recommendation",
                "confidence",
                "confidence_level = low",
                "Warnings indicate uncertainty or rule conflict.",
                {"warnings": warnings},
                "Marks the recommendation as uncertain.",
            )
        ]
    return "high", [
        _trace_item(
            "R047",
            "High Confidence Recommendation",
            "confidence",
            "confidence_level = high",
            "Task type, data type, dataset size, class balance, and missing-value information are available.",
            {
                "problem_type": problem_type,
                "data_type": facts["data_type"],
                "dataset_rows": facts["dataset_rows"],
                "class_balance": facts["class_balance"],
                "missing_value_percentage": facts["missing_value_percentage"],
            },
            "Approves the recommendation without uncertainty flags.",
        )
    ]


def _select_status(facts, confidence):
    # Status is a user-facing readiness label derived from confidence plus the
    # most important missing-input checks.
    if confidence == "high":
        return "approved", [
            _trace_item(
                "R059",
                "Safe Final Recommendation",
                "final_validation",
                "status = approved",
                "The recommendation has enough information and no major conflict.",
                {"confidence_level": confidence},
                "Presents the recommendation as ready to use.",
            )
        ]
    if UNKNOWN in {facts["dataset_rows"], facts["data_type"], facts["has_target_column"]}:
        return "temporary_recommendation", [
            _trace_item(
                "R060",
                "Temporary Recommendation",
                "final_validation",
                "status = temporary_recommendation",
                "Important information is missing.",
                {
                    "dataset_rows": facts["dataset_rows"],
                    "data_type": facts["data_type"],
                    "has_target_column": facts["has_target_column"],
                },
                "Shows a safe recommendation while making uncertainty explicit.",
            )
        ]
    return "approved_with_warnings", [
        _trace_item(
            "R060",
            "Temporary Recommendation",
            "final_validation",
            "status = approved_with_warnings",
            "The recommendation is usable but includes uncertainty or warnings.",
            {"confidence_level": confidence},
            "Keeps the recommendation visible with warning context.",
        )
    ]


def _build_explanation(facts, problem_type, task_reason, confidence):
    priority_text = {
        "transparency": "The ranking favors models that are easier to explain.",
        "performance": "The ranking includes stronger models when the dataset and constraints allow them.",
        "balanced": "The ranking balances accuracy, explainability, and feasibility.",
        UNKNOWN: "The ranking uses safe beginner-friendly defaults.",
    }.get(facts["priority"], "The ranking uses safe beginner-friendly defaults.")
    return f"{task_reason} Confidence is {confidence}. {priority_text}"


def _add_trace(trace, rule_id, rule_name, category, decision, reason, evidence, impact):
    trace.append(_trace_item(rule_id, rule_name, category, decision, reason, evidence, impact))


def _trace_item(rule_id, rule_name, category, decision, reason, evidence, impact):
    return {
        "rule_id": rule_id,
        "rule_name": rule_name,
        "category": category,
        "decision": decision,
        "reason": reason,
        "evidence": evidence,
        "impact": impact,
    }


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
