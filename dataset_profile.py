"""Dataset profiling helpers for upload-driven form defaults."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


UNKNOWN = "unknown"
IMBALANCED_MAJORITY_PERCENTAGE_GT = 80
LOW_UNIQUE_NUMERIC_TARGET_MAX = 20
LOW_UNIQUE_NUMERIC_TARGET_RATIO_MAX = 0.05


def profile_dataset(df, target_column=None):
    """Return inferred form defaults and the fields that were inferred."""
    profile = {
        "data_type": "tabular",
        "dataset_rows": int(len(df)),
        "missing_value_percentage": _missing_value_percentage(df),
        "missing_value_type": _missing_value_type(df),
        "has_outliers": _has_important_outliers(df, target_column),
    }
    auto_fields = set(profile)

    if target_column and target_column in df.columns:
        target_profile = _profile_target(df[target_column])
        profile.update(target_profile)
        auto_fields.update(target_profile)

    return profile, auto_fields


def _missing_value_percentage(df):
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0
    missing_cells = int(df.isna().sum().sum())
    return round((missing_cells / total_cells) * 100)


def _missing_value_type(df):
    missing_columns = [column for column in df.columns if df[column].isna().any()]
    if not missing_columns:
        return UNKNOWN

    has_numeric_missing = any(is_numeric_dtype(df[column]) for column in missing_columns)
    has_categorical_missing = any(not is_numeric_dtype(df[column]) for column in missing_columns)

    if has_numeric_missing and has_categorical_missing:
        return "mixed"
    if has_numeric_missing:
        return "numerical"
    return "categorical"


def _has_important_outliers(df, target_column=None):
    feature_df = df.drop(columns=[target_column], errors="ignore") if target_column else df
    numeric_df = feature_df.select_dtypes(include="number")
    if numeric_df.empty:
        return UNKNOWN

    for column in numeric_df.columns:
        values = numeric_df[column].dropna()
        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_count = int((values != values.median()).sum())
            important_threshold = max(3, round(len(values) * 0.01))
            if outlier_count >= important_threshold:
                return True
            continue

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outlier_count = int(((values < lower_bound) | (values > upper_bound)).sum())
        important_threshold = max(3, round(len(values) * 0.01))
        if outlier_count >= important_threshold:
            return True

    return False


def _profile_target(target):
    non_null_target = target.dropna()
    if non_null_target.empty:
        return {
            "has_target_column": True,
            "target_type": UNKNOWN,
            "target_unique_classes": UNKNOWN,
            "majority_class_percentage": UNKNOWN,
            "class_balance": UNKNOWN,
        }

    unique_classes = int(non_null_target.nunique())
    majority_class_percentage = round((non_null_target.value_counts().iloc[0] / len(non_null_target)) * 100)
    target_type = _target_type(target, unique_classes, len(non_null_target))

    return {
        "has_target_column": True,
        "target_type": target_type,
        "target_unique_classes": unique_classes if target_type != "continuous_numeric" else UNKNOWN,
        "majority_class_percentage": majority_class_percentage,
        "class_balance": (
            "imbalanced"
            if majority_class_percentage > IMBALANCED_MAJORITY_PERCENTAGE_GT
            else "balanced"
        ),
    }


def _target_type(target, unique_classes, row_count):
    if is_bool_dtype(target) or not is_numeric_dtype(target):
        return "categorical"
    if unique_classes <= LOW_UNIQUE_NUMERIC_TARGET_MAX:
        return "numeric_label"
    if row_count and (unique_classes / row_count) <= LOW_UNIQUE_NUMERIC_TARGET_RATIO_MAX:
        return "numeric_label"
    return "continuous_numeric"
