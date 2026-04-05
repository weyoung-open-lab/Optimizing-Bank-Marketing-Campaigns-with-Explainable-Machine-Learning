import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from models import build_model, needs_scaling
from preprocess import build_preprocessor
from metrics import compute_basic_metrics, precision_lift_at_k


# =========================
# Fit one model and score
# =========================
def fit_and_score_model(model_name, X_train, y_train, X_test, y_test, cfg):
    """
    Train one model under a unified preprocessing pipeline,
    then return:
      - trained pipeline
      - predicted probabilities on test set
      - basic model metrics
      - Top-K policy metrics
    """
    model = build_model(model_name, cfg)

    scale_numeric = needs_scaling(model_name)
    preprocessor, cat_cols, num_cols = build_preprocessor(
        X_train,
        scale_numeric=scale_numeric
    )

    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    # probability prediction
    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    else:
        # fallback: convert decision_function to probability-like scores
        scores = pipe.decision_function(X_test)
        y_score = 1.0 / (1.0 + np.exp(-scores))

    # basic metrics
    basic_metrics = compute_basic_metrics(y_test, y_score)

    # policy metrics
    policy_rows = []
    for kf in cfg.k_fracs:
        row = precision_lift_at_k(y_test, y_score, kf)
        policy_rows.append(row)

    return {
        "model_name": model_name,
        "pipeline": pipe,
        "y_score": y_score,
        "basic_metrics": basic_metrics,
        "policy_metrics": policy_rows,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }


# =========================
# Evaluate one split
# =========================
def evaluate_models_on_split(model_names, X_train, y_train, X_test, y_test, cfg, split_name):
    """
    Evaluate a list of models on a given split.

    Returns:
      - model_metrics_df
      - policy_metrics_df
      - trained_pipelines (dict)
    """
    model_metric_rows = []
    policy_metric_rows = []
    trained_pipelines = {}

    for model_name in model_names:
        if cfg.verbose:
            print(f"[{split_name}] Running model: {model_name}")

        result = fit_and_score_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cfg=cfg,
        )

        trained_pipelines[model_name] = result["pipeline"]

        # basic metrics row
        model_metric_rows.append({
            "split": split_name,
            "model": model_name,
            **result["basic_metrics"],
        })

        # policy rows
        for row in result["policy_metrics"]:
            policy_metric_rows.append({
                "split": split_name,
                "model": model_name,
                **row,
            })

    model_metrics_df = pd.DataFrame(model_metric_rows)
    policy_metrics_df = pd.DataFrame(policy_metric_rows)

    return model_metrics_df, policy_metrics_df, trained_pipelines


# =========================
# Select best model
# =========================
def select_best_model(policy_metrics_df, cfg, split_name="month_aware"):
    """
    Select the best model based on Precision@K at cfg.best_model_selection_k
    under the specified split.
    """
    sub = policy_metrics_df[
        (policy_metrics_df["split"] == split_name) &
        (policy_metrics_df["k_frac"] == cfg.best_model_selection_k)
    ].copy()

    if sub.empty:
        raise ValueError(
            f"No policy results found for split={split_name}, "
            f"k={cfg.best_model_selection_k}"
        )

    sub = sub.sort_values(
        by=["precision_at_k", "lift_at_k"],
        ascending=False
    )

    best_row = sub.iloc[0]
    return {
        "best_model_name": best_row["model"],
        "selection_split": split_name,
        "selection_k": best_row["k_frac"],
        "precision_at_k": best_row["precision_at_k"],
        "lift_at_k": best_row["lift_at_k"],
        "base_rate": best_row["base_rate"],
    }


# =========================
# Optional helper
# =========================
def make_lift_curve_df(y_true, y_score, model_name, split_name,
                       k_min=0.01, k_max=0.30, n_points=30):
    """
    Build a Lift@K curve table for one model on one split.
    """
    rows = []
    for kf in np.linspace(k_min, k_max, n_points):
        kf = float(kf)
        out = precision_lift_at_k(y_true, y_score, kf)
        rows.append({
            "split": split_name,
            "model": model_name,
            "k_frac": out["k_frac"],
            "lift_at_k": out["lift_at_k"],
            "precision_at_k": out["precision_at_k"],
        })
    return pd.DataFrame(rows)