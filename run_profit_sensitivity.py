import os
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from config import Config
from data_utils import load_data, split_month_holdout
from preprocess import build_preprocessor
from models import build_model, needs_scaling


def ensure_dirs(cfg):
    os.makedirs(cfg.tables_dir, exist_ok=True)
    os.makedirs(cfg.figures_dir, exist_ok=True)


def load_best_tuned_model_info(tables_dir):
    """
    Load the final tuned SHAP model name and its tuned hyperparameters.
    """
    best_model_path = f"{tables_dir}/best_shap_model_tuned.csv"
    tuning_summary_path = f"{tables_dir}/tuning_summary_top5.csv"

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Missing file: {best_model_path}")
    if not os.path.exists(tuning_summary_path):
        raise FileNotFoundError(f"Missing file: {tuning_summary_path}")

    best_df = pd.read_csv(best_model_path)
    tuning_df = pd.read_csv(tuning_summary_path)

    model_name = best_df.loc[0, "best_shap_model_tuned"]

    row = tuning_df[tuning_df["model"] == model_name]
    if row.empty:
        raise ValueError(f"Model '{model_name}' not found in tuning_summary_top5.csv")

    best_params_json = row.iloc[0]["best_params_json"]
    best_params = json.loads(best_params_json)

    return model_name, best_params


def strip_model_prefix(best_params):
    return {k.replace("model__", ""): v for k, v in best_params.items()}


def sanitize_name(name: str) -> str:
    return (
        name.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
    )


def build_trained_pipeline(model_name, X_train, y_train, cfg, tuned_params=None):
    model = build_model(model_name, cfg)

    if tuned_params is not None:
        model.set_params(**tuned_params)

    scale_numeric = needs_scaling(model_name)
    preprocessor, _, _ = build_preprocessor(X_train, scale_numeric=scale_numeric)

    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def predict_scores(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]

    scores = pipe.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def evaluate_profit_scenario(y_true, y_score, V, C):
    """
    Evaluate a profit-threshold policy:
        contact if p > C / V

    Returns a dict with threshold, contact rate, conversion among contacted,
    realized profit, and expected profit summary.
    """
    if V <= 0:
        raise ValueError("V must be > 0.")
    if C < 0:
        raise ValueError("C must be >= 0.")

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    threshold = C / V
    contact_mask = y_score > threshold

    n_total = len(y_true)
    n_contacted = int(contact_mask.sum())
    contact_rate = n_contacted / n_total if n_total > 0 else np.nan

    if n_contacted == 0:
        return {
            "V": V,
            "C": C,
            "threshold": threshold,
            "n_total": n_total,
            "n_contacted": 0,
            "contact_rate": 0.0,
            "observed_conversion_rate_contacted": np.nan,
            "expected_profit_total": 0.0,
            "realized_profit_total": 0.0,
            "realized_profit_per_customer": 0.0,
        }

    y_contacted = y_true[contact_mask]
    p_contacted = y_score[contact_mask]

    observed_conversion_rate_contacted = float(np.mean(y_contacted))

    # Expected profit from model probabilities
    expected_profit_total = float(np.sum(p_contacted * V - C))

    # Realized profit from actual outcomes
    realized_profit_total = float(np.sum(y_contacted * V - C))

    realized_profit_per_customer = realized_profit_total / n_total

    return {
        "V": V,
        "C": C,
        "threshold": threshold,
        "n_total": n_total,
        "n_contacted": n_contacted,
        "contact_rate": contact_rate,
        "observed_conversion_rate_contacted": observed_conversion_rate_contacted,
        "expected_profit_total": expected_profit_total,
        "realized_profit_total": realized_profit_total,
        "realized_profit_per_customer": realized_profit_per_customer,
    }


def main():
    cfg = Config()
    ensure_dirs(cfg)

    model_name, best_params = load_best_tuned_model_info(cfg.tables_dir)
    tuned_params = strip_model_prefix(best_params)
    model_tag = sanitize_name(model_name)

    print(f"Running profit-threshold sensitivity for tuned model: {model_name}")
    print("Loaded tuned params:", tuned_params)

    df = load_data(cfg)
    X_train, X_test, y_train, y_test = split_month_holdout(df, cfg)

    pipe = build_trained_pipeline(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        cfg=cfg,
        tuned_params=tuned_params,
    )

    y_score = predict_scores(pipe, X_test)

    # You can adjust these scenarios if needed
    scenarios = [
        {"V": 50, "C": 1},
        {"V": 50, "C": 2},
        {"V": 50, "C": 5},
        {"V": 100, "C": 1},
        {"V": 100, "C": 2},
        {"V": 100, "C": 5},
        {"V": 200, "C": 2},
        {"V": 200, "C": 5},
        {"V": 200, "C": 10},
    ]

    rows = []
    for sc in scenarios:
        row = evaluate_profit_scenario(
            y_true=y_test,
            y_score=y_score,
            V=sc["V"],
            C=sc["C"],
        )
        row["model"] = model_name
        row["split"] = "month_aware"
        row["tuned_params_json"] = json.dumps(tuned_params)
        rows.append(row)

    results_df = pd.DataFrame(rows)

    out_path = f"{cfg.tables_dir}/profit_sensitivity_{model_tag}.csv"
    results_df.to_csv(out_path, index=False)

    print("\nSaved profit sensitivity results:")
    print(f" - {out_path}")

    print("\nProfit sensitivity summary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()