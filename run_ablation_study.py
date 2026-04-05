import os
import json
import copy
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from config import Config
from data_utils import load_data, split_month_holdout
from preprocess import build_preprocessor
from models import build_model, needs_scaling
from metrics import compute_basic_metrics, precision_lift_at_k


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


def apply_feature_ablation_to_split(X_train, X_test, cfg):
    """
    Apply ablation only to model features, not to the raw dataframe used for splitting.
    This ensures 'month' can still be used for time-aware splitting even if it is
    listed among contact-history variables.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    drop_cols = []

    if not cfg.use_macro:
        drop_cols.extend([c for c in cfg.macro_cols if c in X_train.columns])

    if not cfg.use_contact_history:
        drop_cols.extend([c for c in cfg.contact_history_cols if c in X_train.columns])

    drop_cols = sorted(set(drop_cols))

    if drop_cols:
        X_train = X_train.drop(columns=drop_cols, errors="ignore")
        X_test = X_test.drop(columns=drop_cols, errors="ignore")

    return X_train, X_test


def evaluate_ablation_setting(setting_name, cfg, model_name, tuned_params):
    """
    Run one ablation setting on month-aware split.
    """
    # Load the full leakage-safe data first
    df = load_data(cfg, apply_feature_ablation=False)

    # Split first, while 'month' is still available
    X_train, X_test, y_train, y_test = split_month_holdout(df, cfg)

    # Then apply ablation only to model inputs
    X_train, X_test = apply_feature_ablation_to_split(X_train, X_test, cfg)

    pipe = build_trained_pipeline(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        cfg=cfg,
        tuned_params=tuned_params,
    )

    y_score = predict_scores(pipe, X_test)
    basic = compute_basic_metrics(y_test, y_score)

    row = {
        "setting": setting_name,
        "model": model_name,
        "use_macro": cfg.use_macro,
        "use_contact_history": cfg.use_contact_history,
        "n_features_after_ablation": X_train.shape[1],
        "train_n": len(y_train),
        "test_n": len(y_test),
        "train_pos_rate": float(np.mean(y_train)),
        "test_pos_rate": float(np.mean(y_test)),
        **basic,
    }

    for kf in cfg.k_fracs:
        pk = precision_lift_at_k(y_test, y_score, kf)
        row[f"precision_at_{int(kf*100)}"] = pk["precision_at_k"]
        row[f"lift_at_{int(kf*100)}"] = pk["lift_at_k"]

    return row


def main():
    base_cfg = Config()
    ensure_dirs(base_cfg)

    model_name, best_params = load_best_tuned_model_info(base_cfg.tables_dir)
    tuned_params = strip_model_prefix(best_params)

    print(f"Running ablation study for final tuned model: {model_name}")
    print("Loaded tuned params:", tuned_params)

    settings = [
        ("full_model", True, True),
        ("no_macro", False, True),
        ("no_contact_history", True, False),
        ("core_only", False, False),
    ]

    rows = []

    for setting_name, use_macro, use_contact_history in settings:
        cfg = copy.deepcopy(base_cfg)
        cfg.use_macro = use_macro
        cfg.use_contact_history = use_contact_history

        print(
            f"\nRunning setting: {setting_name} "
            f"(use_macro={use_macro}, use_contact_history={use_contact_history})"
        )

        row = evaluate_ablation_setting(
            setting_name=setting_name,
            cfg=cfg,
            model_name=model_name,
            tuned_params=tuned_params,
        )
        rows.append(row)

    results_df = pd.DataFrame(rows)

    order = ["full_model", "no_macro", "no_contact_history", "core_only"]
    results_df["setting"] = pd.Categorical(
        results_df["setting"],
        categories=order,
        ordered=True
    )
    results_df = results_df.sort_values("setting").reset_index(drop=True)

    out_path = f"{base_cfg.tables_dir}/ablation_results_{model_name}.csv"
    results_df.to_csv(out_path, index=False)

    print("\nSaved ablation results:")
    print(f" - {out_path}")

    print("\nAblation summary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()