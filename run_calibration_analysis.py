import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve

from config import Config
from data_utils import load_data, split_month_holdout
from preprocess import build_preprocessor
from models import build_model, needs_scaling
from metrics import compute_basic_metrics


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


def save_calibration_plot(y_true, y_score, out_path, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_score,
        n_bins=n_bins,
        strategy="quantile",
    )

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return prob_true, prob_pred


def build_bin_summary(y_true, y_score, n_bins=10):
    """
    Quantile-bin calibration summary table.
    """
    df = pd.DataFrame({
        "y_true": np.asarray(y_true).astype(int),
        "y_score": np.asarray(y_score),
    }).copy()

    # rank-based quantile bins avoids duplicate-edge issues better than qcut on raw scores alone
    df["rank"] = df["y_score"].rank(method="first")
    df["bin"] = pd.qcut(df["rank"], q=n_bins, labels=False)

    out = (
        df.groupby("bin", as_index=False)
        .agg(
            n=("y_true", "size"),
            mean_predicted_probability=("y_score", "mean"),
            observed_frequency=("y_true", "mean"),
        )
    )

    out["calibration_gap"] = out["mean_predicted_probability"] - out["observed_frequency"]
    return out


def main():
    cfg = Config()
    ensure_dirs(cfg)

    model_name, best_params = load_best_tuned_model_info(cfg.tables_dir)
    tuned_params = strip_model_prefix(best_params)
    model_tag = sanitize_name(model_name)

    print(f"Running calibration analysis for tuned model: {model_name}")
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
    basic = compute_basic_metrics(y_test, y_score)

    plot_path = f"{cfg.figures_dir}/calibration_curve_{model_tag}.png"
    bin_path = f"{cfg.tables_dir}/calibration_bins_{model_tag}.csv"
    summary_path = f"{cfg.tables_dir}/calibration_summary_{model_tag}.csv"

    prob_true, prob_pred = save_calibration_plot(
        y_true=y_test,
        y_score=y_score,
        out_path=plot_path,
        n_bins=10,
    )

    bin_df = build_bin_summary(
        y_true=y_test,
        y_score=y_score,
        n_bins=10,
    )
    bin_df.to_csv(bin_path, index=False)

    summary_df = pd.DataFrame([{
        "model": model_name,
        "split": "month_aware",
        "roc_auc": basic["roc_auc"],
        "pr_auc": basic["pr_auc"],
        "brier": basic["brier"],
        "mean_predicted_probability": float(np.mean(y_score)),
        "observed_positive_rate": float(np.mean(y_test)),
        "tuned_params_json": json.dumps(tuned_params),
    }])
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved calibration outputs:")
    print(f" - {plot_path}")
    print(f" - {bin_path}")
    print(f" - {summary_path}")

    print("\nCalibration summary:")
    print(summary_df.to_string(index=False))
    print("\nCalibration bins:")
    print(bin_df.to_string(index=False))


if __name__ == "__main__":
    main()