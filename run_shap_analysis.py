import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.pipeline import Pipeline

from config import Config
from data_utils import load_data, split_month_holdout
from preprocess import build_preprocessor, get_feature_names
from models import build_model, needs_scaling, is_tree_based_model, is_linear_model


def ensure_shap_dirs(cfg):
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
    """
    Convert pipeline parameter names like model__depth to depth.
    """
    return {k.replace("model__", ""): v for k, v in best_params.items()}


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def select_top_k(y_score, k_frac):
    """
    Select indices of Top-K customers by predicted score.
    """
    y_score = np.asarray(y_score)
    n = len(y_score)
    k = max(1, int(np.floor(n * k_frac)))
    idx = np.argsort(-y_score)[:k]
    return idx, k


def build_trained_pipeline(model_name, X_train, y_train, cfg, tuned_params=None):
    """
    Train the final selected tuned model under the unified preprocessing pipeline.
    """
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
    """
    Predict probability scores for the positive class.
    """
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]

    scores = pipe.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def compute_shap_values(pipe, model_name, X_train, X_topk, cfg):
    """
    Compute SHAP values for the Top-K population.
    Supports tree-based and linear models.
    """
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]

    X_train_t = pre.transform(X_train)
    X_topk_t = pre.transform(X_topk)

    feature_names = get_feature_names(pre)

    # sample background for efficiency
    bg_n = min(cfg.shap_background_size, X_train_t.shape[0])
    rng = np.random.default_rng(cfg.random_state)
    bg_idx = rng.choice(X_train_t.shape[0], size=bg_n, replace=False)
    X_bg_t = X_train_t[bg_idx]

    if is_tree_based_model(model_name):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_topk_t)

        # shap may return list for binary classification in some versions
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[-1]

    elif is_linear_model(model_name):
        masker = shap.maskers.Independent(X_bg_t)
        explainer = shap.LinearExplainer(model, masker)
        shap_values = explainer.shap_values(X_topk_t)
        expected_value = explainer.expected_value

    else:
        # fallback for unsupported models
        masker = shap.maskers.Independent(X_bg_t)
        explainer = shap.Explainer(model.predict, masker)
        shap_values = explainer(X_topk_t).values
        expected_value = None

    return shap_values, expected_value, X_topk_t, feature_names


def save_shap_summary_plot(shap_values, X_topk_t, feature_names, out_path):
    """
    Save SHAP summary plot.
    """
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_topk_t,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_feature_importance_table(shap_values, feature_names, out_path):
    """
    Save mean absolute SHAP importance table.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(mean_abs))]

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    df.to_csv(out_path, index=False)
    return df


def save_topk_outputs(X_topk, y_topk, y_score_topk, shap_values, feature_names, cfg, model_name):
    """
    Save Top-K original rows and SHAP matrix.
    """
    model_tag = sanitize_name(model_name)

    topk_df = X_topk.copy()
    topk_df["y_true"] = np.asarray(y_topk)
    topk_df["y_score"] = np.asarray(y_score_topk)
    topk_df.to_csv(f"{cfg.tables_dir}/topk_population_{model_tag}.csv", index=False)

    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names if feature_names is not None else None
    )
    shap_df.to_csv(f"{cfg.tables_dir}/topk_shap_values_{model_tag}.csv", index=False)


def save_local_cases(pipe, model_name, X_train, X_topk, shap_values, X_topk_t, feature_names, cfg):
    """
    Save a few local SHAP bar plots for the highest-scored Top-K cases.
    """
    model_tag = sanitize_name(model_name)

    # save first 3 local explanations by default
    n_cases = min(3, len(X_topk))

    # recompute expected value if needed
    _, expected_value, _, _ = compute_shap_values(pipe, model_name, X_train, X_topk.iloc[:1], cfg)

    for i in range(n_cases):
        exp = shap.Explanation(
            values=shap_values[i],
            base_values=expected_value,
            data=X_topk_t[i],
            feature_names=feature_names,
        )
        plt.figure()
        shap.plots.bar(exp, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(
            f"{cfg.figures_dir}/local_case_{i+1}_{model_tag}.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()


def main():
    cfg = Config()
    ensure_shap_dirs(cfg)

    # =========================
    # Final tuned SHAP model
    # =========================
    model_name, best_params = load_best_tuned_model_info(cfg.tables_dir)
    tuned_params = strip_model_prefix(best_params)
    model_tag = sanitize_name(model_name)

    print(f"Running SHAP analysis for tuned model: {model_name}")
    print("Loaded tuned params:", tuned_params)

    # =========================
    # Data
    # =========================
    df = load_data(cfg)
    X_train, X_test, y_train, y_test = split_month_holdout(df, cfg)

    # =========================
    # Train tuned model
    # =========================
    pipe = build_trained_pipeline(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        cfg=cfg,
        tuned_params=tuned_params,
    )

    # =========================
    # Scores on test set
    # =========================
    y_score = predict_scores(pipe, X_test)

    # =========================
    # Top-K decision population
    # =========================
    topk_idx, k = select_top_k(y_score, cfg.shap_topk_frac)

    X_topk = X_test.iloc[topk_idx].copy()
    y_topk = y_test.iloc[topk_idx].copy()
    y_score_topk = y_score[topk_idx]

    print(f"Selected Top-K population: {k} customers ({cfg.shap_topk_frac:.0%})")

    # =========================
    # SHAP
    # =========================
    shap_values, expected_value, X_topk_t, feature_names = compute_shap_values(
        pipe=pipe,
        model_name=model_name,
        X_train=X_train,
        X_topk=X_topk,
        cfg=cfg,
    )

    # =========================
    # Save outputs
    # =========================
    summary_plot_path = f"{cfg.figures_dir}/shap_summary_topk_month_aware_{model_tag}.png"
    importance_path = f"{cfg.tables_dir}/shap_feature_importance_topk_{model_tag}.csv"
    summary_csv_path = f"{cfg.tables_dir}/shap_analysis_summary_{model_tag}.csv"

    save_shap_summary_plot(
        shap_values=shap_values,
        X_topk_t=X_topk_t,
        feature_names=feature_names,
        out_path=summary_plot_path,
    )

    importance_df = save_feature_importance_table(
        shap_values=shap_values,
        feature_names=feature_names,
        out_path=importance_path,
    )

    save_topk_outputs(
        X_topk=X_topk,
        y_topk=y_topk,
        y_score_topk=y_score_topk,
        shap_values=shap_values,
        feature_names=feature_names,
        cfg=cfg,
        model_name=model_name,
    )

    save_local_cases(
        pipe=pipe,
        model_name=model_name,
        X_train=X_train,
        X_topk=X_topk,
        shap_values=shap_values,
        X_topk_t=X_topk_t,
        feature_names=feature_names,
        cfg=cfg,
    )

    # save a tiny summary file
    summary_df = pd.DataFrame([{
        "model": model_name,
        "split": "month_aware",
        "topk_frac": cfg.shap_topk_frac,
        "n_topk": k,
        "mean_predicted_score_topk": float(np.mean(y_score_topk)),
        "observed_conversion_topk": float(np.mean(y_topk)),
        "tuned_params_json": json.dumps(tuned_params),
    }])
    summary_df.to_csv(summary_csv_path, index=False)

    print("\nSaved SHAP outputs:")
    print(f" - {summary_plot_path}")
    print(f" - {importance_path}")
    print(f" - {cfg.tables_dir}/topk_population_{model_tag}.csv")
    print(f" - {cfg.tables_dir}/topk_shap_values_{model_tag}.csv")
    print(f" - {summary_csv_path}")
    print(f" - {cfg.figures_dir}/local_case_1_{model_tag}.png")
    print(f" - {cfg.figures_dir}/local_case_2_{model_tag}.png")
    print(f" - {cfg.figures_dir}/local_case_3_{model_tag}.png")

    print("\nTop 10 SHAP features:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()