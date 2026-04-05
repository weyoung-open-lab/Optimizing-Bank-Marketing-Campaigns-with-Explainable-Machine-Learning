import os
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from config import Config
from data_utils import load_data, split_month_holdout
from preprocess import build_preprocessor, get_feature_names
from models import build_model, needs_scaling, is_tree_based_model, is_linear_model


def ensure_dirs(cfg):
    os.makedirs(cfg.figures_dir, exist_ok=True)
    os.makedirs(cfg.tables_dir, exist_ok=True)


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


def sanitize_filename(name: str) -> str:
    return (
        name.replace("__", "_")
            .replace(".", "_")
            .replace("/", "_")
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


def select_top_k(y_score, k_frac):
    y_score = np.asarray(y_score)
    n = len(y_score)
    k = max(1, int(np.floor(n * k_frac)))
    idx = np.argsort(-y_score)[:k]
    return idx, k


def compute_shap_values(pipe, model_name, X_train, X_topk, cfg):
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]

    X_train_t = pre.transform(X_train)
    X_topk_t = pre.transform(X_topk)
    feature_names = get_feature_names(pre)

    bg_n = min(cfg.shap_background_size, X_train_t.shape[0])
    rng = np.random.default_rng(cfg.random_state)
    bg_idx = rng.choice(X_train_t.shape[0], size=bg_n, replace=False)
    X_bg_t = X_train_t[bg_idx]

    if is_tree_based_model(model_name):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_topk_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
    elif is_linear_model(model_name):
        masker = shap.maskers.Independent(X_bg_t)
        explainer = shap.LinearExplainer(model, masker)
        shap_values = explainer.shap_values(X_topk_t)
    else:
        masker = shap.maskers.Independent(X_bg_t)
        explainer = shap.Explainer(model.predict, masker)
        shap_values = explainer(X_topk_t).values

    return shap_values, X_topk_t, feature_names


def save_dependence_plot(shap_values, X_topk_t, feature_names, feature_name, out_path):
    if feature_name not in feature_names:
        raise ValueError(f"Feature '{feature_name}' not found in transformed features.")

    plt.figure()
    shap.dependence_plot(
        ind=feature_name,
        shap_values=shap_values,
        features=X_topk_t,
        feature_names=feature_names,
        interaction_index="auto",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    cfg = Config()
    ensure_dirs(cfg)

    model_name, best_params = load_best_tuned_model_info(cfg.tables_dir)
    tuned_params = strip_model_prefix(best_params)
    model_tag = sanitize_filename(model_name)

    target_features = [
        "num__nr.employed",
        "num__pdays",
        "num__age",
        "num__euribor3m",
    ]

    print(f"Running SHAP dependence plots for tuned model: {model_name}")
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

    topk_idx, k = select_top_k(y_score, cfg.shap_topk_frac)
    X_topk = X_test.iloc[topk_idx].copy()

    shap_values, X_topk_t, feature_names = compute_shap_values(
        pipe=pipe,
        model_name=model_name,
        X_train=X_train,
        X_topk=X_topk,
        cfg=cfg,
    )

    for feat in target_features:
        out_name = sanitize_filename(feat)
        out_path = f"{cfg.figures_dir}/dependence_{out_name}_{model_tag}.png"
        save_dependence_plot(
            shap_values=shap_values,
            X_topk_t=X_topk_t,
            feature_names=feature_names,
            feature_name=feat,
            out_path=out_path,
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()