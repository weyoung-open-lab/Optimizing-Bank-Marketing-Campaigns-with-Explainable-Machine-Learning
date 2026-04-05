import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


from config import Config
from data_utils import load_data, split_month_holdout, split_random_stratified
from preprocess import build_preprocessor
from models import build_model, needs_scaling, get_available_model_names
from evaluate import evaluate_models_on_split
from select_shap_model import rank_models_for_shap

warnings.filterwarnings("ignore")


# =========================
# Utility: Top-K scorer
# =========================
def precision_at_k_from_scores(y_true, y_score, k_frac):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    n = len(y_true)
    k = max(1, int(np.floor(n * k_frac)))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].mean())


def mean_precision_at_ks(y_true, y_score, k_fracs):
    vals = [precision_at_k_from_scores(y_true, y_score, k) for k in k_fracs]
    return float(np.mean(vals))


def topk_policy_scorer_factory(k_fracs):
    """
    Callable scorer for RandomizedSearchCV.
    Signature must be: scorer(estimator, X, y)
    """
    def scorer(estimator, X, y):
        if hasattr(estimator, "predict_proba"):
            y_score = estimator.predict_proba(X)[:, 1]
        else:
            scores = estimator.decision_function(X)
            y_score = 1.0 / (1.0 + np.exp(-scores))

        return mean_precision_at_ks(y, y_score, k_fracs)

    return scorer


# =========================
# Parameter spaces
# =========================
def get_param_distributions(model_name, random_state):
    """
    Lightweight tuning spaces for the most competitive models.
    Parameter names are for Pipeline usage, hence model__ prefix.
    """
    spaces = {
        "GradientBoosting": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__min_samples_leaf": [1, 3, 5, 10],
            "model__subsample": [0.7, 0.85, 1.0],
        },
        "CatBoost": {
            "model__iterations": [200, 300, 500, 700],
            "model__depth": [4, 5, 6, 7],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__l2_leaf_reg": [1, 3, 5, 7, 10],
        },
        "KNN": {
            "model__n_neighbors": [15, 25, 35, 50, 75],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        "RandomForest": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [6, 8, 10, 12, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", None],
        },
        "XGBoost": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        },
        "LightGBM": {
            "model__n_estimators": [200, 300, 500, 700],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__num_leaves": [15, 31, 63],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_samples": [10, 20, 40],
        },
        "ExtraTrees": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [6, 8, 10, 12, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", None],
        },
        "LogReg": {
            "model__C": [0.01, 0.1, 1.0, 3.0, 10.0],
            "model__solver": ["lbfgs"],
        },
    }

    return spaces.get(model_name, {})


# =========================
# Build untuned pipeline
# =========================
def build_pipeline_for_model(model_name, X_train, y_train, cfg):
    model = build_model(model_name, cfg)
    scale_numeric = needs_scaling(model_name)
    preprocessor, _, _ = build_preprocessor(X_train, scale_numeric=scale_numeric)

    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", model),
    ])
    return pipe


# =========================
# Tune one model
# =========================
def tune_one_model(model_name, X_train, y_train, cfg, n_iter=12):
    pipe = build_pipeline_for_model(model_name, X_train, y_train, cfg)
    param_dist = get_param_distributions(model_name, cfg.random_state)

    if not param_dist:
        raise ValueError(f"No tuning space defined for model: {model_name}")

    scorer = topk_policy_scorer_factory(cfg.k_fracs)

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=cfg.random_state,
    )

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scorer,
        n_jobs=-1,
        cv=cv,
        refit=True,
        random_state=cfg.random_state,
        verbose=1,
    )

    search.fit(X_train, y_train)

    return {
        "model_name": model_name,
        "best_estimator": search.best_estimator_,
        "best_score_cv": float(search.best_score_),
        "best_params": search.best_params_,
        "cv_results": search.cv_results_,
    }


# =========================
# Evaluate already tuned models
# =========================
def evaluate_pretrained_models_on_split(trained_pipelines, X_test, y_test, split_name, cfg):
    from metrics import compute_basic_metrics, precision_lift_at_k

    model_metric_rows = []
    policy_metric_rows = []

    for model_name, pipe in trained_pipelines.items():
        if hasattr(pipe, "predict_proba"):
            y_score = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)
            y_score = 1.0 / (1.0 + np.exp(-scores))

        basic = compute_basic_metrics(y_test, y_score)
        model_metric_rows.append({
            "split": split_name,
            "model": model_name,
            **basic
        })

        for kf in cfg.k_fracs:
            row = precision_lift_at_k(y_test, y_score, kf)
            policy_metric_rows.append({
                "split": split_name,
                "model": model_name,
                **row
            })

    return pd.DataFrame(model_metric_rows), pd.DataFrame(policy_metric_rows)


# =========================
# Main
# =========================
def main():
    cfg = Config()
    cfg.ensure_dirs()

    ranking_path = f"{cfg.tables_dir}/shap_model_ranking.csv"
    if not os.path.exists(ranking_path):
        raise FileNotFoundError(
            f"{ranking_path} not found. Please run benchmark and SHAP model ranking first."
        )

    ranking_df = pd.read_csv(ranking_path)
    top5_models = ranking_df["model"].head(5).tolist()

    available_models = get_available_model_names(cfg)
    top5_models = [m for m in top5_models if m in available_models]

    print("Top 5 models selected for light tuning:")
    print(top5_models)

    # main training set for tuning = month-aware train
    df = load_data(cfg)
    X_train_m, X_test_m, y_train_m, y_test_m = split_month_holdout(df, cfg)
    X_train_r, X_test_r, y_train_r, y_test_r = split_random_stratified(df, cfg)

    tuned_pipelines = {}
    tuning_rows = []

    for model_name in top5_models:
        print(f"\nTuning model: {model_name}")
        result = tune_one_model(
            model_name=model_name,
            X_train=X_train_m,
            y_train=y_train_m,
            cfg=cfg,
            n_iter=12,   # light tuning
        )

        tuned_pipelines[model_name] = result["best_estimator"]

        tuning_rows.append({
            "model": model_name,
            "best_score_cv": result["best_score_cv"],
            "best_params_json": json.dumps(result["best_params"]),
        })

    tuning_summary_df = pd.DataFrame(tuning_rows)
    tuning_summary_path = f"{cfg.tables_dir}/tuning_summary_top5.csv"
    tuning_summary_df.to_csv(tuning_summary_path, index=False)

    # evaluate tuned models on both splits
    model_metrics_m, policy_metrics_m = evaluate_pretrained_models_on_split(
        tuned_pipelines, X_test_m, y_test_m, "month_aware", cfg
    )
    model_metrics_r, policy_metrics_r = evaluate_pretrained_models_on_split(
        tuned_pipelines, X_test_r, y_test_r, "random", cfg
    )

    model_metrics_df = pd.concat([model_metrics_m, model_metrics_r], ignore_index=True)
    policy_metrics_df = pd.concat([policy_metrics_m, policy_metrics_r], ignore_index=True)

    model_metrics_path = f"{cfg.tables_dir}/model_metrics_tuned_top5.csv"
    policy_metrics_path = f"{cfg.tables_dir}/policy_metrics_tuned_top5.csv"

    model_metrics_df.to_csv(model_metrics_path, index=False)
    policy_metrics_df.to_csv(policy_metrics_path, index=False)

    # rank again for final SHAP model
    ranking_tuned_df, best_model = rank_models_for_shap(
        model_metrics_path=model_metrics_path,
        policy_metrics_path=policy_metrics_path,
        output_path=f"{cfg.tables_dir}/shap_model_ranking_tuned_top5.csv",
        k_fracs=cfg.k_fracs,
    )

    best_model_df = pd.DataFrame([{"best_shap_model_tuned": best_model}])
    best_model_path = f"{cfg.tables_dir}/best_shap_model_tuned.csv"
    best_model_df.to_csv(best_model_path, index=False)

    print("\nSaved tuned outputs:")
    print(f" - {tuning_summary_path}")
    print(f" - {model_metrics_path}")
    print(f" - {policy_metrics_path}")
    print(f" - {cfg.tables_dir}/shap_model_ranking_tuned_top5.csv")
    print(f" - {best_model_path}")

    print("\nBest SHAP model after light tuning:")
    print(best_model)

    print("\nTop 5 tuned ranking:")
    print(ranking_tuned_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()