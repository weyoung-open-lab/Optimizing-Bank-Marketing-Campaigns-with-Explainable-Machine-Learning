import pandas as pd

from config import Config
from data_utils import (
    load_data,
    split_month_holdout,
    split_random_stratified,
    get_split_stats,
)
from models import PRIMARY_MODELS, EXTENDED_MODELS, get_available_model_names
from evaluate import evaluate_models_on_split, select_best_model
from select_shap_model import rank_models_for_shap


def filter_available_models(candidate_models, available_models):
    """
    Keep only models that are available in the current environment.
    """
    return [m for m in candidate_models if m in available_models]


def main():
    cfg = Config()
    cfg.ensure_dirs()

    if cfg.verbose:
        print("Loading data...")
    df = load_data(cfg)

    # available models in current environment
    available_models = get_available_model_names(cfg)

    # main benchmark models
    primary_models = filter_available_models(PRIMARY_MODELS, available_models)
    extended_models = filter_available_models(EXTENDED_MODELS, available_models)


    all_models = primary_models + extended_models

    if cfg.verbose:
        print("Available models:")
        print(all_models)

    # =========================
    # Split 1: month-aware
    # =========================
    if cfg.verbose:
        print("\nRunning month-aware split...")
    X_train_m, X_test_m, y_train_m, y_test_m = split_month_holdout(df, cfg)

    model_metrics_m, policy_metrics_m, _ = evaluate_models_on_split(
        model_names=all_models,
        X_train=X_train_m,
        y_train=y_train_m,
        X_test=X_test_m,
        y_test=y_test_m,
        cfg=cfg,
        split_name="month_aware",
    )

    split_stats_m = get_split_stats(
        y_train=y_train_m,
        y_test=y_test_m,
        split_name="month_aware",
    )
    split_stats_m["test_months"] = ",".join(cfg.month_aware_test_months)

    # =========================
    # Split 2: random stratified
    # =========================
    if cfg.verbose:
        print("\nRunning random stratified split...")
    X_train_r, X_test_r, y_train_r, y_test_r = split_random_stratified(df, cfg)

    model_metrics_r, policy_metrics_r, _ = evaluate_models_on_split(
        model_names=all_models,
        X_train=X_train_r,
        y_train=y_train_r,
        X_test=X_test_r,
        y_test=y_test_r,
        cfg=cfg,
        split_name="random",
    )

    split_stats_r = get_split_stats(
        y_train=y_train_r,
        y_test=y_test_r,
        split_name="random",
    )
    split_stats_r["test_months"] = ""

    # =========================
    # Merge results
    # =========================
    model_metrics_df = pd.concat(
        [model_metrics_m, model_metrics_r],
        ignore_index=True
    )

    policy_metrics_df = pd.concat(
        [policy_metrics_m, policy_metrics_r],
        ignore_index=True
    )

    split_stats_df = pd.DataFrame([split_stats_m, split_stats_r])

    # =========================
    # Select best model
    # =========================
    best_model_info = select_best_model(
        policy_metrics_df=policy_metrics_df,
        cfg=cfg,
        split_name="month_aware",
    )
    best_model_df = pd.DataFrame([best_model_info])

    # =========================
    # Select SHAP-best model
    # =========================
    shap_ranking_df, best_shap_model = rank_models_for_shap(
        model_metrics_path=f"{cfg.tables_dir}/model_metrics.csv",
        policy_metrics_path=f"{cfg.tables_dir}/policy_metrics.csv",
        output_path=f"{cfg.tables_dir}/shap_model_ranking.csv",
        k_fracs=cfg.k_fracs,
    )

    best_shap_model_df = pd.DataFrame([{
        "best_shap_model": best_shap_model
    }])

    # =========================
    # Save outputs
    # =========================
    model_metrics_path = f"{cfg.tables_dir}/model_metrics.csv"
    policy_metrics_path = f"{cfg.tables_dir}/policy_metrics.csv"
    split_stats_path = f"{cfg.tables_dir}/split_stats.csv"
    best_model_path = f"{cfg.tables_dir}/best_model_summary.csv"
    best_shap_model_path = f"{cfg.tables_dir}/best_shap_model_summary.csv"


    model_metrics_df.to_csv(model_metrics_path, index=False)
    policy_metrics_df.to_csv(policy_metrics_path, index=False)
    split_stats_df.to_csv(split_stats_path, index=False)
    best_model_df.to_csv(best_model_path, index=False)
    best_shap_model_df.to_csv(best_shap_model_path, index=False)
    # =========================
    # Console summary
    # =========================
    print("\nSaved outputs:")
    print(f" - {model_metrics_path}")
    print(f" - {policy_metrics_path}")
    print(f" - {split_stats_path}")
    print("\nBest policy model (single-rule benchmark):")
    print(best_model_df.to_string(index=False))

    print("\nBest SHAP model (composite ranking):")
    print(best_shap_model_df.to_string(index=False))

    print("\nBest model selected from month-aware split:")
    print(best_model_df.to_string(index=False))


if __name__ == "__main__":
    main()