import pandas as pd
import numpy as np


def minmax_scale(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Min-max normalize a metric to [0, 1].
    If higher_is_better=False, smaller values get higher normalized scores.
    """
    s = series.astype(float).copy()
    smin, smax = s.min(), s.max()

    if np.isclose(smax, smin):
        return pd.Series(np.ones(len(s)), index=s.index)

    if higher_is_better:
        return (s - smin) / (smax - smin)
    else:
        return (smax - s) / (smax - smin)


def build_month_policy_summary(policy_df: pd.DataFrame, k_fracs=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """
    Aggregate month-aware policy performance across multiple K values.
    """
    sub = policy_df[
        (policy_df["split"] == "month_aware") &
        (policy_df["k_frac"].isin(k_fracs))
    ].copy()

    if sub.empty:
        raise ValueError("No month_aware policy rows found for the specified K values.")

    agg = (
        sub.groupby("model", as_index=False)
        .agg(
            precision_at_k_mean=("precision_at_k", "mean"),
            precision_at_k_std=("precision_at_k", "std"),
            lift_at_k_mean=("lift_at_k", "mean"),
            lift_at_k_std=("lift_at_k", "std"),
        )
    )

    # std may be NaN if only one K is present
    agg["precision_at_k_std"] = agg["precision_at_k_std"].fillna(0.0)
    agg["lift_at_k_std"] = agg["lift_at_k_std"].fillna(0.0)

    return agg


def build_metric_summary(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract month-aware predictive/probability-quality metrics.
    """
    sub = model_df[model_df["split"] == "month_aware"].copy()
    if sub.empty:
        raise ValueError("No month_aware rows found in model metrics.")

    cols = ["model", "roc_auc", "pr_auc", "brier"]
    return sub[cols].copy()


def build_random_stability_summary(policy_df: pd.DataFrame, k_fracs=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """
    Optional stability reference from random split.
    """
    sub = policy_df[
        (policy_df["split"] == "random") &
        (policy_df["k_frac"].isin(k_fracs))
    ].copy()

    if sub.empty:
        # return empty frame; caller can ignore
        return pd.DataFrame(columns=["model", "precision_at_k_mean_random", "lift_at_k_mean_random"])

    agg = (
        sub.groupby("model", as_index=False)
        .agg(
            precision_at_k_mean_random=("precision_at_k", "mean"),
            lift_at_k_mean_random=("lift_at_k", "mean"),
        )
    )
    return agg


def rank_models_for_shap(
    model_metrics_path: str,
    policy_metrics_path: str,
    output_path: str = "outputs/tables/shap_model_ranking.csv",
    k_fracs=(0.05, 0.10, 0.20),
    weights=None,
):
    """
    Rank models for SHAP selection using a composite score.

    Default idea:
    - month-aware policy mean precision across multiple K: strongest weight
    - month-aware policy mean lift across multiple K
    - month-aware PR-AUC
    - month-aware Brier (smaller is better)
    - optional random-split policy mean as a light stability bonus
    """
    if weights is None:
        weights = {
            "precision_at_k_mean": 0.40,
            "lift_at_k_mean": 0.20,
            "pr_auc": 0.20,
            "brier": 0.15,
            "precision_at_k_mean_random": 0.05,
        }

    model_df = pd.read_csv(model_metrics_path)
    policy_df = pd.read_csv(policy_metrics_path)

    month_policy = build_month_policy_summary(policy_df, k_fracs=k_fracs)
    month_metrics = build_metric_summary(model_df)
    random_stab = build_random_stability_summary(policy_df, k_fracs=k_fracs)

    df = month_policy.merge(month_metrics, on="model", how="inner")
    df = df.merge(random_stab, on="model", how="left")

    if "precision_at_k_mean_random" not in df.columns:
        df["precision_at_k_mean_random"] = np.nan

    # fill missing random stability with month-aware mean, or 0 if needed
    df["precision_at_k_mean_random"] = df["precision_at_k_mean_random"].fillna(df["precision_at_k_mean"])

    # normalize metrics
    df["score_precision_mean"] = minmax_scale(df["precision_at_k_mean"], higher_is_better=True)
    df["score_lift_mean"] = minmax_scale(df["lift_at_k_mean"], higher_is_better=True)
    df["score_pr_auc"] = minmax_scale(df["pr_auc"], higher_is_better=True)
    df["score_brier"] = minmax_scale(df["brier"], higher_is_better=False)
    df["score_precision_random"] = minmax_scale(df["precision_at_k_mean_random"], higher_is_better=True)

    # composite score
    df["shap_selection_score"] = (
        weights["precision_at_k_mean"] * df["score_precision_mean"] +
        weights["lift_at_k_mean"] * df["score_lift_mean"] +
        weights["pr_auc"] * df["score_pr_auc"] +
        weights["brier"] * df["score_brier"] +
        weights["precision_at_k_mean_random"] * df["score_precision_random"]
    )

    # tie-breakers:
    # 1) higher composite
    # 2) higher month-aware precision mean
    # 3) higher PR-AUC
    # 4) lower Brier
    df = df.sort_values(
        by=["shap_selection_score", "precision_at_k_mean", "pr_auc", "brier"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1)

    cols = [
        "rank",
        "model",
        "shap_selection_score",
        "precision_at_k_mean",
        "precision_at_k_std",
        "lift_at_k_mean",
        "lift_at_k_std",
        "pr_auc",
        "roc_auc",
        "brier",
        "precision_at_k_mean_random",
        "score_precision_mean",
        "score_lift_mean",
        "score_pr_auc",
        "score_brier",
        "score_precision_random",
    ]
    df = df[cols]

    df.to_csv(output_path, index=False)

    best_model = df.iloc[0]["model"]
    return df, best_model


if __name__ == "__main__":
    ranking_df, best_model = rank_models_for_shap(
        model_metrics_path="outputs/tables/model_metrics.csv",
        policy_metrics_path="outputs/tables/policy_metrics.csv",
        output_path="outputs/tables/shap_model_ranking.csv",
        k_fracs=(0.05, 0.10, 0.20),
    )

    print("Best model for SHAP:")
    print(best_model)
    print("\nTop 5 ranked models:")
    print(ranking_df.head(5).to_string(index=False))