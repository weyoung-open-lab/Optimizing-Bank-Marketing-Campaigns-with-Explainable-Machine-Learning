import os
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[-1]

    elif is_linear_model(model_name):
        masker = shap.maskers.Independent(X_bg_t)
        explainer = shap.LinearExplainer(model, masker)
        shap_values = explainer.shap_values(X_topk_t)
        expected_value = explainer.expected_value

    else:
        masker = shap.maskers.Independent(X_bg_t)
        explainer = shap.Explainer(model.predict, masker)
        explanation = explainer(X_topk_t)
        shap_values = explanation.values
        expected_value = None

    return shap_values, expected_value, X_topk_t, feature_names


def cluster_in_shap_space(shap_values, n_segments, random_state):
    km = KMeans(
        n_clusters=n_segments,
        n_init=20,
        random_state=random_state,
    )
    labels = km.fit_predict(shap_values)

    sil = np.nan
    if len(np.unique(labels)) > 1 and len(shap_values) > n_segments:
        sil = silhouette_score(shap_values, labels)

    return labels, sil


def top_drivers_for_segment(shap_values_seg, feature_names, top_n=8):
    mean_abs = np.mean(np.abs(shap_values_seg), axis=0)
    order = np.argsort(-mean_abs)[:top_n]
    drivers = []
    for j in order:
        drivers.append(feature_names[j] if feature_names is not None else f"f{j}")
    return drivers


def recommend_segment_action(drivers, observed_conversion_rate=None, mean_predicted_score=None):
    """
    Generate a more differentiated, manager-friendly recommendation
    based on dominant SHAP drivers and segment performance.
    """
    d = " ".join(drivers)

    has_macro = any(
        x in d for x in [
            "nr.employed", "euribor3m", "cons.conf.idx",
            "cons.price.idx", "emp.var.rate"
        ]
    )
    has_response = any(
        x in d for x in [
            "pdays", "poutcome_success", "poutcome_failure", "previous"
        ]
    )
    has_channel = any(x in d for x in ["contact_cellular", "contact_telephone"])
    has_age = "age" in d

    # high-confidence responsive segment
    if has_response and has_macro and mean_predicted_score is not None and mean_predicted_score >= 0.75:
        return (
            "Treat this group as a high-confidence priority segment. "
            "Use early outreach and response-based follow-up, as both macro conditions "
            "and prior responsiveness strongly support targeting."
        )

    # response-driven core segment
    if has_response and has_macro and observed_conversion_rate is not None and observed_conversion_rate >= 0.65:
        return (
            "Prioritize this segment in the main campaign wave. "
            "Use timely follow-up and response-based messaging, as prior campaign signals "
            "and current economic conditions jointly support high conversion potential."
        )

    # macro-sensitive but less response-proven
    if has_macro and ("poutcome_success" not in d) and observed_conversion_rate is not None and observed_conversion_rate < 0.60:
        return (
            "Use more selective and carefully framed outreach for this segment. "
            "Prioritization appears to rely more on macroeconomic and profile signals "
            "than on strong prior response evidence, so contact efficiency should be monitored closely."
        )

    # channel-sensitive
    if has_channel and not has_response:
        return (
            "Differentiate contact strategy by communication channel. "
            "Adjust message format and timing to channel-related preferences and monitor response quality."
        )

    # age-sensitive fallback
    if has_age:
        return (
            "Consider age-sensitive value framing and tailored communication, "
            "while maintaining selective outreach based on the dominant decision drivers."
        )

    return (
        "Apply differentiated targeting based on the dominant SHAP drivers and monitor "
        "segment-level conversion outcomes to refine campaign actions."
    )


def save_segment_local_case(
    expected_value,
    shap_row,
    x_row,
    feature_names,
    out_path,
    title=None,
):
    exp = shap.Explanation(
        values=shap_row,
        base_values=expected_value,
        data=x_row,
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.bar(exp, max_display=15, show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    cfg = Config()
    ensure_dirs(cfg)

    model_name, best_params = load_best_tuned_model_info(cfg.tables_dir)
    tuned_params = strip_model_prefix(best_params)
    model_tag = sanitize_name(model_name)

    n_segments = cfg.n_segments
    local_cases_per_segment = cfg.local_cases_per_segment

    print(f"Running SHAP segmentation for tuned model: {model_name}")
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
    y_topk = y_test.iloc[topk_idx].copy()
    y_score_topk = y_score[topk_idx]

    shap_values, expected_value, X_topk_t, feature_names = compute_shap_values(
        pipe=pipe,
        model_name=model_name,
        X_train=X_train,
        X_topk=X_topk,
        cfg=cfg,
    )

    labels, sil = cluster_in_shap_space(
        shap_values=shap_values,
        n_segments=n_segments,
        random_state=cfg.random_state,
    )

    rows = []
    X_topk_reset = X_topk.reset_index(drop=True)
    y_topk_reset = y_topk.reset_index(drop=True)

    for seg_id in range(n_segments):
        mask = labels == seg_id
        n_seg = int(mask.sum())
        if n_seg == 0:
            continue

        seg_shap = shap_values[mask]
        seg_y = np.asarray(y_topk_reset)[mask]
        seg_scores = np.asarray(y_score_topk)[mask]

        drivers = top_drivers_for_segment(seg_shap, feature_names, top_n=8)
        recommendation = recommend_segment_action(
            drivers=drivers,
            observed_conversion_rate=float(np.mean(seg_y)),
            mean_predicted_score=float(np.mean(seg_scores)),
        )

        rows.append({
            "split": "month_aware",
            "model": model_name,
            "topk_frac": cfg.shap_topk_frac,
            "segment_id": seg_id,
            "n_customers": n_seg,
            "share_in_topk": n_seg / len(X_topk_reset),
            "observed_conversion_rate": float(np.mean(seg_y)),
            "mean_predicted_score": float(np.mean(seg_scores)),
            "top_drivers": "; ".join(drivers),
            "recommendation": recommendation,
            "silhouette_overall": sil,
            "tuned_params_json": json.dumps(tuned_params),
        })

        seg_indices = np.where(mask)[0]
        order_within_seg = np.argsort(-seg_scores)
        chosen = seg_indices[order_within_seg[:local_cases_per_segment]]

        for j, ii in enumerate(chosen, start=1):
            out_path = f"{cfg.figures_dir}/segment_{seg_id}_local_case_{j}_{model_tag}.png"
            save_segment_local_case(
                expected_value=expected_value,
                shap_row=shap_values[ii],
                x_row=X_topk_t[ii],
                feature_names=feature_names,
                out_path=out_path,
                title=f"Segment {seg_id} - Local SHAP Case {j} ({model_name})",
            )

    segment_df = pd.DataFrame(rows).sort_values("segment_id")
    segment_summary_path = f"{cfg.tables_dir}/segment_summary_topk_{model_tag}.csv"
    segment_df.to_csv(segment_summary_path, index=False)

    topk_with_segment = X_topk_reset.copy()
    topk_with_segment["y_true"] = y_topk_reset.values
    topk_with_segment["y_score"] = y_score_topk
    topk_with_segment["segment_id"] = labels
    topk_with_segment["model"] = model_name
    topk_with_segment.to_csv(
        f"{cfg.tables_dir}/topk_population_with_segments_{model_tag}.csv",
        index=False
    )

    print("\nSaved segmentation outputs:")
    print(f" - {segment_summary_path}")
    print(f" - {cfg.tables_dir}/topk_population_with_segments_{model_tag}.csv")

    for seg_id in range(n_segments):
        for j in range(1, local_cases_per_segment + 1):
            path = f"{cfg.figures_dir}/segment_{seg_id}_local_case_{j}_{model_tag}.png"
            if os.path.exists(path):
                print(f" - {path}")

    print("\nSegment summary:")
    print(segment_df.to_string(index=False))


if __name__ == "__main__":
    main()