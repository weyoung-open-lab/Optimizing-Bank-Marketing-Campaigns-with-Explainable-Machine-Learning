import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
)


# =========================
# Basic predictive metrics
# =========================
def compute_basic_metrics(y_true, y_score):
    """
    Metrics based on predicted probabilities / scores.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
    }


# =========================
# Threshold-based reporting metrics
# =========================
def compute_threshold_metrics(y_true, y_score, threshold=0.5):
    """
    Metrics based on a reporting threshold.
    This is for confusion-matrix-style reporting,
    not the main business decision rule.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    y_pred = (y_score >= threshold).astype(int)

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def find_best_f1_threshold(y_true, y_score, thresholds=None):
    """
    Select a reporting threshold by maximizing F1 on validation data.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_row = None
    best_f1 = -1.0

    for th in thresholds:
        row = compute_threshold_metrics(y_true, y_score, threshold=float(th))
        if row["f1"] > best_f1:
            best_f1 = row["f1"]
            best_row = row

    return best_row


# =========================
# Top-K policy metrics
# =========================
def precision_lift_at_k(y_true, y_score, k_frac):
    """
    Evaluate Top-K targeting policy.
    Customers are ranked by score, and only the top K fraction is contacted.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    n = len(y_true)
    k = max(1, int(np.floor(n * k_frac)))

    idx = np.argsort(-y_score)[:k]

    base_rate = y_true.mean()
    precision_at_k = y_true[idx].mean()
    lift_at_k = precision_at_k / base_rate if base_rate > 0 else np.nan

    return {
        "k_frac": float(k_frac),
        "k": int(k),
        "base_rate": float(base_rate),
        "conversion_rate_contacted": float(precision_at_k),
        "precision_at_k": float(precision_at_k),
        "lift_at_k": float(lift_at_k),
    }


def build_lift_curve(y_true, y_score, k_fracs):
    """
    Build a list of Top-K policy results across multiple K fractions.
    """
    rows = []
    for kf in k_fracs:
        rows.append(precision_lift_at_k(y_true, y_score, kf))
    return rows


# =========================
# Cost-benefit / profit policy
# =========================
def expected_profit_per_customer(y_score, benefit, cost):
    """
    Expected profit per customer:
        E[pi] = p * V - C
    """
    y_score = np.asarray(y_score)
    return y_score * benefit - cost


def apply_profit_threshold_policy(y_score, benefit, cost):
    """
    Contact if expected profit is positive:
        p * V - C > 0
    equivalent to:
        p > C / V
    """
    y_score = np.asarray(y_score)
    threshold = cost / benefit
    contact = (y_score > threshold).astype(int)
    return contact, float(threshold)


def evaluate_profit_policy(y_true, y_score, benefit, cost):
    """
    Evaluate a cost-benefit targeting policy using true outcomes.
    Realized profit:
      +benefit for a contacted true positive
      -cost for every contacted customer
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    contact, threshold = apply_profit_threshold_policy(y_score, benefit, cost)

    n_contacted = int(contact.sum())
    if n_contacted == 0:
        return {
            "benefit": float(benefit),
            "cost": float(cost),
            "profit_threshold": float(threshold),
            "n_contacted": 0,
            "contact_rate": 0.0,
            "conversion_rate_contacted": 0.0,
            "realized_profit_total": 0.0,
            "realized_profit_per_customer": 0.0,
        }

    contacted_true = y_true[contact == 1]
    conversions = int(contacted_true.sum())

    realized_profit_total = conversions * benefit - n_contacted * cost
    realized_profit_per_customer = realized_profit_total / len(y_true)
    conversion_rate_contacted = conversions / n_contacted

    return {
        "benefit": float(benefit),
        "cost": float(cost),
        "profit_threshold": float(threshold),
        "n_contacted": int(n_contacted),
        "contact_rate": float(n_contacted / len(y_true)),
        "conversion_rate_contacted": float(conversion_rate_contacted),
        "realized_profit_total": float(realized_profit_total),
        "realized_profit_per_customer": float(realized_profit_per_customer),
    }


def evaluate_profit_scenarios(y_true, y_score, benefit_values, cost_values):
    """
    Evaluate multiple (benefit, cost) combinations.
    """
    rows = []
    for benefit in benefit_values:
        for cost in cost_values:
            if benefit <= 0:
                continue
            rows.append(evaluate_profit_policy(y_true, y_score, benefit, cost))
    return rows