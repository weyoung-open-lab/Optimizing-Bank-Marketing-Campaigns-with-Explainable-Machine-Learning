from typing import Dict
from sklearn.base import clone

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier


# =========================
# Model groups
# =========================
PRIMARY_MODELS = [
    "LogReg",
    "NaiveBayes",
    "KNN",
    "DecisionTree",
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]

EXTENDED_MODELS = [
    "AdaBoost",
    "GradientBoosting",
    "MLP",
]


# =========================
# Model registry
# =========================
def get_model_registry(cfg) -> Dict[str, object]:
    rs = cfg.random_state
    registry = {}

    # ===== Baselines =====
    registry["LogReg"] = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=rs,
    )
    registry["NaiveBayes"] = GaussianNB()
    registry["KNN"] = KNeighborsClassifier(
        n_neighbors=50,
        weights="distance",
    )

    # ===== Tree =====
    registry["DecisionTree"] = DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=rs,
    )
    registry["RandomForest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=rs,
    )
    registry["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=rs,
    )

    # ===== Boosting =====
    registry["AdaBoost"] = AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=rs,
    )
    registry["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=rs,
    )

    # ===== Neural =====
    registry["MLP"] = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=rs,
    )

    # ===== XGBoost =====
    try:
        from xgboost import XGBClassifier

        registry["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=rs,
        )
    except Exception:
        pass

    # ===== LightGBM =====
    try:
        from lightgbm import LGBMClassifier

        registry["LightGBM"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            random_state=rs,
        )
    except Exception:
        pass

    # ===== CatBoost =====
    try:
        from catboost import CatBoostClassifier

        registry["CatBoost"] = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            random_seed=rs,
        )
    except Exception:
        pass

    return registry


# =========================
# Utilities
# =========================
def build_model(model_name: str, cfg):
    registry = get_model_registry(cfg)
    if model_name not in registry:
        raise ValueError(f"Model not found: {model_name}")
    return clone(registry[model_name])


def get_available_model_names(cfg):
    return list(get_model_registry(cfg).keys())


def needs_scaling(model_name: str) -> bool:
    return model_name in ["LogReg", "KNN", "MLP"]


def is_tree_based_model(model_name: str) -> bool:
    return model_name in {
        "DecisionTree",
        "RandomForest",
        "ExtraTrees",
        "AdaBoost",
        "GradientBoosting",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    }


def is_linear_model(model_name: str) -> bool:
    return model_name in {"LogReg"}