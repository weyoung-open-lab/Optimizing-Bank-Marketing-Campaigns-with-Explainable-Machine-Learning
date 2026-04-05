import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X, scale_numeric: bool = False):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    cat_tf = OneHotEncoder(handle_unknown="ignore")
    num_tf = StandardScaler() if scale_numeric else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_tf, cat_cols),
            ("num", num_tf, num_cols),
        ],
        remainder="drop"
    )
    return preprocessor, cat_cols, num_cols


def get_feature_names(preprocessor):
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return None