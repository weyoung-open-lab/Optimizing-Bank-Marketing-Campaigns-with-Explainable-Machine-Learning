import pandas as pd
from sklearn.model_selection import train_test_split


# =========================
# Load data
# =========================
def load_data(cfg, apply_feature_ablation=True):
    """
    Read raw dataset, convert target to 0/1,
    drop leakage columns, and optionally apply feature ablation.
    """
    df = pd.read_csv(cfg.data_path, sep=cfg.sep)

    # strip column names just in case
    df.columns = df.columns.str.strip()

    if cfg.target not in df.columns:
        raise ValueError(
            f"Target column '{cfg.target}' not found.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Check cfg.data_path and cfg.sep."
        )

    # target: 'yes'/'no' -> 1/0
    df[cfg.target] = (df[cfg.target] == "yes").astype(int)

    # drop leakage columns (e.g., duration)
    for c in cfg.leakage_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # apply feature ablation only if requested
    if apply_feature_ablation:
        df = apply_ablation(df, cfg)

    return df


# =========================
# Feature ablation
# =========================
def apply_ablation(df, cfg):
    """
    Optionally remove macro variables or contact-history variables.
    """
    df = df.copy()

    if not cfg.use_macro:
        drop_cols = [c for c in cfg.macro_cols if c in df.columns]
        df = df.drop(columns=drop_cols)

    if not cfg.use_contact_history:
        drop_cols = [c for c in cfg.contact_history_cols if c in df.columns]
        df = df.drop(columns=drop_cols)

    return df


# =========================
# Random stratified split
# =========================
def split_random_stratified(df, cfg):
    """
    Stratified random split (benchmark setting).
    """
    X = df.drop(columns=[cfg.target])
    y = df[cfg.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.random_state,
    )

    return X_train, X_test, y_train, y_test


# =========================
# Month-aware split
# =========================
def split_month_holdout(df, cfg):
    """
    Time-aware split using 'month'.
    Train on earlier months, test on later months.
    """
    if "month" not in df.columns:
        raise ValueError("Column 'month' not found. Cannot perform month-aware split.")

    test_months = cfg.month_aware_test_months
    if not test_months:
        raise ValueError("cfg.month_aware_test_months is empty.")

    test_mask = df["month"].isin(test_months)

    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    X_train = train_df.drop(columns=[cfg.target])
    y_train = train_df[cfg.target]

    X_test = test_df.drop(columns=[cfg.target])
    y_test = test_df[cfg.target]

    return X_train, X_test, y_train, y_test


# =========================
# Split statistics
# =========================
def get_split_stats(y_train, y_test, split_name=""):
    """
    Return basic statistics for sanity check.
    """
    return {
        "split": split_name,
        "train_n": len(y_train),
        "test_n": len(y_test),
        "train_pos_rate": float(y_train.mean()),
        "test_pos_rate": float(y_test.mean()),
    }