from dataclasses import dataclass
from typing import Tuple
import os


@dataclass
class Config:
    # =========================
    # Basic data settings
    # =========================
    data_path: str = "D:\\project1\\pythonProject\\thesis-sgh\\bank-additional-full.csv"#dataset
    sep: str = ";"
    target: str = "y"

    # Leakage-safe setting
    leakage_cols: Tuple[str, ...] = ("duration",)

    # =========================
    # Output settings
    # =========================
    output_dir: str = "outputs"
    tables_dir: str = "outputs/tables"
    figures_dir: str = "outputs/figures"
    logs_dir: str = "outputs/logs"

    # =========================
    # Randomness / reproducibility
    # =========================
    random_state: int = 42

    # =========================
    # Split settings
    # =========================
    test_size: float = 0.2
    month_aware_test_months: Tuple[str, ...] = ("sep", "oct", "nov", "dec")

    # =========================
    # Policy settings
    # =========================
    k_fracs: Tuple[float, ...] = (0.05, 0.10, 0.20)
    best_model_selection_k: float = 0.10

    # Profit-threshold sensitivity analysis
    # Contact if p > C / V
    profit_benefit_values: Tuple[float, ...] = (50.0, 100.0, 200.0)
    contact_cost_values: Tuple[float, ...] = (1.0, 2.0, 5.0)

    # =========================
    # Feature group toggles (for ablation)
    # =========================
    use_macro: bool = True
    use_contact_history: bool = True

    macro_cols: Tuple[str, ...] = (
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    )

    contact_history_cols: Tuple[str, ...] = (
        "contact",
        "month",
        "day_of_week",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
    )

    # =========================
    # Calibration settings
    # =========================
    use_calibration: bool = False
    calibration_method: str = "sigmoid"   # "sigmoid" or "isotonic"
    calibration_cv: int = 3

    # =========================
    # SHAP / segmentation settings
    # =========================
    shap_topk_frac: float = 0.10
    shap_background_size: int = 1000
    n_segments: int = 3
    local_cases_per_segment: int = 2

    # =========================
    # Runtime / logging
    # =========================
    save_lift_curves: bool = True
    save_calibration_curves: bool = True
    verbose: bool = True

    def ensure_dirs(self) -> None:
        """Create output directories if they do not exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)