"""
metrics.py
----------
Evaluation metrics for BP estimation.

BHS standard grades:
    Grade A: ≤5 mmHg in ≥60%, ≤10 mmHg in ≥85%, ≤15 mmHg in ≥95%
    Grade B: ≤5 mmHg in ≥50%, ≤10 mmHg in ≥75%, ≤15 mmHg in ≥90%
    Grade C: ≤5 mmHg in ≥40%, ≤10 mmHg in ≥65%, ≤15 mmHg in ≥85%

AAMI standard: mean error ≤5 mmHg, SD ≤8 mmHg
"""

import numpy as np


def bhs_grade(errors: np.ndarray) -> str:
    abs_err = np.abs(errors)
    p5  = (abs_err <= 5).mean()
    p10 = (abs_err <= 10).mean()
    p15 = (abs_err <= 15).mean()

    if p5 >= 0.60 and p10 >= 0.85 and p15 >= 0.95:
        return "A"
    elif p5 >= 0.50 and p10 >= 0.75 and p15 >= 0.90:
        return "B"
    elif p5 >= 0.40 and p10 >= 0.65 and p15 >= 0.85:
        return "C"
    else:
        return "D"


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    pred, target: (N, 2)  columns [ΔSBP, ΔDBP]
    Returns dict of metrics.
    """
    err_sbp = pred[:, 0] - target[:, 0]
    err_dbp = pred[:, 1] - target[:, 1]

    mae_sbp = np.abs(err_sbp).mean()
    mae_dbp = np.abs(err_dbp).mean()
    combined_mae = (mae_sbp + mae_dbp) / 2

    return {
        # MAE
        "mae_sbp":      float(mae_sbp),
        "mae_dbp":      float(mae_dbp),
        "combined_mae": float(combined_mae),
        # RMSE
        "rmse_sbp":     float(np.sqrt((err_sbp**2).mean())),
        "rmse_dbp":     float(np.sqrt((err_dbp**2).mean())),
        # Mean error + SD (AAMI)
        "me_sbp":       float(err_sbp.mean()),
        "me_dbp":       float(err_dbp.mean()),
        "std_sbp":      float(err_sbp.std()),
        "std_dbp":      float(err_dbp.std()),
        # BHS grades
        "bhs_sbp":      bhs_grade(err_sbp),
        "bhs_dbp":      bhs_grade(err_dbp),
        # Correlation
        "r_sbp":        float(np.corrcoef(pred[:, 0], target[:, 0])[0, 1]),
        "r_dbp":        float(np.corrcoef(pred[:, 1], target[:, 1])[0, 1]),
    }


def print_metrics(pred: np.ndarray, target: np.ndarray, label: str = ""):
    m = compute_metrics(pred, target)
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Results")
    print(f"  {'Metric':<20} {'SBP':>10} {'DBP':>10}")
    print(f"  {'-'*42}")
    print(f"  {'MAE (mmHg)':<20} {m['mae_sbp']:>10.2f} {m['mae_dbp']:>10.2f}")
    print(f"  {'Combined MAE':<20} {m['combined_mae']:>10.2f}")
    print(f"  {'RMSE (mmHg)':<20} {m['rmse_sbp']:>10.2f} {m['rmse_dbp']:>10.2f}")
    print(f"  {'Mean Error':<20} {m['me_sbp']:>10.2f} {m['me_dbp']:>10.2f}")
    print(f"  {'Std Error':<20} {m['std_sbp']:>10.2f} {m['std_dbp']:>10.2f}")
    print(f"  {'Pearson r':<20} {m['r_sbp']:>10.3f} {m['r_dbp']:>10.3f}")
    print(f"  {'BHS Grade':<20} {m['bhs_sbp']:>10} {m['bhs_dbp']:>10}")
    print(f"\n  Benchmark (Schlesinger et al.): combined MAE = 4.68 mmHg")
    return m
