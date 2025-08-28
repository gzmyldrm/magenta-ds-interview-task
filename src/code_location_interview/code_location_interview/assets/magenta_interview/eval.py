import logging
import sys
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score,confusion_matrix

from dagster import MetadataValue, Output, asset, get_dagster_logger,AssetIn



log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)
    
group_name = "evaluation"

# thresholds
topks = [0.05, 0.10, 0.20]
default_decision_threshold = 0.50
promotion_min_auc = 0.70
budget_frac = 0.10

@asset(
    name="eval_report",
    group_name=group_name,
    ins={
        "rf_model": AssetIn(),
        "X_test": AssetIn(),
        "y_test": AssetIn(),
        "feature_selection": AssetIn(),
    },
)
def eval_report(rf_model, X_test: pd.DataFrame, y_test, feature_selection) -> Output[pd.DataFrame]:
    # Scores (prob of positive class)

    X = X_test[feature_selection]
    y = pd.Series(y_test)
    scores = rf_model.predict_proba(X)[:, 1]

    n = len(y)
    pos_rate = float(y.mean())
    total_pos = max(int(y.sum()), 1)

    # AUCs
    pr_auc = float(average_precision_score(y, scores))
    roc_auc = float(roc_auc_score(y, scores))

    # Rank metrics
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y.to_numpy()[order]
    cum_pos = np.cumsum(y_sorted)

    rows = []
    md: dict[str, Union[float, MetadataValue]] = {
        "BaselinePosRate": pos_rate,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
    }
    for kf in topks:
        k = max(1, int(round(kf * n)))
        tp = int(cum_pos[k - 1])
        precision = tp / k
        recall = tp / total_pos
        lift = precision / max(pos_rate, 1e-12)
        rows.append({
            "k_frac": kf, "k_n": k,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "lift_at_k": lift,
        })
        pct = int(kf * 100)
        md[f"P@{pct}%"] = precision
        md[f"R@{pct}%"] = recall
        md[f"Lift@{pct}%"] = lift

    metrics = pd.DataFrame(rows)
    md["Precision_Recall_Lift_table"] = MetadataValue.md(metrics.to_markdown(index=False))

    # Confusion matrix at budget (top 10%)
    kb = max(1, int(round(budget_frac * n)))
    tp_b = int(cum_pos[kb - 1])
    fp_b = kb - tp_b
    fn_b = total_pos - tp_b
    tn_b = (n - total_pos) - fp_b
    cm = pd.DataFrame(
        [[tn_b, fp_b], [fn_b, tp_b]],
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"],
    )
    md[f"ConfusionMatrix@{int(budget_frac*100)}%"] = MetadataValue.md(cm.to_markdown())

    md[f"ConfusionMatrix@{int(budget_frac*100)}%"] = MetadataValue.md(cm.to_markdown())

    # Standard confusion matrix at fixed probability threshold
    preds_thr = (scores >= default_decision_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds_thr).ravel()
    cm_thr_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"],
    )
    md["DecisionThreshold"] = default_decision_threshold
    md["ConfusionMatrix@Threshold"] = MetadataValue.md(cm_thr_df.to_markdown())

    # Normalized confusion matrix
    cm_thr_row_norm = cm_thr_df.div(cm_thr_df.sum(axis=1), axis=0).round(4)
    md["ConfusionMatrixRowNorm@Threshold"] = MetadataValue.md(cm_thr_row_norm.to_markdown())


    return Output(metrics, metadata=md)