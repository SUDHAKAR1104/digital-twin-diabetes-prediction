"""
train.py
Trains and evaluates the glucose-spike risk model, then saves everything the dashboard needs.

Run from the repo root:
    python src/train.py

Outputs
    models/risk_model.pkl         model + calibrator + thresholds (used by src/predict.py)
    docs/metrics.json             all numbers for the README / slides
    docs/results_table.md         ready-to-paste results table
    docs/roc_curve.png, confusion_matrix.png, feature_importance.png, risk_levels.png

Method
    * features use PAST data only (src/features.py)
    * the split is BY PATIENT: no patient appears in more than one of train / val / test
    * 3 models are compared: a simple glucose rule, logistic regression, gradient boosting
    * probabilities are calibrated on the validation patients
    * Low / Moderate / High thresholds are chosen on the validation patients
    * the test patients are used once, at the end
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (FEATURE_COLS, FEATURE_LABELS, LABEL_COL, MONOTONIC,  # noqa: E402
                      build_features)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_patients.csv"
MODEL_PATH = ROOT / "models" / "risk_model.pkl"
DOCS = ROOT / "docs"
SEED = 42


# ----------------------------------------------------------------------------- helpers
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.argmax(f1))])


def operating_point(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"threshold": round(float(thr), 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate(name, y_val, p_val, y_test, p_test):
    thr = best_f1_threshold(y_val, p_val)          # threshold picked on VALIDATION patients
    res = {"roc_auc": round(float(roc_auc_score(y_test, p_test)), 4),
           "pr_auc": round(float(average_precision_score(y_test, p_test)), 4)}
    res.update(operating_point(y_test, p_test, thr))
    print(f"  {name:<28} ROC-AUC {res['roc_auc']:.3f} | PR-AUC {res['pr_auc']:.3f} | "
          f"recall {res['recall']:.3f} | precision {res['precision']:.3f} | F1 {res['f1']:.3f}")
    return res


# ----------------------------------------------------------------------------- main
def main():
    if not DATA.exists():
        sys.exit(f"Dataset not found: {DATA}\nRun first:  python src/generate_data.py")

    print("Loading data and building features (past data only)...")
    raw = pd.read_csv(DATA, parse_dates=["timestamp"])
    X = build_features(raw)
    print(f"  {len(X):,} rows, {X['patient_id'].nunique()} patients, "
          f"{len(FEATURE_COLS)} features, spike rate {X[LABEL_COL].mean():.1%}")

    # ---- split BY PATIENT ----
    pids = np.array(sorted(X["patient_id"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(pids)
    n = len(pids)
    n_test, n_val = max(1, round(n * 0.16)), max(1, round(n * 0.14))
    test_p, val_p, train_p = pids[:n_test], pids[n_test:n_test + n_val], pids[n_test + n_val:]
    tr, va, te = (X[X["patient_id"].isin(s)] for s in (train_p, val_p, test_p))
    assert not (set(train_p) & set(val_p)) and not (set(train_p) & set(test_p)) \
        and not (set(val_p) & set(test_p)), "patient leakage between splits!"
    print(f"Patient split: train {len(train_p)} | validation {len(val_p)} | test {len(test_p)} "
          f"(no patient appears in two sets)")
    y_tr, y_va, y_te = tr[LABEL_COL].values, va[LABEL_COL].values, te[LABEL_COL].values

    # ---- models ----
    print("\nTraining models...")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    lr.fit(tr[FEATURE_COLS], y_tr)

    cst = [MONOTONIC.get(c, 0) for c in FEATURE_COLS]
    gb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_depth=6,
                                        l2_regularization=1.0, monotonic_cst=cst,
                                        random_state=SEED)
    gb.fit(tr[FEATURE_COLS], y_tr)

    # ---- calibration (Platt scaling) on the validation patients ----
    raw_val = gb.predict_proba(va[FEATURE_COLS])[:, 1]
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(logit(raw_val).reshape(-1, 1), y_va)

    def calibrated(model_probs):
        return calibrator.predict_proba(logit(model_probs).reshape(-1, 1))[:, 1]

    p_val = calibrated(raw_val)
    p_test = calibrated(gb.predict_proba(te[FEATURE_COLS])[:, 1])

    # ---- Low / Moderate / High thresholds from the validation patients ----
    thr_high = best_f1_threshold(y_va, p_val)
    order = np.argsort(p_val)
    cum_missed = np.cumsum(y_va[order]) / max(y_va.sum(), 1)       # share of spikes below each prob
    ok = np.where(cum_missed <= 0.05)[0]                            # "Low" may hide <= 5% of spikes
    thr_low = float(p_val[order][ok[-1]]) if len(ok) else 0.1
    if thr_low >= thr_high:
        thr_low = thr_high / 2
    thresholds = {"low": round(thr_low, 3), "high": round(thr_high, 3)}

    def level(p):
        return np.where(p < thresholds["low"], "Low",
                        np.where(p < thresholds["high"], "Moderate", "High"))

    # ---- evaluation on the TEST patients (used once) ----
    print("\nResults on the held-out TEST patients:")
    glucose_rule = evaluate("Baseline: glucose rule", y_va, va["glucose"].values,
                            y_te, te["glucose"].values)
    log_reg = evaluate("Logistic regression", y_va, lr.predict_proba(va[FEATURE_COLS])[:, 1],
                       y_te, lr.predict_proba(te[FEATURE_COLS])[:, 1])
    boosted = evaluate("Gradient boosting (final)", y_va, p_val, y_te, p_test)

    lvl_test = level(p_test)
    by_level = {}
    for name in ("Low", "Moderate", "High"):
        m = lvl_test == name
        by_level[name] = {"rows": int(m.sum()),
                          "share_of_rows": round(float(m.mean()), 4),
                          "observed_spike_rate": round(float(y_te[m].mean()), 4) if m.any() else None}
    alert = (p_test >= thresholds["low"]).astype(int)               # Moderate or High
    spikes_caught = float(alert[y_te == 1].mean())
    print(f"\nLevels: Low < {thresholds['low']} <= Moderate < {thresholds['high']} <= High")
    for k, v in by_level.items():
        print(f"  {k:<9} {v['share_of_rows']:.1%} of rows -> observed spike rate {v['observed_spike_rate']:.1%}")
    print(f"  Spikes flagged Moderate or High: {spikes_caught:.1%}")

    # Early-warning cases: glucose is STILL below 180 now, so the model must really look ahead.
    # (If glucose is already above 180, "spike in the next 2 h" is almost automatic.)
    early = te["glucose"].values < 180
    ew_alert = operating_point(y_te[early], p_test[early], thresholds["low"])
    ew_high = operating_point(y_te[early], p_test[early], thresholds["high"])
    early_warning = {
        "rows": int(early.sum()),
        "share_of_test_rows": round(float(early.mean()), 4),
        "spike_rate": round(float(y_te[early].mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_te[early], p_test[early])), 4),
        "pr_auc": round(float(average_precision_score(y_te[early], p_test[early])), 4),
        "alert_moderate_or_high": ew_alert,
        "alert_high_only": ew_high,
    }
    print(f"\nEarly-warning cases (glucose still < 180 mg/dL, {early.mean():.0%} of test rows):")
    print(f"  ROC-AUC {early_warning['roc_auc']:.3f} | PR-AUC {early_warning['pr_auc']:.3f} | "
          f"Moderate-or-High catches {ew_alert['recall']:.1%} of upcoming spikes "
          f"(precision {ew_alert['precision']:.1%})")

    # ---- permutation importance (global explainability) ----
    print("\nComputing feature importance...")
    sub = te.sample(min(20000, len(te)), random_state=SEED)
    pi = permutation_importance(gb, sub[FEATURE_COLS], sub[LABEL_COL], scoring="roc_auc",
                                n_repeats=3, random_state=SEED)
    imp = (pd.DataFrame({"feature": FEATURE_COLS, "importance": pi.importances_mean})
           .sort_values("importance", ascending=False))
    imp["label"] = imp["feature"].map(FEATURE_LABELS)
    # features that share a label (e.g. the two time-of-day features) are added together
    imp = (imp.groupby("label", sort=False)
              .agg(importance=("importance", "sum"), feature=("feature", ", ".join))
              .reset_index().sort_values("importance", ascending=False))
    top_imp = imp.head(10)
    for _, r in top_imp.head(5).iterrows():
        print(f"  {r['label']:<32} {r['importance']:.4f}")

    # ---- plots ----
    DOCS.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 5))
    for name, score in [("Glucose rule", te["glucose"].values),
                        ("Logistic regression", lr.predict_proba(te[FEATURE_COLS])[:, 1]),
                        ("Gradient boosting", p_test)]:
        fpr, tpr, _ = roc_curve(y_te, score)
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, score):.2f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate (recall)")
    plt.title("ROC curve - test patients"); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(DOCS / "roc_curve.png", dpi=150); plt.close()

    cm = np.array([[boosted["tn"], boosted["fp"]], [boosted["fn"], boosted["tp"]]])
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    plt.xticks([0, 1], ["No spike", "Spike"]); plt.yticks([0, 1], ["No spike", "Spike"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion matrix (threshold {boosted['threshold']:.2f})"); plt.tight_layout()
    plt.savefig(DOCS / "confusion_matrix.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4.5))
    t = top_imp.iloc[::-1]
    plt.barh(t["label"], t["importance"], color="#1f77b4")
    plt.xlabel("Drop in ROC-AUC when the feature is shuffled")
    plt.title("Most important factors"); plt.tight_layout()
    plt.savefig(DOCS / "feature_importance.png", dpi=150); plt.close()

    plt.figure(figsize=(5.5, 4))
    names = list(by_level)
    rates = [by_level[k]["observed_spike_rate"] or 0 for k in names]
    plt.bar(names, [r * 100 for r in rates], color=["#2e9e5b", "#f0a202", "#d62828"])
    plt.ylabel("Real spike rate (%)"); plt.title("Predicted level vs what happened")
    plt.tight_layout(); plt.savefig(DOCS / "risk_levels.png", dpi=150); plt.close()

    # ---- save everything ----
    metrics = {
        "dataset": {"rows": int(len(X)), "patients": int(n), "features": len(FEATURE_COLS),
                    "spike_rate": round(float(X[LABEL_COL].mean()), 4),
                    "train_patients": int(len(train_p)), "val_patients": int(len(val_p)),
                    "test_patients": int(len(test_p))},
        "models": {"glucose_rule": glucose_rule, "logistic_regression": log_reg,
                   "gradient_boosting": boosted},
        "early_warning": early_warning,
        "levels": {"thresholds": thresholds, "on_test": by_level,
                   "spikes_flagged_moderate_or_high": round(spikes_caught, 4)},
        "top_features": [{"feature": r["feature"], "label": r["label"],
                          "importance": round(float(r["importance"]), 4)}
                         for _, r in top_imp.iterrows()],
        "sklearn_version": sklearn.__version__,
    }
    (DOCS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    rows = [("Baseline: glucose rule", glucose_rule), ("Logistic Regression", log_reg),
            ("Gradient Boosting (final)", boosted)]
    md = ["| Model | ROC-AUC | Recall | Precision | F1 |", "|---|---|---|---|---|"]
    for name, r in rows:
        md.append(f"| {name} | {r['roc_auc']:.3f} | {r['recall']:.3f} | {r['precision']:.3f} | {r['f1']:.3f} |")
    md += ["", "Early-warning cases only (glucose still below 180 mg/dL at prediction time):", "",
           "| ROC-AUC | Recall (Moderate or High) | Precision (Moderate or High) |", "|---|---|---|",
           f"| {early_warning['roc_auc']:.3f} | {ew_alert['recall']:.3f} | {ew_alert['precision']:.3f} |"]
    md += ["", "| Predicted level | Share of readings | Real spike rate |", "|---|---|---|"]
    for k, v in by_level.items():
        md.append(f"| {k} | {v['share_of_rows']:.1%} | {v['observed_spike_rate']:.1%} |")
    (DOCS / "results_table.md").write_text("\n".join(md) + "\n")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": gb, "calibrator": calibrator, "feature_cols": FEATURE_COLS,
                 "thresholds": thresholds,
                 "medians": X.loc[X["patient_id"].isin(train_p), FEATURE_COLS].median().to_dict(),
                 "sklearn_version": sklearn.__version__}, MODEL_PATH)

    print(f"\nSaved: {MODEL_PATH.relative_to(ROOT)}, docs/metrics.json, docs/results_table.md and 4 plots")
    print("Reminder: these results are on SYNTHETIC data. They show the pipeline works, not clinical performance.")


if __name__ == "__main__":
    main()
