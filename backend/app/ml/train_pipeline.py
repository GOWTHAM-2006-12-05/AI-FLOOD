"""
train_pipeline.py — End-to-end training pipeline for the flood prediction system.

Usage:
    python -m backend.app.ml.train_pipeline

Steps:
    1. Generate synthetic data (or load CSV if available)
    2. Preprocess: split, scale, SMOTE
    3. Train XGBoost with cross-validation
    4. Optionally tune hyperparameters (Optuna)
    5. Train LSTM on temporal sequences
    6. Find optimal ensemble alpha
    7. Evaluate ensemble on held-out test set
    8. Save models, scaler, and metrics
    9. Run drift detection baseline

Target: 90%+ validation accuracy via ensemble of XGBoost + LSTM.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from backend.app.ml.ensemble import (
    EnsembleResult,
    StackingEnsemble,
    batch_ensemble_predict,
    blend_predictions,
    classify_risk,
    optimise_alpha,
    predict_ensemble,
)
from backend.app.ml.lstm_model import (
    LSTMTrainResult,
    create_sequences,
    generate_synthetic_sequences,
    predict_flood_proba_lstm,
    save_lstm,
    train_lstm,
)
from backend.app.ml.preprocessing import (
    FloodDataset,
    check_feature_drift,
    compute_psi,
    create_cv_folds,
    generate_synthetic_flood_data,
    preprocess,
    save_scaler,
)
from backend.app.ml.xgboost_model import (
    XGBTrainResult,
    predict_flood_proba,
    save_model as save_xgb,
    train_xgboost,
    train_xgboost_cv,
    tune_xgboost,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_pipeline")

# Output directory
MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"


def run_pipeline(
    *,
    n_samples: int = 5000,
    do_tune: bool = False,
    tune_trials: int = 30,
    n_cv_folds: int = 5,
    save: bool = True,
    model_dir: Path = MODEL_DIR,
) -> Dict[str, Any]:
    """
    Execute the full training pipeline.

    Parameters
    ----------
    n_samples : int
        Number of synthetic training samples.
    do_tune : bool
        Whether to run Optuna hyperparameter tuning.
    tune_trials : int
        Number of Optuna trials (if do_tune=True).
    n_cv_folds : int
        Cross-validation folds for XGBoost.
    save : bool
        Whether to save models to disk.
    model_dir : Path
        Directory for saved models.

    Returns
    -------
    dict
        Summary metrics from the pipeline.
    """
    t0 = time.time()
    results: Dict[str, Any] = {}

    # ==========================================
    # STEP 1: Generate synthetic data
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 1: Generating synthetic flood data (%d samples)", n_samples)
    logger.info("=" * 60)

    df = generate_synthetic_flood_data(n_samples=n_samples)
    logger.info(
        "  Class distribution: %s",
        dict(df["flood_occurred"].value_counts()),
    )
    results["n_samples"] = n_samples
    results["flood_ratio"] = float(df["flood_occurred"].mean())

    # ==========================================
    # STEP 2: Preprocess
    # ==========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing (split + scale + SMOTE)")
    logger.info("=" * 60)

    dataset = preprocess(df, apply_smote=True)
    logger.info("  %s", dataset.summary())
    results["preprocessing"] = dataset.summary()

    # ==========================================
    # STEP 3: XGBoost training
    # ==========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Training XGBoost")
    logger.info("=" * 60)

    xgb_params = None

    # Optional: Optuna tuning
    if do_tune:
        logger.info("  Running Optuna hyperparameter tuning (%d trials)...", tune_trials)
        xgb_params = tune_xgboost(
            dataset.X_train,
            dataset.y_train,
            dataset.X_val,
            dataset.y_val,
            n_trials=tune_trials,
            timeout=120,
        )
        logger.info("  Best params: %s", xgb_params)

    # Cross-validated training
    logger.info("  Running %d-fold cross-validation...", n_cv_folds)
    xgb_cv_result = train_xgboost_cv(
        np.vstack([dataset.X_train, dataset.X_val]),
        np.concatenate([dataset.y_train, dataset.y_val]),
        feature_names=dataset.feature_names,
        n_folds=n_cv_folds,
        params=xgb_params,
    )
    logger.info(
        "  CV Results — mean AUC=%.4f ± %.4f",
        np.mean(xgb_cv_result.cv_scores),
        np.std(xgb_cv_result.cv_scores),
    )

    # Train final model on train split (eval on val)
    xgb_result = train_xgboost(
        dataset.X_train,
        dataset.y_train,
        dataset.X_val,
        dataset.y_val,
        feature_names=dataset.feature_names,
        params=xgb_params,
    )
    logger.info("  Validation: %s", xgb_result.summary())
    logger.info("\n  Classification Report:\n%s", xgb_result.report)
    logger.info("  Top features: %s",
                dict(list(xgb_result.feature_importance.items())[:5]))

    results["xgboost"] = xgb_result.summary()
    results["xgboost"]["cv_scores"] = xgb_cv_result.cv_scores
    results["xgboost"]["feature_importance"] = dict(
        list(xgb_result.feature_importance.items())[:10]
    )

    # ==========================================
    # STEP 4: LSTM training
    # ==========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Training LSTM (time-series)")
    logger.info("=" * 60)

    X_train_seq, y_train_seq, X_val_seq, y_val_seq = generate_synthetic_sequences(
        n_timesteps=n_samples,
        seed=42,
    )
    logger.info(
        "  Sequence shapes: train=%s  val=%s",
        X_train_seq.shape, X_val_seq.shape,
    )

    lstm_result = train_lstm(
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq,
        epochs=50,
        patience=8,
    )
    logger.info("  LSTM Results: %s", lstm_result.summary())

    results["lstm"] = lstm_result.summary()

    # ==========================================
    # STEP 5: Ensemble optimisation
    # ==========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: Ensemble blending optimisation")
    logger.info("=" * 60)

    # Get XGBoost probabilities on val set
    xgb_val_proba = predict_flood_proba(xgb_result.model, dataset.X_val)

    # Get LSTM probabilities on test portion of sequences
    lstm_val_proba = predict_flood_proba_lstm(lstm_result.model, X_val_seq)

    # Since XGBoost val and LSTM val may have different sizes,
    # we align to the smaller set for alpha optimisation
    min_len = min(len(xgb_val_proba), len(lstm_val_proba))
    xgb_aligned = xgb_val_proba[:min_len]
    lstm_aligned = lstm_val_proba[:min_len]
    y_aligned = dataset.y_val[:min_len]

    best_alpha, best_auc = optimise_alpha(
        xgb_aligned, lstm_aligned, y_aligned, metric="roc_auc"
    )
    logger.info("  Optimal α = %.3f  (AUC = %.4f)", best_alpha, best_auc)

    results["ensemble"] = {
        "optimal_alpha": best_alpha,
        "blend_auc": best_auc,
    }

    # ==========================================
    # STEP 6: Test set evaluation
    # ==========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6: Final test-set evaluation")
    logger.info("=" * 60)

    xgb_test_proba = predict_flood_proba(xgb_result.model, dataset.X_test)

    # For LSTM test, create sequences from test portion
    # (simplified: generate fresh sequences from same distribution)
    _, _, X_test_seq, y_test_seq = generate_synthetic_sequences(
        n_timesteps=len(dataset.y_test) + 24 + 100,
        seed=99,
    )
    X_test_seq = X_test_seq[: len(dataset.y_test)]
    y_test_seq = y_test_seq[: len(dataset.y_test)]
    lstm_test_proba = predict_flood_proba_lstm(lstm_result.model, X_test_seq)

    # Align
    min_test = min(len(xgb_test_proba), len(lstm_test_proba))
    xgb_test_a = xgb_test_proba[:min_test]
    lstm_test_a = lstm_test_proba[:min_test]
    y_test_a = dataset.y_test[:min_test]

    ensemble_proba = blend_predictions(xgb_test_a, lstm_test_a, alpha=best_alpha)
    ensemble_preds = (ensemble_proba >= 0.5).astype(int)

    test_acc = accuracy_score(y_test_a, ensemble_preds)
    test_f1 = f1_score(y_test_a, ensemble_preds, zero_division=0)
    try:
        test_auc = roc_auc_score(y_test_a, ensemble_proba)
    except ValueError:
        test_auc = 0.0

    logger.info("  Test Accuracy:  %.4f", test_acc)
    logger.info("  Test F1:        %.4f", test_f1)
    logger.info("  Test ROC-AUC:   %.4f", test_auc)
    logger.info("\n  Classification Report:\n%s",
                classification_report(y_test_a, ensemble_preds, zero_division=0))

    results["test"] = {
        "accuracy": round(test_acc, 4),
        "f1": round(test_f1, 4),
        "roc_auc": round(test_auc, 4),
    }

    # ==========================================
    # STEP 7: Drift detection baseline
    # ==========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 7: Drift detection baseline (PSI)")
    logger.info("=" * 60)

    # Compare train vs test distributions
    import pandas as pd

    train_df = pd.DataFrame(dataset.X_train, columns=dataset.feature_names)
    test_df = pd.DataFrame(dataset.X_test, columns=dataset.feature_names)
    drift_results = check_feature_drift(train_df, test_df)

    n_drifted = sum(1 for v in drift_results.values() if v["drifted"])
    logger.info(
        "  Features checked: %d | Drifted: %d",
        len(drift_results), n_drifted,
    )
    for feat, info in drift_results.items():
        if info["severity"] != "none":
            logger.info("    %s — PSI=%.4f (%s)", feat, info["psi"], info["severity"])

    results["drift_baseline"] = drift_results

    # ==========================================
    # STEP 8: Save models
    # ==========================================
    if save:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 8: Saving models to %s", model_dir)
        logger.info("=" * 60)

        model_dir.mkdir(parents=True, exist_ok=True)

        save_xgb(xgb_result.model, model_dir / "xgb_flood.joblib")
        save_scaler(dataset.scaler, model_dir / "scaler.joblib")
        save_lstm(
            lstm_result.model,
            model_dir / ("lstm_flood.joblib" if lstm_result.using_fallback else "lstm_flood"),
            is_fallback=lstm_result.using_fallback,
        )

        # Save metrics
        metrics_path = model_dir / "training_metrics.json"
        serialisable = _make_serialisable(results)
        metrics_path.write_text(json.dumps(serialisable, indent=2))
        logger.info("  Metrics saved to %s", metrics_path)

    # ==========================================
    # Summary
    # ==========================================
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — %.1f seconds", elapsed)
    logger.info("=" * 60)
    logger.info("  XGBoost val accuracy:   %.4f", xgb_result.accuracy)
    logger.info("  LSTM val accuracy:      %.4f", lstm_result.accuracy)
    logger.info("  Ensemble test accuracy: %.4f", test_acc)
    logger.info("  Ensemble test AUC:      %.4f", test_auc)
    logger.info("  Target: 90%%+ ✓" if test_acc >= 0.90 else "  Target: 90%%+ ✗ (keep tuning)")

    results["elapsed_seconds"] = round(elapsed, 2)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_serialisable(obj: Any) -> Any:
    """Convert numpy types to Python natives for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Flood Prediction — Training Pipeline")
    print("=" * 60)
    print()

    do_tune = "--tune" in sys.argv
    metrics = run_pipeline(do_tune=do_tune)

    print("\n\nFinal Metrics:")
    print(json.dumps(_make_serialisable(metrics), indent=2, default=str))
