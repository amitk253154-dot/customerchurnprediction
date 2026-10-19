import os
import sys
from dataclasses import dataclass

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.customer_chrun.exception import CustomException
from src.customer_chrun.logger import logging
from src.customer_chrun.utils import save_object

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    beta: float = 2.0          # F-beta: beta > 1 favours recall (catching churners)
    n_trials: int = 50         # Optuna trials
    cv_splits: int = 5
    random_state: int = 42


def pick_threshold(y_true, proba, beta):
    """Threshold that maximizes F-beta on the given out-of-fold predictions."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]
    fb = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-12)
    i = int(np.nanargmax(fb))
    return float(thr[i]), float(fb[i])


def report_at_threshold(y_true, proba, thresh):
    pred = (proba >= thresh).astype(int)
    return {
        "threshold": round(float(thresh), 3),
        "recall": round(recall_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "flagged_pct": round(float(pred.mean() * 100), 1),
    }


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def get_baseline_models(self, spw):
        rs = self.model_trainer_config.random_state
        return {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, class_weight="balanced", min_samples_leaf=5, n_jobs=-1, random_state=rs),
            "LightGBM": LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=30,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                class_weight="balanced", random_state=rs, n_jobs=-1, verbose=-1),
            "XGBoost": XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=3, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
                eval_metric="aucpr", tree_method="hist", random_state=rs, n_jobs=-1),
        }

    def tune_xgboost(self, X_train, y_train, cv, spw, preprocessor):
        """Optuna search on cross-validated PR-AUC, training data only."""
        rs = self.model_trainer_config.random_state

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10, log=True),
            }
            clf = XGBClassifier(**params, scale_pos_weight=spw, tree_method="hist",
                                random_state=rs, n_jobs=-1)
            model = Pipeline([("prep", clone(preprocessor)), ("clf", clf)])
            return cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision").mean()

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=rs))
        study.optimize(objective, n_trials=self.model_trainer_config.n_trials)
        return study

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test, preprocessor):
        cfg = self.model_trainer_config
        try:
            spw = float((y_train == 0).sum() / (y_train == 1).sum())  # y_train is a 0/1 Series
            cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)

            # 1. Baseline models: cross-validated PR-AUC on the TRAINING set
            baseline_report = {}
            for name, model in self.get_baseline_models(spw).items():
                pipe = Pipeline([("prep", clone(preprocessor)), ("clf", model)])
                scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision")
                baseline_report[name] = round(float(scores.mean()), 4)
                logging.info(f"{name}: CV PR-AUC {scores.mean():.4f} +/- {scores.std():.4f}")

            # 2. Tune XGBoost with Optuna (training data only)
            logging.info("Optuna tuning started")
            study = self.tune_xgboost(X_train, y_train, cv, spw, preprocessor)
            logging.info(f"Best CV PR-AUC {study.best_value:.4f} with params {study.best_params}")

            best_clf = XGBClassifier(**study.best_params, scale_pos_weight=spw, tree_method="hist",
                                     random_state=cfg.random_state, n_jobs=-1)
            best_model = Pipeline([("prep", clone(preprocessor)), ("clf", best_clf)])

            # 3. Threshold from out-of-fold predictions on the TRAINING set
            oof = cross_val_predict(best_model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
            best_t, best_fb = pick_threshold(y_train, oof, cfg.beta)
            logging.info(f"Tuned threshold {best_t:.3f} (F{cfg.beta:g} = {best_fb:.3f})")

            # 4. Fit on the full training set, evaluate once on the test set
            best_model.fit(X_train, y_train)
            test_proba = best_model.predict_proba(X_test)[:, 1]
            metrics = {
                "baseline_cv_pr_auc": baseline_report,
                "tuned_cv_pr_auc": round(float(study.best_value), 4),
                "test_pr_auc": round(average_precision_score(y_test, test_proba), 4),
                "test_roc_auc": round(roc_auc_score(y_test, test_proba), 4),
                "at_default_0.5": report_at_threshold(y_test, test_proba, 0.5),
                "at_tuned_threshold": report_at_threshold(y_test, test_proba, best_t),
            }
            logging.info(f"Final test metrics: {metrics}")

            # 5. Save the full pipeline (preprocessor + model) together with its threshold
            save_object(cfg.trained_model_file_path, {"model": best_model, "threshold": best_t})
            logging.info("Saved the trained model and threshold")
            return metrics

        except Exception as e:
            logging.error("Exception occurred in model training")
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.customer_chrun.components.data_ingestion import DataIngestion
    from src.customer_chrun.components.data_transformation import DataTransformation

    train_path, test_path = DataIngestion().initiate_data_ingestion()
    X_train, y_train, X_test, y_test, preprocessor = DataTransformation().initiate_data_transformation(train_path, test_path)
    results = ModelTrainer().initiate_model_trainer(X_train, y_train, X_test, y_test, preprocessor)
    for key, value in results.items():
        print(f"{key}: {value}")