from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit

from src.config import LOW_RISK_MAX, MEDIUM_RISK_MAX, RANDOM_STATE
from src.feature_engineering import align_columns_for_inference

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


@dataclass
class ChurnArtifacts:
    model: object
    X_train_columns: list[str]
    train_index: np.ndarray
    test_index: np.ndarray
    roc_auc: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    algorithm_name: str
    accuracy: float = float('nan')
    precision: float = float('nan')
    recall: float = float('nan')
    f1: float = float('nan')
    logloss: float = float('nan')
    confusion_matrix_table: Optional[pd.DataFrame] = None


class ChurnModelService:
    def __init__(self, use_xgboost: bool = True, random_state: int = RANDOM_STATE) -> None:
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        self.random_state = random_state

    def _build_model(self):
        if self.use_xgboost:
            return XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                eval_metric='logloss',
            ), 'XGBoost'

        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.random_state,
        ), 'GradientBoosting'

    def train(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, test_size: float = 0.15) -> ChurnArtifacts:
        if y.nunique() < 2:
            raise ValueError('The churn target must contain at least two classes for training.')

        train_idx = test_idx = None
        for split_offset in range(10):
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=self.random_state + split_offset,
            )
            candidate_train_idx, candidate_test_idx = next(gss.split(X, y, groups))
            y_train_candidate = y.iloc[candidate_train_idx]
            y_test_candidate = y.iloc[candidate_test_idx]

            if y_train_candidate.nunique() >= 2 and y_test_candidate.nunique() >= 2:
                train_idx, test_idx = candidate_train_idx, candidate_test_idx
                break

        if train_idx is None or test_idx is None:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)
            train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model, algorithm_name = self._build_model()
        model.fit(X_train, y_train)

        proba_test = model.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= 0.5).astype(int)

        if y_test.nunique() >= 2:
            fpr, tpr, thresholds = roc_curve(y_test, proba_test)
            roc_auc = auc(fpr, tpr)
            model_logloss = log_loss(y_test, proba_test, labels=[0, 1])
        else:
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            thresholds = np.array([np.inf, 0.0])
            roc_auc = float('nan')
            model_logloss = float('nan')

        model_accuracy = accuracy_score(y_test, pred_test)
        model_precision = precision_score(y_test, pred_test, zero_division=0)
        model_recall = recall_score(y_test, pred_test, zero_division=0)
        model_f1 = f1_score(y_test, pred_test, zero_division=0)

        confusion_table = pd.DataFrame(
            confusion_matrix(y_test, pred_test, labels=[0, 1]),
            index=['Actual 0', 'Actual 1'],
            columns=['Predicted 0', 'Predicted 1'],
        )

        return ChurnArtifacts(
            model=model,
            X_train_columns=list(X_train.columns),
            train_index=train_idx,
            test_index=test_idx,
            roc_auc=float(roc_auc),
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            algorithm_name=algorithm_name,
            accuracy=float(model_accuracy),
            precision=float(model_precision),
            recall=float(model_recall),
            f1=float(model_f1),
            logloss=float(model_logloss),
            confusion_matrix_table=confusion_table,
        )

    def predict_proba(self, artifacts: ChurnArtifacts, X_new: pd.DataFrame) -> np.ndarray:
        X_aligned = align_columns_for_inference(
            pd.DataFrame(columns=artifacts.X_train_columns),
            X_new,
        )
        return artifacts.model.predict_proba(X_aligned)[:, 1]


def probability_to_risk(probability: float) -> str:
    if probability <= LOW_RISK_MAX:
        return 'Low'
    if probability <= MEDIUM_RISK_MAX:
        return 'Medium'
    return 'High'


def attach_predictions(df_original: pd.DataFrame, X_full: pd.DataFrame, artifacts: ChurnArtifacts) -> pd.DataFrame:
    df_result = df_original.copy()
    X_aligned = align_columns_for_inference(
        pd.DataFrame(columns=artifacts.X_train_columns),
        X_full,
    )
    probabilities = artifacts.model.predict_proba(X_aligned)[:, 1]
    df_result['churn_probability'] = probabilities
    df_result['churn_probability_percent'] = np.round(probabilities * 100, 1)
    df_result['risk_class'] = df_result['churn_probability'].apply(probability_to_risk)
    return df_result


def feature_importance_table(artifacts: ChurnArtifacts) -> pd.DataFrame:
    model = artifacts.model
    columns = artifacts.X_train_columns

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        table = pd.DataFrame({
            'feature': columns,
            'importance': importances,
        }).sort_values('importance', ascending=False)
        return table

    return pd.DataFrame(columns=['feature', 'importance'])
