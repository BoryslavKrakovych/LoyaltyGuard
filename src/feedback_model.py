"""
Feedback response model — predicts P(positive reaction | customer features, category).

Trained on a feedback CSV with columns: customer_id, suggested_item, reaction.
Used by retention recommendations to:
  1. Pick the BEST category per customer (not the static dominant_category).
  2. Scale discount % by predicted response probability (vector of offers, not single value).
  3. Show ROC-AUC of the response model alongside the churn model.

Reaction mapping (case-insensitive):
  POSITIVE: 'purchased', 'converted', 'clicked', 'positive', 'interested'
  NEGATIVE: 'not interested', 'unsubscribed', 'rejected', 'no action', 'ignored'
  Unknown reactions are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

from src.config import RANDOM_STATE


POSITIVE_REACTIONS = {'purchased', 'converted', 'clicked', 'positive', 'interested'}
NEGATIVE_REACTIONS = {'not interested', 'unsubscribed', 'rejected', 'no action', 'ignored'}

_CID_ALIASES = {
    'customer_id', 'customerid', 'customer id',
    'user_id', 'userid', 'user id',
    'client_id', 'clientid', 'client id',
    'cust_id', 'custid',
}
_ITEM_ALIASES = {
    'suggested_item', 'suggested item', 'item', 'item_name',
    'product', 'product_name', 'товар',
}
_REACTION_ALIASES = {
    'reaction', 'status', 'response', 'feedback', 'result', 'action',
}


@dataclass
class FeedbackArtifacts:
    model: object
    feature_columns: list[str]
    category_columns: list[str]
    known_categories: list[str]
    roc_auc: float
    n_train: int
    n_test: int
    n_positive: int
    n_negative: int
    algorithm_name: str


def _binarize_reaction(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in POSITIVE_REACTIONS:
        return 1
    if text in NEGATIVE_REACTIONS:
        return 0
    return None


def _find_column(columns: list[str], aliases: set[str]) -> Optional[str]:
    for col in columns:
        if str(col).strip().lower() in aliases:
            return col
    return None


def autodetect_feedback_columns(feedback_df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Auto-detect customer_id / item / reaction columns by common aliases."""
    cols = list(feedback_df.columns)
    return {
        'customer_id': _find_column(cols, _CID_ALIASES),
        'item': _find_column(cols, _ITEM_ALIASES),
        'reaction': _find_column(cols, _REACTION_ALIASES),
    }


def prepare_feedback_training_set(
    feedback_df: pd.DataFrame,
    customer_features: pd.DataFrame,
    categorizer: Callable[[str], str],
    cid_col: str,
    item_col: str,
    reaction_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    """Build (X, y, groups, feature_cols, category_cols) for the response model.

    customer_features must contain a 'customer_id' column and numeric feature
    columns (typically the latest_customers DataFrame).
    """
    fb = feedback_df.copy()
    fb[cid_col] = fb[cid_col].astype(str).str.strip()

    cf = customer_features.copy()
    cf['customer_id'] = cf['customer_id'].astype(str).str.strip()

    fb['_y'] = fb[reaction_col].apply(_binarize_reaction)
    fb = fb.dropna(subset=['_y']).copy()
    if fb.empty:
        raise ValueError('No usable reactions in feedback (need at least one positive and one negative).')
    fb['_y'] = fb['_y'].astype(int)

    fb['_category'] = fb[item_col].fillna('').astype(str).apply(categorizer)

    merged = fb.merge(cf, left_on=cid_col, right_on='customer_id', how='inner')
    if merged.empty:
        raise ValueError('No overlap between feedback customer IDs and modelled customers.')

    category_dummies = pd.get_dummies(merged['_category'], prefix='cat').astype(int)
    if category_dummies.shape[1] == 0:
        raise ValueError('Could not build category one-hot encoding from feedback.')

    excluded = {
        'customer_id', 'churn_probability', 'churn_probability_percent',
        'feature_row_index', 'customer_cluster', '_y',
    }
    numeric_cols = [
        c for c in cf.select_dtypes(include=[np.number]).columns
        if c not in excluded
    ]
    if not numeric_cols:
        raise ValueError('customer_features has no numeric columns to use as features.')

    X = pd.concat(
        [merged[numeric_cols].reset_index(drop=True),
         category_dummies.reset_index(drop=True)],
        axis=1,
    )
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = merged['_y'].reset_index(drop=True)
    groups = merged[cid_col].reset_index(drop=True)
    return X, y, groups, numeric_cols, list(category_dummies.columns)


class FeedbackResponseService:
    def __init__(self, use_xgboost: bool = True, random_state: int = RANDOM_STATE):
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        self.random_state = random_state

    def _build_model(self):
        if self.use_xgboost:
            return XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                random_state=self.random_state, eval_metric='logloss',
            ), 'XGBoost'
        return GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            random_state=self.random_state,
        ), 'GradientBoosting'

    def train(
        self,
        X: pd.DataFrame, y: pd.Series, groups: pd.Series,
        feature_cols: list[str], category_cols: list[str],
        test_size: float = 0.20,
    ) -> FeedbackArtifacts:
        if y.nunique() < 2:
            raise ValueError('Feedback target must have both positive (1) and negative (0).')

        if groups.nunique() >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)
            train_idx, test_idx = next(gss.split(X, y, groups))
        else:
            n = len(X)
            split = int(n * (1 - test_size))
            train_idx, test_idx = np.arange(split), np.arange(split, n)

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        model, algorithm_name = self._build_model()
        model.fit(X_tr, y_tr)

        if y_te.nunique() >= 2:
            probas = model.predict_proba(X_te)[:, 1]
            roc = float(roc_auc_score(y_te, probas))
        else:
            roc = float('nan')

        known_categories = [c.replace('cat_', '', 1) for c in category_cols]

        return FeedbackArtifacts(
            model=model,
            feature_columns=feature_cols,
            category_columns=category_cols,
            known_categories=known_categories,
            roc_auc=roc,
            n_train=int(len(X_tr)),
            n_test=int(len(X_te)),
            n_positive=int((y == 1).sum()),
            n_negative=int((y == 0).sum()),
            algorithm_name=algorithm_name,
        )


def predict_response_matrix(
    customer_features: pd.DataFrame,
    artifacts: FeedbackArtifacts,
) -> pd.DataFrame:
    """For each (customer × candidate category) return P(positive reaction).

    Returns DataFrame indexed by customer_id, columns = known categories.
    """
    work = customer_features.copy()
    work['customer_id'] = work['customer_id'].astype(str).str.strip()

    feature_block = work[artifacts.feature_columns].copy()
    feature_block = feature_block.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    result = {}
    for cat_col, cat_name in zip(artifacts.category_columns, artifacts.known_categories):
        X_full = feature_block.copy()
        for c in artifacts.category_columns:
            X_full[c] = 1 if c == cat_col else 0
        X_full = X_full[artifacts.feature_columns + artifacts.category_columns]
        result[cat_name] = artifacts.model.predict_proba(X_full)[:, 1]

    matrix = pd.DataFrame(result, index=work['customer_id'].values)
    matrix.index.name = 'customer_id'
    return matrix


def best_category_per_customer(matrix: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Rank categories per customer by predicted P(positive)."""
    rows = []
    for cust_id, scores in matrix.iterrows():
        ranked = scores.sort_values(ascending=False)
        rows.append({
            'customer_id': cust_id,
            'best_category': ranked.index[0],
            'best_response_prob': float(ranked.iloc[0]),
            'top_categories_ranked': ', '.join(ranked.index[:top_k].tolist()),
            'response_prob_top3_avg': float(ranked.iloc[:top_k].mean()),
        })
    return pd.DataFrame(rows)


def recommend_discount_pct(response_prob: float) -> int:
    """Map predicted response probability → discount %.

    - P >= 0.50  → 10% (small nudge enough)
    - P >= 0.20  → 15%  (standard)
    - P <  0.20  → 25% (need to motivate hard)
    """
    if response_prob >= 0.50:
        return 10
    if response_prob >= 0.20:
        return 15
    return 25


def build_offer_vector(row: pd.Series) -> dict:
    """Translate (risk_class, best_category, best_response_prob) into a concrete offer."""
    risk = str(row.get('risk_class', ''))
    category = str(row.get('best_category', 'General Merchandise'))
    prob = float(row.get('best_response_prob', 0.0))
    discount = recommend_discount_pct(prob)

    if risk == 'High':
        offer_type = 'Прицільна знижка'
        channel = 'SMS + e-mail'
        message = f'{discount}% знижки на наступну покупку в категорії {category}'
    elif risk == 'Medium':
        offer_type = 'Нагадування + знижка'
        channel = 'Push + e-mail'
        message = f'Підбірка з категорії {category} + {discount}% при покупці протягом тижня'
    else:
        offer_type = 'Контентна добірка'
        channel = 'App'
        message = f'Персональна добірка товарів з категорії {category}'

    return {
        'offer_type': offer_type,
        'discount_pct': discount,
        'recommended_message': message,
        'recommended_channel': channel,
        'response_prob_pct': round(prob * 100, 1),
    }


def attach_feedback_recommendations(
    audience: pd.DataFrame,
    artifacts: FeedbackArtifacts,
) -> pd.DataFrame:
    """Add feedback-aware columns to an audience DataFrame.

    Adds:
      best_category, best_response_prob, top_categories_ranked,
      offer_type, discount_pct, recommended_message,
      recommended_channel, response_prob_pct.
    """
    if audience is None or audience.empty:
        return audience.copy() if audience is not None else pd.DataFrame()

    matrix = predict_response_matrix(audience, artifacts)
    best = best_category_per_customer(matrix)

    work = audience.copy()
    work['customer_id'] = work['customer_id'].astype(str).str.strip()
    work = work.merge(best, on='customer_id', how='left')

    work['best_response_prob'] = work['best_response_prob'].fillna(0.0)
    work['best_category'] = work['best_category'].fillna('General Merchandise')
    work['top_categories_ranked'] = work['top_categories_ranked'].fillna('')

    offers = work.apply(build_offer_vector, axis=1)
    offers_df = pd.DataFrame(list(offers))
    for col in offers_df.columns:
        work[col] = offers_df[col].values

    return work
