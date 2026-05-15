"""
Retention recommendations.

Two recommendation engines:
  1. Heuristic (legacy): recommend_action / recommend_channel based on risk_class
     and dominant_category. Used when no feedback model is available.
  2. Feedback-aware: recommend_action_with_model / recommend_channel_with_model.
     Uses FeedbackArtifacts to pick the BEST category per customer and a
     dynamically-scaled discount %.

The feedback-aware path requires `best_category` and `best_response_prob`
columns to already be on the row (produced by
src.feedback_model.attach_feedback_recommendations).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.config import LOW_RISK_MAX, MEDIUM_RISK_MAX
from src.feature_engineering import align_columns_for_inference


BAD_CATEGORY_VALUES = {
    '',
    'unknown',
    'general assortment',
    'general merchandise',
    'загальний асортимент',
    'невизначена категорія',
    'інший кластер',
    'інше',
    'other',
    'n/a',
    'none',
    'null',
    'nan',
    '<na>',
}


# ── Risk class ────────────────────────────────────────────────────────────

def probability_to_risk(probability: float) -> str:
    if probability <= LOW_RISK_MAX:
        return 'Low'
    if probability <= MEDIUM_RISK_MAX:
        return 'Medium'
    return 'High'


def get_latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if 'transaction_date' in work.columns:
        work = work.sort_values(by=['customer_id', 'transaction_date'])
    return work.groupby('customer_id').tail(1).copy()


def normalize_category_name(value) -> str:
    if pd.isna(value):
        return ''
    text = str(value).strip()
    if not text:
        return ''
    if '—' in text:
        text = text.split('—', 1)[0].strip()
    if ' - ' in text:
        text = text.split(' - ', 1)[0].strip()
    if text.lower() in BAD_CATEGORY_VALUES:
        return ''
    return text


def first_valid_category(value) -> str:
    if pd.isna(value):
        return ''
    for item in str(value).split(','):
        cleaned = normalize_category_name(item)
        if cleaned:
            return cleaned
    return ''


# ── Heuristic (legacy) path ───────────────────────────────────────────────

def recommend_channel(row: pd.Series) -> str:
    risk = str(row.get('risk_class', ''))
    if risk == 'High':
        return 'SMS + e-mail'
    if risk == 'Medium':
        return 'Push + e-mail'
    return 'App'


def recommend_action(row: pd.Series) -> str:
    """Heuristic action — fixed discount style, only varies by risk class."""
    risk = str(row.get('risk_class', ''))
    category_name = normalize_category_name(row.get('campaign_category', ''))
    if not category_name:
        category_name = normalize_category_name(row.get('dominant_category_display', ''))
    if not category_name:
        category_name = normalize_category_name(row.get('dominant_category', ''))
    if not category_name:
        category_name = first_valid_category(row.get('top_categories_display', ''))
    if not category_name:
        category_name = first_valid_category(row.get('top_categories', ''))
    if not category_name:
        category_name = 'General Merchandise'

    if risk == 'High':
        return f'Точкова знижка + персональна рекомендація по категорії {category_name}'
    if risk == 'Medium':
        return f'Нагадування + добірка товарів по категорії {category_name}'
    return f'Контентна комунікація по категорії {category_name}'


def add_recommendations(queue_df: pd.DataFrame) -> pd.DataFrame:
    queue_df = queue_df.copy()
    queue_df['recommended_action'] = queue_df.apply(recommend_action, axis=1)
    queue_df['recommended_channel'] = queue_df.apply(recommend_channel, axis=1)
    return queue_df


# ── Feedback-aware path ───────────────────────────────────────────────────

def recommend_discount_pct(response_prob: float) -> int:
    """Discount % based on predicted response probability.

    - P >= 0.50 → 10% (small nudge enough)
    - P >= 0.20 → 15% (standard)
    - P <  0.20 → 25% (need to motivate hard)
    """
    if response_prob >= 0.50:
        return 10
    if response_prob >= 0.20:
        return 15
    return 25


def recommend_action_with_model(row: pd.Series) -> str:
    """Build action text using feedback-model output.

    Expects: risk_class, best_category, best_response_prob.
    """
    risk = str(row.get('risk_class', ''))
    category = normalize_category_name(row.get('best_category', '')) or 'General Merchandise'
    prob = float(row.get('best_response_prob', 0.0))
    discount = recommend_discount_pct(prob)
    response_pct = round(prob * 100, 1)

    if risk == 'High':
        return (f'{discount}% знижки на наступну покупку в категорії {category} '
                f'(P(response)={response_pct}%)')
    if risk == 'Medium':
        return (f'Підбірка з категорії {category} + {discount}% при покупці протягом тижня '
                f'(P(response)={response_pct}%)')
    return f'Персональна добірка товарів з категорії {category} (P(response)={response_pct}%)'


def recommend_channel_with_model(row: pd.Series) -> str:
    """Channel choice — same as legacy unless we have strong negative signal."""
    risk = str(row.get('risk_class', ''))
    prob = float(row.get('best_response_prob', 0.0))

    # If even the BEST category has very low predicted response → soft channel
    if prob < 0.10:
        return 'App'

    if risk == 'High':
        return 'SMS + e-mail'
    if risk == 'Medium':
        return 'Push + e-mail'
    return 'App'


def add_recommendations_with_model(queue_df: pd.DataFrame) -> pd.DataFrame:
    """Drop-in replacement for add_recommendations when feedback model is loaded."""
    queue_df = queue_df.copy()
    queue_df['recommended_action'] = queue_df.apply(recommend_action_with_model, axis=1)
    queue_df['recommended_channel'] = queue_df.apply(recommend_channel_with_model, axis=1)
    if 'best_response_prob' in queue_df.columns:
        queue_df['discount_pct'] = queue_df['best_response_prob'].apply(recommend_discount_pct)
    return queue_df


# ── Rescue queue ──────────────────────────────────────────────────────────

def build_rescue_queue(
    predicted_df: pd.DataFrame,
    feedback_artifacts: Optional[object] = None,
) -> pd.DataFrame:
    """Build the rescue queue.

    If feedback_artifacts is provided, recommendations are computed using
    the response model. Otherwise the legacy heuristic is used.
    """
    work = get_latest_rows(predicted_df)

    if 'is_currently_active' in work.columns:
        work = work[work['is_currently_active'] == True].copy()
    elif 'churn' in work.columns:
        work = work[work['churn'] == 0].copy()

    if feedback_artifacts is not None:
        from src.feedback_model import attach_feedback_recommendations
        work = attach_feedback_recommendations(work, feedback_artifacts)
        work = add_recommendations_with_model(work)
    else:
        work = add_recommendations(work)

    cols = [
        'customer_id',
        'churn_probability_percent',
        'risk_class',
        'rfm_segment',
        'customer_cluster_name',
        'best_category',
        'top_categories_ranked',
        'discount_pct',
        'response_prob_pct',
        'dominant_category_display',
        'top_categories_display',
        'recommended_action',
        'recommended_channel',
    ]
    cols = [col for col in cols if col in work.columns]
    return work[cols].sort_values(by='churn_probability_percent', ascending=False).reset_index(drop=True)


# ── What-if simulator (unchanged contract, kept for backward compat) ──────

def apply_scenario_to_feature_row(
    customer_x_row: pd.Series,
    freq_change_pct: float,
    ticket_change_pct: float,
    promo_change_pct: float,
) -> pd.Series:
    updated = customer_x_row.copy()

    freq_change_pct = min(max(float(freq_change_pct), 0.0), 0.15)
    ticket_change_pct = min(max(float(ticket_change_pct), -0.50), 0.50)
    promo_change_pct = min(max(float(promo_change_pct), 0.0), 0.10)

    if 'days_since_prev_purchase' in updated.index:
        updated['days_since_prev_purchase'] = max(
            float(updated['days_since_prev_purchase']) * (1.0 - freq_change_pct),
            0.0,
        )
    if 'customer_mean_gap' in updated.index:
        updated['customer_mean_gap'] = max(
            float(updated['customer_mean_gap']) * (1.0 - freq_change_pct),
            0.0,
        )
    if 'avg_ticket_size' in updated.index:
        updated['avg_ticket_size'] = float(updated['avg_ticket_size']) * (1.0 + ticket_change_pct)
    if 'promo_response_rate' in updated.index:
        updated['promo_response_rate'] = min(
            float(updated['promo_response_rate']) + promo_change_pct,
            1.0,
        )
    return updated


def predict_probability_from_feature_row(customer_x_row: pd.Series, artifacts) -> float:
    X_new = pd.DataFrame([customer_x_row])
    X_aligned = align_columns_for_inference(pd.DataFrame(columns=artifacts.X_train_columns), X_new)
    probability = artifacts.model.predict_proba(X_aligned)[:, 1][0]
    return float(probability)


def simulate_what_if(
    customer_row: pd.Series,
    customer_x_row: pd.Series,
    artifacts,
    freq_change_pct: float,
    ticket_change_pct: float,
    promo_change_pct: float,
) -> pd.DataFrame:
    base_probability = float(
        customer_row.get('churn_probability',
                         predict_probability_from_feature_row(customer_x_row, artifacts))
    )

    scenario_x_row = apply_scenario_to_feature_row(
        customer_x_row, freq_change_pct, ticket_change_pct, promo_change_pct,
    )
    new_probability = predict_probability_from_feature_row(scenario_x_row, artifacts)

    result = pd.DataFrame({
        'scenario': ['Current', 'After what-if'],
        'churn_probability_percent': [round(base_probability * 100, 1), round(new_probability * 100, 1)],
        'risk_class': [probability_to_risk(base_probability), probability_to_risk(new_probability)],
    })
    result['change_pp'] = result['churn_probability_percent'] - result['churn_probability_percent'].iloc[0]
    return result


def estimate_campaign_effect(
    audience_df: pd.DataFrame,
    audience_X: pd.DataFrame,
    artifacts,
    freq_change_pct: float,
    ticket_change_pct: float,
    promo_change_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fast what-if estimation without rerunning ML predictions.

    The churn model is used once when customers are prepared. UI sliders then
    apply a transparent business approximation to the already computed churn
    probability, so moving sliders or downloading a campaign does not trigger
    model inference for every customer again.
    """
    if audience_df is None or len(audience_df) == 0:
        empty_summary = pd.DataFrame(columns=[
            'audience_size', 'avg_risk_before_pct', 'avg_risk_after_pct',
            'avg_reduction_pp', 'customers_improved',
            'high_risk_before', 'high_risk_after',
        ])
        empty_details = pd.DataFrame(columns=[
            'customer_id', 'risk_before', 'risk_after',
            'probability_before_pct', 'probability_after_pct', 'reduction_pp',
        ])
        return empty_summary, empty_details

    work = audience_df.reset_index(drop=True).copy()

    base_prob = pd.to_numeric(
        work.get('churn_probability', work.get('churn_probability_percent', 0) / 100),
        errors='coerce',
    ).fillna(0).clip(0, 1)

    # Positive business changes should reduce churn risk.
    freq_change_pct = min(max(float(freq_change_pct), 0.0), 0.15)
    ticket_change_pct = min(max(float(ticket_change_pct), -0.50), 0.50)
    promo_change_pct = min(max(float(promo_change_pct), 0.0), 0.10)

    reduction = (
        0.20 * freq_change_pct
        + 0.25 * max(ticket_change_pct, 0.0)
        + 0.10 * promo_change_pct
    )
    increase = 0.18 * max(-ticket_change_pct, 0.0)

    new_prob = (base_prob * (1.0 - reduction + increase)).clip(0, 1)

    details = pd.DataFrame({
        'customer_id': work['customer_id'].astype(str),
        'risk_before': base_prob.apply(probability_to_risk),
        'risk_after': new_prob.apply(probability_to_risk),
        'probability_before_pct': (base_prob * 100).round(1),
        'probability_after_pct': (new_prob * 100).round(1),
    })
    details['reduction_pp'] = (
        details['probability_before_pct'] - details['probability_after_pct']
    ).round(1)
    details = details.sort_values(by='reduction_pp', ascending=False).reset_index(drop=True)

    summary = pd.DataFrame([{
        'audience_size': int(len(details)),
        'avg_risk_before_pct': round(details['probability_before_pct'].mean(), 2),
        'avg_risk_after_pct': round(details['probability_after_pct'].mean(), 2),
        'avg_reduction_pp': round(details['reduction_pp'].mean(), 2),
        'customers_improved': int((details['reduction_pp'] > 0).sum()),
        'high_risk_before': int((details['risk_before'] == 'High').sum()),
        'high_risk_after': int((details['risk_after'] == 'High').sum()),
    }])

    return summary, details
