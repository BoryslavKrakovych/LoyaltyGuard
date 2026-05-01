from __future__ import annotations

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


def recommend_channel(row: pd.Series) -> str:
    risk = str(row.get('risk_class', ''))

    if risk == 'High':
        return 'SMS + e-mail'
    if risk == 'Medium':
        return 'Push + e-mail'
    return 'App'


def recommend_action(row: pd.Series) -> str:
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
        category_name = 'Інше'

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


def build_rescue_queue(predicted_df: pd.DataFrame) -> pd.DataFrame:
    work = get_latest_rows(predicted_df)

    if 'is_currently_active' in work.columns:
        work = work[work['is_currently_active'] == True].copy()
    elif 'churn' in work.columns:
        work = work[work['churn'] == 0].copy()

    work = add_recommendations(work)

    cols = [
        'customer_id',
        'churn_probability_percent',
        'risk_class',
        'rfm_segment',
        'customer_cluster_name',
        'dominant_category_display',
        'top_categories_display',
        'recommended_action',
        'recommended_channel',
    ]
    cols = [col for col in cols if col in work.columns]

    return work[cols].sort_values(by='churn_probability_percent', ascending=False).reset_index(drop=True)


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
    base_probability = float(customer_row.get('churn_probability', predict_probability_from_feature_row(customer_x_row, artifacts)))

    scenario_x_row = apply_scenario_to_feature_row(
        customer_x_row,
        freq_change_pct,
        ticket_change_pct,
        promo_change_pct,
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
    if audience_df is None or len(audience_df) == 0:
        empty_summary = pd.DataFrame(columns=[
            'audience_size',
            'avg_risk_before_pct',
            'avg_risk_after_pct',
            'avg_reduction_pp',
            'customers_improved',
            'high_risk_before',
            'high_risk_after',
        ])
        empty_details = pd.DataFrame(columns=[
            'customer_id',
            'risk_before',
            'risk_after',
            'probability_before_pct',
            'probability_after_pct',
            'reduction_pp',
        ])
        return empty_summary, empty_details

    work_audience = audience_df.reset_index(drop=True).copy()
    work_X = audience_X.reset_index(drop=True).copy()

    detail_rows = []

    for pos in range(len(work_audience)):
        row = work_audience.iloc[pos]
        x_row = work_X.iloc[pos]

        base_probability = float(
            row.get('churn_probability', predict_probability_from_feature_row(x_row, artifacts))
        )

        scenario_x_row = apply_scenario_to_feature_row(
            x_row,
            freq_change_pct,
            ticket_change_pct,
            promo_change_pct,
        )
        new_probability = predict_probability_from_feature_row(scenario_x_row, artifacts)

        detail_rows.append({
            'customer_id': row['customer_id'],
            'risk_before': probability_to_risk(base_probability),
            'risk_after': probability_to_risk(new_probability),
            'probability_before_pct': round(base_probability * 100, 1),
            'probability_after_pct': round(new_probability * 100, 1),
            'reduction_pp': round((base_probability - new_probability) * 100, 1),
        })

    details = pd.DataFrame(detail_rows)
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
