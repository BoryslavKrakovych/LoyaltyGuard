from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.config import LOW_RISK_MAX, MEDIUM_RISK_MAX
from src.feature_engineering import align_columns_for_inference


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


def first_category(value: str) -> str:
    if pd.isna(value):
        return 'unknown'

    items = [x.strip() for x in str(value).split(',') if x.strip()]
    if not items:
        return 'unknown'
    return items[0]


def choose_category_with_feedback(top_categories: str, dominant_category: str, blocked_categories: list[str]) -> str:
    candidates = []

    if pd.notna(dominant_category) and str(dominant_category).strip():
        candidates.append(str(dominant_category).strip())

    if pd.notna(top_categories):
        candidates.extend([x.strip() for x in str(top_categories).split(',') if x.strip()])

    blocked_set = {str(x).strip().lower() for x in blocked_categories if str(x).strip()}

    for category in candidates:
        if category.lower() not in blocked_set:
            return category

    if candidates:
        return candidates[0]

    return 'general assortment'


def recommend_channel(row: pd.Series) -> str:
    risk = str(row.get('risk_class', ''))
    cooling = bool(row.get('cooling_flag', False))

    if risk == 'High' and cooling:
        return 'SMS + push'
    if risk == 'High':
        return 'SMS + e-mail'
    if risk == 'Medium':
        return 'push + e-mail'
    return 'app / e-mail'


def recommend_action(row: pd.Series) -> str:
    risk = str(row.get('risk_class', ''))
    segment = str(row.get('rfm_segment', ''))
    cooling = bool(row.get('cooling_flag', False))
    dominant_category = row.get('dominant_category', '')
    category_name = dominant_category if pd.notna(dominant_category) and str(dominant_category).strip() else first_category(row.get('top_categories', ''))

    if risk == 'High' and 'At Risk' in segment and cooling:
        return f'Сильна retention-пропозиція + персональний контакт по категорії {category_name}'
    if risk == 'High':
        return f'Точкова знижка + нагадування по категорії {category_name}'
    if risk == 'Medium' and cooling:
        return f'М’яка акція + персональна рекомендація по категорії {category_name}'
    if risk == 'Medium':
        return 'М’яке нагадування + добірка релевантних товарів'
    return 'Лояльність / контентне нагадування'


def add_recommendations(queue_df: pd.DataFrame) -> pd.DataFrame:
    queue_df = queue_df.copy()
    queue_df['recommended_action'] = queue_df.apply(recommend_action, axis=1)
    queue_df['recommended_channel'] = queue_df.apply(recommend_channel, axis=1)
    return queue_df


def build_rescue_queue(predicted_df: pd.DataFrame) -> pd.DataFrame:
    work = get_latest_rows(predicted_df)
    if 'is_currently_active' in work.columns:
        active_customers = work[work['is_currently_active'] == True].copy()
    else:
        active_customers = work[work['churn'] == 0].copy()

    cols = [
        'customer_id',
        'churn_probability_percent',
        'risk_class',
        'rfm_segment',
        'customer_cluster_name',
        'top_categories',
        'dominant_category',
        'avg_ticket_size',
        'basket_items',
        'days_since_prev_purchase',
        'cooling_flag',
        'cooling_score',
        'frequency_drop_flag',
        'basket_drop_flag',
        'items_drop_flag',
        'recommended_action',
        'recommended_channel',
    ]
    cols = [col for col in cols if col in active_customers.columns]

    result = active_customers[cols].sort_values(by='churn_probability_percent', ascending=False)
    return result.reset_index(drop=True)


def compute_cooling_triggers(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if 'customer_id' not in work.columns:
        return pd.DataFrame(columns=['customer_id', 'cooling_flag'])

    if 'transaction_date' in work.columns:
        work = work.sort_values(['customer_id', 'transaction_date']).copy()

    if 'total_sales' in work.columns:
        work['prev_total_sales'] = work.groupby('customer_id')['total_sales'].shift(1)
        work['basket_drop_flag'] = (
            work['total_sales'] < work['prev_total_sales'] * 0.7
        ).fillna(False)
    else:
        work['basket_drop_flag'] = False

    if 'basket_items' in work.columns:
        work['prev_basket_items'] = work.groupby('customer_id')['basket_items'].shift(1)
        work['items_drop_flag'] = (
            work['basket_items'] < work['prev_basket_items'] * 0.7
        ).fillna(False)
    else:
        work['items_drop_flag'] = False

    if 'days_since_prev_purchase' in work.columns:
        customer_mean_gap = work.groupby('customer_id')['days_since_prev_purchase'].transform(
            lambda s: s.replace(0, np.nan).mean()
        )
        global_gap = work['days_since_prev_purchase'].replace(0, np.nan).median()
        if pd.isna(global_gap):
            global_gap = 0

        customer_mean_gap = customer_mean_gap.fillna(global_gap)
        work['frequency_drop_flag'] = (
            work['days_since_prev_purchase'] > customer_mean_gap * 1.5
        ).fillna(False)
    else:
        work['frequency_drop_flag'] = False

    latest = get_latest_rows(work)
    latest['cooling_score'] = (
        latest['basket_drop_flag'].astype(int)
        + latest['items_drop_flag'].astype(int)
        + latest['frequency_drop_flag'].astype(int)
    )
    latest['cooling_flag'] = latest['cooling_score'] > 0

    return latest[
        [
            'customer_id',
            'cooling_flag',
            'cooling_score',
            'basket_drop_flag',
            'items_drop_flag',
            'frequency_drop_flag',
        ]
    ].copy()


def build_feedback_map(feedback_df: pd.DataFrame) -> pd.DataFrame:
    if feedback_df is None or feedback_df.empty or 'customer_id' not in feedback_df.columns:
        return pd.DataFrame(columns=['customer_id', 'blocked_categories', 'feedback_status'])

    work = feedback_df.copy()

    if 'category' not in work.columns:
        work['category'] = ''

    if 'feedback_status' not in work.columns:
        work['feedback_status'] = ''

    work['category'] = work['category'].fillna('').astype(str)
    work['feedback_status'] = work['feedback_status'].fillna('').astype(str).str.lower().str.strip()

    blocked_statuses = {'not_interested', 'already_bought', 'already bought', 'not interested'}
    blocked = work[work['feedback_status'].isin(blocked_statuses)].copy()

    if len(blocked) > 0:
        blocked_map = blocked.groupby('customer_id')['category'].apply(list).reset_index()
        blocked_map = blocked_map.rename(columns={'category': 'blocked_categories'})
    else:
        blocked_map = pd.DataFrame(columns=['customer_id', 'blocked_categories'])

    latest_status = work.groupby('customer_id').tail(1)[['customer_id', 'feedback_status']].copy()

    result = latest_status.merge(blocked_map, on='customer_id', how='left')
    result['blocked_categories'] = result['blocked_categories'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    return result


def build_trigger_reason(row: pd.Series) -> str:
    reasons = []

    if bool(row.get('frequency_drop_flag', False)):
        reasons.append('падає частота замовлень')

    if bool(row.get('basket_drop_flag', False)):
        reasons.append('падає сума чека')

    if bool(row.get('items_drop_flag', False)):
        reasons.append('падає кількість товарів у кошику')

    if not reasons:
        return 'загальне охолодження активності'

    return ', '.join(reasons)


def build_individual_triggers(predicted_df: pd.DataFrame, feedback_df: pd.DataFrame | None = None) -> pd.DataFrame:
    latest = get_latest_rows(predicted_df)
    if 'is_currently_active' in latest.columns:
        latest = latest[latest['is_currently_active'] == True].copy()
    else:
        latest = latest[(latest['churn'] == 0)].copy()

    if 'cooling_flag' in latest.columns:
        latest = latest[latest['cooling_flag'] == True].copy()

    if len(latest) == 0:
        return pd.DataFrame(
            columns=[
                'customer_id',
                'risk_class',
                'trigger_reason',
                'suggested_category',
                'feedback_status',
                'recommended_action',
                'recommended_channel',
            ]
        )

    feedback_map = build_feedback_map(feedback_df)
    if len(feedback_map) > 0:
        latest = latest.merge(feedback_map, on='customer_id', how='left')
    else:
        latest['feedback_status'] = ''
        latest['blocked_categories'] = [[] for _ in range(len(latest))]

    latest['suggested_category'] = latest.apply(
        lambda row: choose_category_with_feedback(
            row.get('top_categories', ''),
            row.get('dominant_category', ''),
            row.get('blocked_categories', []),
        ),
        axis=1,
    )

    latest['trigger_reason'] = latest.apply(build_trigger_reason, axis=1)
    latest['recommended_action'] = latest.apply(
        lambda row: f"Персональне нагадування по категорії {row['suggested_category']}",
        axis=1,
    )
    latest['recommended_channel'] = latest.apply(recommend_channel, axis=1)

    cols = [
        'customer_id',
        'risk_class',
        'trigger_reason',
        'suggested_category',
        'feedback_status',
        'recommended_action',
        'recommended_channel',
    ]
    return latest[cols].sort_values(by='risk_class', ascending=False).reset_index(drop=True)


def apply_scenario_to_feature_row(
    customer_x_row: pd.Series,
    freq_change_pct: float,
    ticket_change_pct: float,
    promo_change_pct: float,
) -> pd.Series:
    updated = customer_x_row.copy()

    if 'days_since_prev_purchase' in updated.index:
        updated['days_since_prev_purchase'] = max(
            float(updated['days_since_prev_purchase']) * (1.0 - freq_change_pct),
            0.0
        )

    if 'customer_mean_gap' in updated.index:
        updated['customer_mean_gap'] = max(
            float(updated['customer_mean_gap']) * (1.0 - freq_change_pct * 0.5),
            0.0
        )

    if 'customer_order_count' in updated.index:
        updated['customer_order_count'] = float(updated['customer_order_count']) * (1.0 + freq_change_pct)

    if 'online_purchases' in updated.index:
        updated['online_purchases'] = float(updated['online_purchases']) * (1.0 + freq_change_pct)

    if 'avg_ticket_size' in updated.index:
        updated['avg_ticket_size'] = float(updated['avg_ticket_size']) * (1.0 + ticket_change_pct)

    if 'total_sales' in updated.index:
        updated['total_sales'] = float(updated['total_sales']) * (1.0 + ticket_change_pct)

    if 'avg_item_value' in updated.index:
        updated['avg_item_value'] = float(updated['avg_item_value']) * (1.0 + ticket_change_pct * 0.5)

    if 'promo_response_rate' in updated.index:
        updated['promo_response_rate'] = min(
            float(updated['promo_response_rate']) * (1.0 + promo_change_pct),
            1.0
        )

    if 'avg_discount_used' in updated.index:
        updated['avg_discount_used'] = float(updated['avg_discount_used']) * (1.0 + promo_change_pct * 0.5)

    if 'discount_impact' in updated.index:
        updated['discount_impact'] = float(updated['discount_impact']) * (1.0 + promo_change_pct * 0.3)

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

    detail_rows = []

    for idx, row in audience_df.iterrows():
        x_row = audience_X.loc[idx]
        base_probability = float(row.get('churn_probability', predict_probability_from_feature_row(x_row, artifacts)))

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