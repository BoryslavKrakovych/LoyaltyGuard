from __future__ import annotations

import joblib
import pandas as pd

from .xcelerator import align_columns_for_inference, build_customer_dataset


CLUSTER_NAMES = {
    0: 'Часті компактні кошики',
    1: 'Цінні стабільні клієнти',
    2: 'Рідкі великі замовлення',
    3: 'Промо-чутливі клієнти',
    4: 'Потрібна увага',
}


RISK_PRIORITY = {'high': 3, 'medium': 2, 'low': 1}


def _risk_class(probability: float) -> str:
    if probability <= 0.30:
        return 'low'
    if probability <= 0.60:
        return 'medium'
    return 'high'


def _recommended_action(row: pd.Series) -> str:
    if row['risk_class'] == 'high':
        return 'Термінова персональна знижка'
    if row['rfm_segment'] in ['At Risk', 'Lost']:
        return 'Реактиваційна кампанія'
    if bool(row['cooling_flag']):
        return 'Нагадування + добірка релевантних товарів'
    if row['rfm_segment'] == 'VIP / Champions':
        return 'Утримання без глибокої знижки'
    return 'Мяке персональне нагадування'


def _build_priority_score(row: pd.Series) -> float:
    score = float(row['churn_probability']) * 100
    score += 20 if row['risk_class'] == 'high' else 10 if row['risk_class'] == 'medium' else 0
    score += 10 if bool(row['cooling_flag']) else 0
    score += float(row.get('Monetary', 0)) * 0.02
    return round(score, 2)


def assign_clusters(customer_df: pd.DataFrame, kmeans_model_path: str) -> pd.DataFrame:
    result = customer_df.copy()
    try:
        artifacts = joblib.load(kmeans_model_path)
    except FileNotFoundError:
        result['cluster'] = -1
        result['cluster_name'] = 'Кластер не визначено'
        return result

    features = artifacts['features']
    scaler = artifacts['scaler']
    model = artifacts['model']

    X = result[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_scaled = scaler.transform(X)
    result['cluster'] = model.predict(X_scaled)
    result['cluster_name'] = result['cluster'].map(CLUSTER_NAMES).fillna('Інший кластер')
    return result


def perform_churn_and_cooling_analysis(
    df_featured: pd.DataFrame,
    model_path: str = 'models/churn_model.joblib',
    kmeans_model_path: str | None = None,
) -> pd.DataFrame:
    print('   -> Аналіз відтоку, RFM та активності...')

    if df_featured.empty:
        return pd.DataFrame()

    if 'customer_id' not in df_featured.columns:
        raise ValueError("У featured_data немає колонки 'customer_id'.")

    customer_df = build_customer_dataset(df_featured)

    try:
        artifacts = joblib.load(model_path)
        model = artifacts['model']
        features = artifacts['features']
        X_new = align_columns_for_inference(features, customer_df)
        customer_df['churn_probability'] = model.predict_proba(X_new)[:, 1]
    except FileNotFoundError:
        customer_df['churn_probability'] = 0.0

    base_gap = pd.to_numeric(customer_df.get('customer_mean_gap', 0), errors='coerce').fillna(0)
    base_gap = base_gap.where(base_gap > 0, 30)

    current_basket = pd.to_numeric(customer_df.get('basket_items', 0), errors='coerce').fillna(0)
    baseline_basket = pd.to_numeric(customer_df.get('customer_avg_basket_before', 0), errors='coerce').fillna(0)
    current_sales = pd.to_numeric(customer_df.get('total_sales', 0), errors='coerce').fillna(0)
    baseline_sales = pd.to_numeric(customer_df.get('customer_avg_order_value_before', 0), errors='coerce').fillna(0)

    customer_df['frequency_drop_flag'] = pd.to_numeric(
        customer_df.get('days_since_prev_purchase', 0), errors='coerce'
    ).fillna(0) > (base_gap * 1.5)

    customer_df['basket_drop_flag'] = (baseline_basket > 0) & (current_basket < (baseline_basket * 0.7))
    customer_df['amount_drop_flag'] = (baseline_sales > 0) & (current_sales < (baseline_sales * 0.7))
    customer_df['cooling_flag'] = (
        customer_df['frequency_drop_flag'] |
        customer_df['basket_drop_flag'] |
        customer_df['amount_drop_flag']
    )

    customer_df['risk_class'] = customer_df['churn_probability'].apply(_risk_class)
    customer_df['is_target'] = (
        customer_df['risk_class'].isin(['high', 'medium']) |
        customer_df['cooling_flag'] |
        customer_df['rfm_segment'].isin(['At Risk', 'Lost'])
    )

    if kmeans_model_path:
        customer_df = assign_clusters(customer_df, kmeans_model_path)
    else:
        customer_df['cluster'] = -1
        customer_df['cluster_name'] = 'Кластер не визначено'

    customer_df['recommended_action'] = customer_df.apply(_recommended_action, axis=1)
    customer_df['priority_score'] = customer_df.apply(_build_priority_score, axis=1)

    if 'dominant_category' not in customer_df.columns:
        customer_df['dominant_category'] = 'улюблені товари'

    queue = customer_df[[
        'customer_id', 'Recency', 'Frequency', 'Monetary',
        'R_score', 'F_score', 'M_score', 'RFM_score', 'rfm_segment',
        'churn_probability', 'risk_class',
        'frequency_drop_flag', 'basket_drop_flag', 'amount_drop_flag', 'cooling_flag',
        'cluster', 'cluster_name', 'dominant_category', 'recommended_action',
        'is_target', 'priority_score'
    ]].copy()

    queue = queue.sort_values(
        by=['is_target', 'priority_score', 'Monetary', 'churn_probability'],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return queue


def predict_single_customer_what_if(
    customer_row: pd.Series,
    model_path: str,
    frequency_change_pct: float,
    ticket_change_pct: float,
    promo_change_pp: float
) -> dict:
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    features = artifacts['features']

    row = customer_row.copy()

    frequency_multiplier = max(0.2, 1 + frequency_change_pct / 100.0)
    ticket_multiplier = max(0.2, 1 + ticket_change_pct / 100.0)

    if 'purchase_frequency' in row:
        row['purchase_frequency'] = float(row['purchase_frequency']) * frequency_multiplier
    if 'days_since_prev_purchase' in row:
        row['days_since_prev_purchase'] = float(row['days_since_prev_purchase']) / frequency_multiplier
    if 'avg_ticket_size' in row:
        row['avg_ticket_size'] = float(row['avg_ticket_size']) * ticket_multiplier
    if 'avg_purchase_value' in row:
        row['avg_purchase_value'] = float(row['avg_purchase_value']) * ticket_multiplier
    if 'total_sales' in row:
        row['total_sales'] = float(row['total_sales']) * ticket_multiplier
    if 'promo_response_rate' in row:
        row['promo_response_rate'] = min(
            1.0,
            max(0.0, float(row['promo_response_rate']) + promo_change_pp / 100.0)
        )

    X = pd.DataFrame([row])
    X = align_columns_for_inference(features, X)
    probability = float(model.predict_proba(X)[:, 1][0])

    return {
        'churn_probability': probability,
        'risk_class': _risk_class(probability),
    }