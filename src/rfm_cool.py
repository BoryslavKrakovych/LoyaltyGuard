from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from xcelerator import align_columns_for_inference


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


def _load_joblib_dataframe(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    obj = joblib.load(path)
    return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)


def _safe_numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors='coerce').fillna(default)
    return pd.Series(default, index=df.index, dtype='float64')


def build_customer_dataset(
    df_featured: pd.DataFrame,
    rfm_path: str | Path | None = None,
    clusters_path: str | Path | None = None,
    rfm_df: pd.DataFrame | None = None,
    clusters_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if df_featured.empty:
        return pd.DataFrame()

    work = df_featured.copy()
    if 'transaction_date' in work.columns:
        work['transaction_date'] = pd.to_datetime(work['transaction_date'], errors='coerce')
        sort_cols = ['customer_id', 'transaction_date']
        if 'transaction_id' in work.columns:
            sort_cols.append('transaction_id')
        latest = work.sort_values(sort_cols).groupby('customer_id').tail(1).copy()
    else:
        latest = work.drop_duplicates('customer_id', keep='last').copy()

    rfm_df = rfm_df.copy() if rfm_df is not None else _load_joblib_dataframe(rfm_path)
    if not rfm_df.empty:
        rename_map = {'recency': 'Recency', 'frequency': 'Frequency', 'monetary': 'Monetary'}
        rfm_df = rfm_df.rename(columns=rename_map)
        latest = latest.merge(rfm_df, on='customer_id', how='left')

    clusters_df = clusters_df.copy() if clusters_df is not None else _load_joblib_dataframe(clusters_path)
    if not clusters_df.empty:
        clusters_df = clusters_df.rename(columns={
            'customer_cluster': 'cluster',
            'customer_cluster_name': 'cluster_name',
        })
        latest = latest.merge(clusters_df, on='customer_id', how='left')

    if 'product_text' not in latest.columns and 'product_name' in latest.columns:
        latest['product_text'] = latest['product_name']
    if 'dominant_category' not in latest.columns and 'category' in latest.columns:
        latest['dominant_category'] = latest['category']
    if 'dominant_category' not in latest.columns:
        latest['dominant_category'] = 'Загальний асортимент'
    if 'cluster' not in latest.columns:
        latest['cluster'] = -1
    if 'cluster_name' not in latest.columns:
        latest['cluster_name'] = 'Кластер не визначено'

    return latest


def _extract_model_and_features(artifacts):
    if hasattr(artifacts, 'model') and hasattr(artifacts, 'X_train_columns'):
        return artifacts.model, list(artifacts.X_train_columns)
    if isinstance(artifacts, dict):
        return artifacts['model'], list(artifacts.get('features', []))
    raise ValueError('Невідомий формат модельного артефакту.')


def perform_churn_and_cooling_analysis(
    df_featured: pd.DataFrame,
    model_path: str = 'models/churn_artifacts.joblib',
    rfm_path: str | Path | None = None,
    clusters_path: str | Path | None = None,
) -> pd.DataFrame:
    print('   -> Аналіз відтоку, RFM та активності...')

    if df_featured.empty:
        return pd.DataFrame()
    if 'customer_id' not in df_featured.columns:
        raise ValueError("У featured_data немає колонки 'customer_id'.")

    customer_df = build_customer_dataset(df_featured, rfm_path=rfm_path, clusters_path=clusters_path)

    try:
        artifacts = joblib.load(model_path)
        model, features = _extract_model_and_features(artifacts)
        X_ref = pd.DataFrame(columns=features)
        X_new = align_columns_for_inference(X_ref, customer_df)
        customer_df['churn_probability'] = model.predict_proba(X_new)[:, 1]
    except FileNotFoundError:
        customer_df['churn_probability'] = 0.0

    base_gap = _safe_numeric_series(customer_df, 'customer_mean_gap', 0.0)
    base_gap = base_gap.where(base_gap > 0, 30)

    current_basket = _safe_numeric_series(customer_df, 'basket_items', 0.0)
    baseline_basket = _safe_numeric_series(customer_df, 'customer_avg_basket_before', 0.0)
    current_sales = _safe_numeric_series(customer_df, 'total_sales', 0.0)
    baseline_sales = _safe_numeric_series(customer_df, 'customer_avg_order_value_before', 0.0)
    days_since_prev_purchase = _safe_numeric_series(customer_df, 'days_since_prev_purchase', 0.0)

    customer_df['frequency_drop_flag'] = days_since_prev_purchase > (base_gap * 1.5)
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

    customer_df['recommended_action'] = customer_df.apply(_recommended_action, axis=1)
    customer_df['priority_score'] = customer_df.apply(_build_priority_score, axis=1)

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
    promo_change_pp: float,
) -> dict:
    artifacts = joblib.load(model_path)
    model, features = _extract_model_and_features(artifacts)

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
        row['promo_response_rate'] = min(1.0, max(0.0, float(row['promo_response_rate']) + promo_change_pp / 100.0))

    X = pd.DataFrame([row])
    X_ref = pd.DataFrame(columns=features)
    X = align_columns_for_inference(X_ref, X)
    probability = float(model.predict_proba(X)[:, 1][0])

    return {
        'churn_probability': probability,
        'risk_class': _risk_class(probability),
    }
