from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


def build_customer_clusters(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    work = df.copy()

    candidate_cols = [
        'total_sales',
        'avg_ticket_size',
        'basket_items',
        'online_purchases',
        'in_store_purchases',
        'promo_response_rate',
        'distinct_products',
        'distinct_categories',
    ]
    feature_cols = [col for col in candidate_cols if col in work.columns]

    if 'customer_id' not in work.columns or len(feature_cols) < 2:
        return pd.DataFrame(columns=['customer_id', 'customer_cluster', 'customer_cluster_name'])

    latest = work.sort_values(['customer_id', 'transaction_date']).groupby('customer_id').tail(1).copy()
    model_data = latest[['customer_id'] + feature_cols].copy()
    model_data[feature_cols] = model_data[feature_cols].fillna(model_data[feature_cols].median())

    scaler = StandardScaler()
    X = scaler.fit_transform(model_data[feature_cols])

    n_clusters = max(2, min(n_clusters, len(model_data)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    model_data['customer_cluster'] = kmeans.fit_predict(X)

    cluster_stats = model_data.groupby('customer_cluster')[feature_cols].mean().copy()

    score_cols = [col for col in ['total_sales', 'avg_ticket_size', 'basket_items', 'online_purchases'] if col in cluster_stats.columns]
    cluster_stats['business_score'] = cluster_stats[score_cols].sum(axis=1)

    cluster_stats = cluster_stats.sort_values('business_score', ascending=False).reset_index()

    fixed_names = [
        'VIP клієнти',
        'Активні покупці',
        'Помірні покупці',
        'Малоактивні клієнти',
    ]

    rank_to_name = {}
    for i, row in cluster_stats.iterrows():
        if i < len(fixed_names):
            rank_to_name[int(row['customer_cluster'])] = fixed_names[i]
        else:
            rank_to_name[int(row['customer_cluster'])] = f'Сегмент {i + 1}'

    model_data['customer_cluster_name'] = model_data['customer_cluster'].map(rank_to_name)

    return model_data[['customer_id', 'customer_cluster', 'customer_cluster_name']]