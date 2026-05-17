"""
Customer clustering with BEHAVIOR-BASED names instead of hard-coded labels.

Cluster names are derived from the cluster centroid's rank on three axes:
  - Frequency      (online_purchases / basket_items)
  - Ticket size    (avg_ticket_size / total_sales)
  - Promo response (promo_response_rate)

For K=4 clusters this gives unambiguous names like:
  'Часті покупки / великий чек'
  'Часті покупки / малий чек'
  'Рідкі покупки / великий чек'
  'Рідкі покупки / малий чек'
plus an optional '/ реагує на промо' descriptor if relevant.

If two clusters land on the same label (rare, only with K >= 5), a '#N'
suffix is appended.

Also exposes cluster_strategy_text(name) → (title, description) for the UI.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


def _pick_column(cluster_means: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in cluster_means.columns:
            return col
    return None


def _describe_clusters_by_rank(cluster_means: pd.DataFrame) -> dict[int, str]:
    """Assign presentation-friendly cluster names.

    The labels match the project presentation and demo screenshots:
      - Клієнти з високим ризиком відтоку
      - Клієнти зі стандартною поведінкою
      - Активні покупці
      - Клієнти з високим середнім чеком
      - VIP клієнти
    """
    available = list(cluster_means.index)
    names: dict[int, str] = {}

    def pick_best(candidates: list[str], ascending: bool = False) -> Optional[int]:
        for col in candidates:
            if col in cluster_means.columns:
                ordered = cluster_means.loc[[i for i in available if i not in names], col].sort_values(ascending=ascending)
                if len(ordered) > 0:
                    return int(ordered.index[0])
        return None

    high_risk = pick_best(['churn_probability_percent', 'days_since_last_purchase', 'customer_mean_gap'])
    if high_risk is not None:
        names[high_risk] = 'Клієнти з високим ризиком відтоку'

    high_ticket = pick_best(['avg_ticket_size', 'avg_purchase_value'])
    if high_ticket is not None:
        names[high_ticket] = 'Клієнти з високим середнім чеком'

    vip = pick_best(['total_sales'])
    if vip is not None:
        names[vip] = 'VIP клієнти'

    active = pick_best(['online_purchases', 'total_transactions', 'basket_items'])
    if active is not None:
        # Захист від refund-heavy кластерів: висока кількість транзакцій ще не
        # означає "активний покупець". Якщо середній чек кластера <= 0 або
        # середній net total_sales <= 0 — не вішаємо ярлик 'Активні покупці',
        # хай випадає в 'Клієнти зі стандартною поведінкою'.
        ticket_col = next(
            (c for c in ['avg_ticket_size', 'avg_purchase_value', 'total_sales']
             if c in cluster_means.columns),
            None,
        )
        if ticket_col is None or cluster_means.loc[active, ticket_col] > 0:
            names[active] = 'Активні покупці'

    for cluster_id in available:
        if int(cluster_id) not in names:
            names[int(cluster_id)] = 'Клієнти зі стандартною поведінкою'

    return names


def _ensure_unique_names(names: dict[int, str]) -> dict[int, str]:
    """Append '#N' suffix if multiple clusters got the same descriptive name."""
    counts: dict[str, int] = {}
    for n in names.values():
        counts[n] = counts.get(n, 0) + 1

    counters: dict[str, int] = {n: 0 for n, c in counts.items() if c > 1}
    result: dict[int, str] = {}
    for cluster_id, name in names.items():
        if counts[name] == 1:
            result[cluster_id] = name
        else:
            counters[name] += 1
            result[cluster_id] = f'{name} #{counters[name]}'
    return result


def build_customer_clusters(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    work = df.copy()

    candidate_cols = [
        'churn_probability_percent', 'days_since_last_purchase', 'customer_mean_gap',
        'total_sales', 'avg_ticket_size', 'basket_items',
        'online_purchases', 'in_store_purchases',
        'promo_response_rate', 'distinct_products', 'distinct_categories',
    ]
    feature_cols = [col for col in candidate_cols if col in work.columns]

    if 'customer_id' not in work.columns or len(feature_cols) < 2:
        return pd.DataFrame(columns=['customer_id', 'customer_cluster', 'customer_cluster_name'])

    if 'transaction_date' in work.columns:
        latest = (
            work.sort_values(['customer_id', 'transaction_date'])
                .groupby('customer_id').tail(1).copy()
        )
    else:
        latest = work.groupby('customer_id').tail(1).copy()

    model_data = latest[['customer_id'] + feature_cols].copy()
    model_data[feature_cols] = model_data[feature_cols].fillna(model_data[feature_cols].median())

    scaler = StandardScaler()
    X = scaler.fit_transform(model_data[feature_cols])

    n_clusters = max(2, min(n_clusters, len(model_data)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    model_data['customer_cluster'] = kmeans.fit_predict(X)

    cluster_means = model_data.groupby('customer_cluster')[feature_cols].mean()

    score_cols = [c for c in ['total_sales', 'avg_ticket_size', 'basket_items', 'online_purchases']
                  if c in cluster_means.columns]
    if score_cols:
        cluster_means = (
            cluster_means.assign(_score=cluster_means[score_cols].sum(axis=1))
            .sort_values('_score', ascending=False)
            .drop(columns=['_score'])
        )

    raw_names = _describe_clusters_by_rank(cluster_means)
    unique_names = _ensure_unique_names(raw_names)

    model_data['customer_cluster_name'] = model_data['customer_cluster'].map(unique_names)
    return model_data[['customer_id', 'customer_cluster', 'customer_cluster_name']]


def cluster_strategy_text(cluster_name: str) -> tuple[str, str]:
    """Map presentation cluster name → (strategy title, description)."""
    lower = str(cluster_name).lower()

    if 'високим ризиком' in lower:
        return ('🚨 Пріоритетне утримання',
                'Основна аудиторія для retention: персональна знижка та швидка комунікація. '
                'Якщо ризик понад 70% — Push + SMS, інакше SMS.')
    if 'стандартною поведінкою' in lower:
        return ('📌 Базова підтримка лояльності',
                'Короткі SMS-нагадування та невеликі персональні пропозиції без надмірного бюджету.')
    if 'активні покупці' in lower:
        return ('🛒 Підтримка активності',
                'Персональні рекомендації та крос-сейл, щоб зберегти частоту покупок.')
    if 'високим середнім чеком' in lower:
        return ('💳 Збереження високого чека',
                'Пропозиції на суміжні категорії та персональні добірки, щоб не знижувати середній чек.')
    if 'vip' in lower:
        return ('👑 VIP-утримання',
                'Найбільш цінні клієнти: персональні пропозиції, ранній доступ і пріоритетна комунікація.')

    return ('💡 Загальна стратегія',
            'Сегментована комунікація відповідно до поведінки клієнта.')
