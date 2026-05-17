"""
Customer clustering with BEHAVIOR-BASED names instead of hard-coded labels.

Pipeline:
  1) З latest-знімка клієнтів відокремлюємо net-refund аномалії
     (total_sales <= 0 або avg_ticket_size <= 0). Вони не йдуть у KMeans
     взагалі — отримують спеціальний кластер ANOMALY_CLUSTER_ID (-1)
     з ім'ям ANOMALY_CLUSTER_NAME. Це усуває проблему, коли 2-3 клієнти
     з негативним чеком утворювали власний мікрокластер #N з ярликом
     'Клієнти зі стандартною поведінкою' і потрапляли в retention-кампанію
     як 'Базова підтримка лояльності', хоча насправді вони системно
     повертають товар.
  2) Нормальних клієнтів кластеризуємо KMeans (StandardScaler + k=5).
  3) Імена кластерів видаються евристикою _describe_clusters_by_rank
     на основі агрегатів по фічах (churn, ticket, sales, активність).
  4) Дублі імен отримують суфікс #N через _ensure_unique_names.

Також експортується cluster_strategy_text(name) → (title, description)
для UI — включно з окремою гілкою для аномального кластера.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


# ── Константи для аномального кластера ────────────────────────────────────
ANOMALY_CLUSTER_ID = -1
ANOMALY_CLUSTER_NAME = 'Net-refund клієнти (виключити з кампанії)'


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
        # Подвійний запобіжник: після pre-filter аномалій сюди вже не мали б
        # доходити негативні чеки, але про всяк випадок лишаємо перевірку —
        # якщо середній чек кластера все одно <= 0, ярлик 'Активні покупці'
        # не вішаємо, хай випадає в 'Клієнти зі стандартною поведінкою'.
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


def _split_anomalous(model_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Відокремлюємо net-refund клієнтів від нормальних.

    Net-refund клієнт — той, у кого по агрегатах за весь період:
      - total_sales <= 0      (повернень більше, ніж покупок у грошах), або
      - avg_ticket_size <= 0  (середній чек від'ємний)

    Для retention такі клієнти безглузді: їм нема чого «утримувати», вони
    системно повертають товар. До того ж KMeans на них реагує болісно —
    кілька таких клієнтів утворюють власний мікрокластер у далекому кутку
    простору ознак і з'їдають один із k слотів сегментації.

    Повертає (normal, anomalous) у вигляді двох незалежних DataFrame.
    """
    if model_data.empty:
        return model_data.copy(), model_data.iloc[0:0].copy()

    mask = pd.Series(False, index=model_data.index)
    if 'total_sales' in model_data.columns:
        mask |= pd.to_numeric(model_data['total_sales'], errors='coerce').fillna(0) <= 0
    if 'avg_ticket_size' in model_data.columns:
        mask |= pd.to_numeric(model_data['avg_ticket_size'], errors='coerce').fillna(0) <= 0

    anomalous = model_data[mask].copy()
    normal = model_data[~mask].copy()
    return normal, anomalous


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=['customer_id', 'customer_cluster', 'customer_cluster_name'])


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
        return _empty_result()

    if 'transaction_date' in work.columns:
        latest = (
            work.sort_values(['customer_id', 'transaction_date'])
                .groupby('customer_id').tail(1).copy()
        )
    else:
        latest = work.groupby('customer_id').tail(1).copy()

    model_data = latest[['customer_id'] + feature_cols].copy()
    model_data[feature_cols] = model_data[feature_cols].fillna(model_data[feature_cols].median())

    # ── Виокремлюємо net-refund аномалії ДО KMeans ──
    normal, anomalous = _split_anomalous(model_data)

    # Якщо після фільтрації нормальних клієнтів недостатньо для кластеризації —
    # повертаємо тільки аномальних (якщо вони є), щоб UI міг чесно показати,
    # що звичайних сегментів просто немає.
    if len(normal) < 2:
        if anomalous.empty:
            return _empty_result()
        anomalous['customer_cluster'] = ANOMALY_CLUSTER_ID
        anomalous['customer_cluster_name'] = ANOMALY_CLUSTER_NAME
        return anomalous[['customer_id', 'customer_cluster', 'customer_cluster_name']]

    scaler = StandardScaler()
    X = scaler.fit_transform(normal[feature_cols])

    n_clusters = max(2, min(n_clusters, len(normal)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    normal['customer_cluster'] = kmeans.fit_predict(X)

    cluster_means = normal.groupby('customer_cluster')[feature_cols].mean()

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

    normal['customer_cluster_name'] = normal['customer_cluster'].map(unique_names)

    parts = [normal[['customer_id', 'customer_cluster', 'customer_cluster_name']]]

    if not anomalous.empty:
        anomalous['customer_cluster'] = ANOMALY_CLUSTER_ID
        anomalous['customer_cluster_name'] = ANOMALY_CLUSTER_NAME
        parts.append(anomalous[['customer_id', 'customer_cluster', 'customer_cluster_name']])

    return pd.concat(parts, ignore_index=True)


def cluster_strategy_text(cluster_name: str) -> tuple[str, str]:
    """Map presentation cluster name → (strategy title, description)."""
    lower = str(cluster_name).lower()

    if 'net-refund' in lower or 'виключити з кампанії' in lower:
        return ('⚠️ Виключити з retention-кампанії',
                'Клієнти з негативним нетто-оборотом за весь період — повернули більше, ніж купили. '
                'У retention-кампанії не беруть участь: окрема перевірка на потенційний фрод, '
                'системні проблеми з товаром або помилки в даних.')
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
