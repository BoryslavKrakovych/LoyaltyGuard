from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = [
    'customer_id', 'rfm_segment', 'favorite_category', 'churn_probability_pct',
    'push_text', 'channel', 'risk_class', 'cooling_flag',
    'cluster_name', 'recommended_action'
]


def _load_feedback_log(feedback_path: str | Path) -> pd.DataFrame:
    path = Path(feedback_path)
    if not path.exists():
        return pd.DataFrame(columns=['customer_id', 'category', 'feedback_type', 'date'])
    return pd.read_csv(path)


def _apply_feedback_suppression(df: pd.DataFrame, feedback_path: str | Path) -> pd.DataFrame:
    feedback = _load_feedback_log(feedback_path)
    if feedback.empty:
        return df

    feedback['customer_id'] = feedback['customer_id'].astype(str)
    feedback['category'] = feedback['category'].astype(str).str.lower().str.strip()

    result = df.copy()
    result['customer_id'] = result['customer_id'].astype(str)
    result['favorite_category_norm'] = result['favorite_category'].astype(str).str.lower().str.strip()

    block_pairs = set(zip(feedback['customer_id'], feedback['category']))
    mask = result.apply(lambda row: (row['customer_id'], row['favorite_category_norm']) not in block_pairs, axis=1)
    result = result[mask].copy()
    return result.drop(columns=['favorite_category_norm'])


def _build_message_and_channel(row: pd.Series) -> tuple[str, str]:
    category = str(row['favorite_category']).strip()
    risk_pct = float(pd.to_numeric(row.get('churn_probability', 0.0), errors='coerce') or 0.0) * 100.0
    segment = str(row.get('rfm_segment', ''))

    if risk_pct > 70 and 'At Risk' in segment:
        return f'🔥 Повертайтесь! Знижка -20% на {category}', 'Push + SMS'
    if risk_pct > 50:
        return f'✨ Ми зібрали для вас новинки: {category}', 'Push'
    return f'👋 Здається, вам сподобається: {category}', 'Push'


def generate_push_campaign(
    analysis_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    feedback_path: str | Path = 'feedback_log.csv',
) -> pd.DataFrame:
    print('   -> Генерація черги на порятунок і push-повідомлень...')

    if analysis_df.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    if profiles_df.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    final = analysis_df.merge(profiles_df, on='customer_id', how='inner')
    final['favorite_category'] = final['favorite_category'].astype(str).str.strip()
    final = final[final['favorite_category'] != ''].copy()

    final['cooling_flag'] = final['cooling_flag'].fillna(False).astype(bool)
    final['is_target'] = final['is_target'].fillna(False).astype(bool)
    final['churn_probability'] = pd.to_numeric(final['churn_probability'], errors='coerce').fillna(0.0)

    targets = final[final['is_target']].copy()
    if targets.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    messages = targets.apply(_build_message_and_channel, axis=1)
    targets['push_text'] = [item[0] for item in messages]
    targets['channel'] = [item[1] for item in messages]
    targets['churn_probability_pct'] = (targets['churn_probability'] * 100).round(1)

    targets = _apply_feedback_suppression(targets, feedback_path)
    targets = targets.sort_values(
        by=['priority_score', 'Monetary', 'churn_probability'],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return targets[[
        'customer_id', 'rfm_segment', 'favorite_category', 'churn_probability_pct',
        'push_text', 'channel', 'risk_class', 'cooling_flag',
        'cluster_name', 'recommended_action'
    ]]


def filter_campaign_audience(
    campaign_df: pd.DataFrame,
    risk_class: str | None = None,
    rfm_segment: str | None = None,
    category: str | None = None,
    cluster_name: str | None = None,
) -> pd.DataFrame:
    result = campaign_df.copy()

    if risk_class and risk_class != 'Усі':
        result = result[result['risk_class'] == risk_class]
    if rfm_segment and rfm_segment != 'Усі':
        result = result[result['rfm_segment'] == rfm_segment]
    if category and category != 'Усі':
        result = result[result['favorite_category'] == category]
    if cluster_name and cluster_name != 'Усі':
        result = result[result['cluster_name'] == cluster_name]

    return result.reset_index(drop=True)


def append_feedback(feedback_path: str | Path, customer_id: str, category: str, feedback_type: str, date_value: str) -> Path:
    path = Path(feedback_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([
        {
            'customer_id': str(customer_id),
            'category': str(category),
            'feedback_type': str(feedback_type),
            'date': str(date_value),
        }
    ])

    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(path, index=False)
    return path