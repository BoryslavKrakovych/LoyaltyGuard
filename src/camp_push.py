from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = [
    'customer_id', 'favorite_category', 'rfm_segment', 'cluster_name',
    'churn_probability_pct', 'risk_class', 'cooling_flag',
    'recommended_action', 'push_text'
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


def _build_push_text(row: pd.Series) -> str:
    category = str(row['favorite_category']).strip().title()
    action = str(row['recommended_action'])
    risk = str(row['risk_class'])

    if risk == 'high':
        return f"Ми сумуємо за вами. Для категорії '{category}' підготували персональну пропозицію."
    if bool(row['cooling_flag']):
        return f"Бачимо спад активності. Перегляньте добірку у категорії '{category}' та поверніться до покупок вигідно."
    if action == 'Утримання без глибокої знижки':
        return f"Дякуємо, що ви з нами. Для вас зібрані новинки у категорії '{category}'."
    return f"Персональна пропозиція для вас: категорія '{category}' може бути особливо цікавою саме зараз."


def generate_push_campaign(
    analysis_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    feedback_path: str | Path = 'feedback_log.csv',
) -> pd.DataFrame:
    print('   -> Генерація черги на порятунок і push-повідомлень...')

    if analysis_df.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    final = analysis_df.merge(profiles_df, on='customer_id', how='left')
    final['favorite_category'] = final['favorite_category'].fillna(
        final.get('dominant_category', pd.Series('улюблені товари', index=final.index))
    ).fillna('улюблені товари')
    final['cooling_flag'] = final['cooling_flag'].fillna(False).astype(bool)
    final['is_target'] = final['is_target'].fillna(False).astype(bool)
    final['churn_probability'] = pd.to_numeric(final['churn_probability'], errors='coerce').fillna(0.0)

    targets = final[final['is_target']].copy()
    if targets.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    targets['push_text'] = targets.apply(_build_push_text, axis=1)
    targets['churn_probability_pct'] = (targets['churn_probability'] * 100).round(1).astype(str) + '%'

    targets = _apply_feedback_suppression(targets, feedback_path)
    targets = targets.sort_values(
        by=['priority_score', 'Monetary', 'churn_probability'],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return targets[[
        'customer_id', 'favorite_category', 'rfm_segment', 'cluster_name',
        'churn_probability_pct', 'risk_class', 'cooling_flag',
        'recommended_action', 'push_text'
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
