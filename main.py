from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from collections import Counter

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as _mpl
CHART_COLORS = [
    '#2563eb',  # blue
    '#7c3aed',  # violet
    '#059669',  # green
    '#d97706',  # orange
    '#dc2626',  # red
    '#0891b2',  # cyan
    '#c026d3',  # pink
    '#ea580c',  # deep orange
]


def _apply_mpl_theme(dark: bool) -> None:
    if dark:
        _mpl.rcParams.update({
            'figure.facecolor':  '#13151f',
            'axes.facecolor':    '#13151f',
            'axes.edgecolor':    '#2a2e44',
            'axes.labelcolor':   '#8891aa',
            'axes.titlecolor':   '#c8d3f5',
            'text.color':        '#c8d3f5',
            'xtick.color':       '#4a5580',
            'ytick.color':       '#4a5580',
            'grid.color':        '#1e2233',
            'legend.facecolor':  '#13151f',
            'legend.edgecolor':  '#2a2e44',
            'legend.labelcolor': '#8891aa',
            'figure.edgecolor':  '#13151f',
            'savefig.facecolor': '#13151f',
            'savefig.edgecolor': '#13151f',
            'axes.prop_cycle': _mpl.cycler(color=[
                '#7c9fe6',
                '#a78bfa',
                '#34d399',
                '#fbbf24',
                '#f87171',
                '#38bdf8',
                '#e879f9',
                '#fb923c',
            ]),
        })
    else:
        _mpl.rcParams.update({
            'figure.facecolor':  '#ffffff',
            'axes.facecolor':    '#ffffff',
            'axes.edgecolor':    '#d1d5db',
            'axes.labelcolor':   '#111827',
            'axes.titlecolor':   '#111827',
            'text.color':        '#111827',
            'xtick.color':       '#374151',
            'ytick.color':       '#374151',
            'grid.color':        '#e5e7eb',
            'legend.facecolor':  '#ffffff',
            'legend.edgecolor':  '#d1d5db',
            'legend.labelcolor': '#111827',
            'figure.edgecolor':  '#ffffff',
            'savefig.facecolor': '#ffffff',
            'savefig.edgecolor': '#ffffff',
            'axes.prop_cycle': _mpl.cycler(color=CHART_COLORS),
        })

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WordCloud = None
    WORDCLOUD_AVAILABLE = False

from src.churn_model import ChurnModelService, attach_predictions, feature_importance_table
from src.clustering import build_customer_clusters, cluster_strategy_text
from src.config import SAVED_CHURN_MODEL_PATH
from src.data_loader import (
    CANONICAL_TRANSACTION_COLUMNS,
    infer_column_mapping,
    list_database_tables,
    list_mapping_templates,
    load_and_prepare_transactions_from_dataframe,
    load_mapping_template,
    load_table_bytes,
    load_table_from_database,
    preview_column_info,
    save_mapping_template,
    template_matches_schema,
)
from src.feature_engineering import add_base_features, build_forward_churn_dataset, build_training_matrices
from src.retention import estimate_campaign_effect, get_latest_rows
from src.rfm import build_rfm_table
from src.ui_helpers import download_dataframe_button, metric_row, plot_roc_curve
from src.product_analytics import (
    WORD2VEC_AVAILABLE,
    train_word2vec,
    cluster_products,
    semantic_cluster_summary,
)
from src.categorization import (
    categorize_product,
    categorize_dataframe,
    FALLBACK_CATEGORY,
)
from src.feedback_model import (
    autodetect_feedback_columns,
    prepare_feedback_training_set,
    FeedbackResponseService,
    attach_feedback_recommendations,
)

st.set_page_config(page_title='LoyaltyGuard', page_icon='🛡️', layout='wide', initial_sidebar_state='expanded')

APP_VERSION = 'clean-ui-index-fix-v4'
CATEGORY_OTHER_LABEL = 'Інше'

GENERIC_CATEGORY_VALUES = {
    '',
    'unknown',
    'general merchandise',
    'general assortment',
    'загальний асортимент',
    'невизначена категорія',
    'інший кластер',
    'інше',
    'other',
    'nan',
    'none',
    'null',
    'n/a',
    '<na>',
}

AUTO_CATEGORY_RULES = {
    'Seasonal & Holiday': [
        'christmas', 'xmas', 'santa', 'snowman', 'snowflake', 'reindeer',
        'bauble', 'stocking', 'advent', 'noel', 'holiday', 'festive',
        'angel', 'tree', 'easter', 'valentine', 'halloween', 'fairy'
    ],
    'Home Decor': [
        'decor', 'decoration', 'ornament', 'vase', 'frame', 'mirror',
        'clock', 'wreath', 'sign', 'plaque', 'hanger', 'hook', 'wall',
        'ceramic', 'porcelain', 'trinket', 'lantern', 'candle', 'candles',
        'holder', 'tealight', 'lamp', 'heart', 'photo', 'picture',
        'wooden', 'letters', 'block', 'bird', 'garland'
    ],
    'Kitchen & Dining': [
        'plate', 'bowl', 'dish', 'tray', 'fork', 'knife', 'spoon',
        'cutlery', 'teapot', 'saucer', 'jar', 'mug', 'cup',
        'glass', 'bottle', 'flask', 'tumbler', 'goblet', 'kitchen',
        'cakestand', 'cake', 'napkin', 'lunchbox', 'lunch', 'tin',
        'jug', 'recipe', 'tea', 'coffee'
    ],
    'Textiles & Soft Furnishings': [
        'blanket', 'throw', 'towel', 'rug', 'mat', 'curtain', 'linen',
        'fabric', 'knit', 'apron', 'cloth', 'cover', 'cushion',
        'pillow', 'doormat', 'textile', 'bunting', 'baguette'
    ],
    'Bags & Accessories': [
        'bag', 'handbag', 'purse', 'wallet', 'case', 'pouch', 'backpack',
        'shopper', 'shop', 'luggage', 'necklace', 'bracelet', 'ring',
        'earring', 'brooch', 'scarf', 'hat', 'glove', 'jewellery', 'jewelry'
    ],
    'Stationery & Gift': [
        'card', 'cards', 'paper', 'notebook', 'journal', 'pen', 'pencil',
        'sticker', 'craft', 'ribbon', 'wrap', 'wrapping', 'tag',
        'gift', 'present', 'party', 'celebration', 'balloon',
        'birthday', 'postcard', 'album'
    ],
    'Garden & Outdoor': [
        'garden', 'plant', 'flower', 'pot', 'outdoor', 'birdhouse',
        'feeder', 'watering', 'planter', 'gardeners', 'bucket'
    ],
    'Kids & Toys': [
        'toy', 'doll', 'kids', 'child', 'children', 'baby', 'game',
        'teddy', 'puzzle', 'train', 'car', 'blocks'
    ],
}

OFFER_TYPE_LABELS = {
    'Auto': 'Авто',
    'Discount': 'Знижка',
    'Bonus points': 'Бонусні бали',
    'Personal recommendation': 'Персональна рекомендація',
    'Reminder': 'Нагадування',
    'Bundle offer': 'Комплектна пропозиція',
}

CHANNEL_LABELS = {
    'Auto': 'Авто',
    'SMS': 'SMS',
    'E-mail': 'E-mail',
    'Push': 'Push',
    'App': 'App',
    'SMS + e-mail': 'SMS + e-mail',
    'Push + e-mail': 'Push + e-mail',
}

CANONICAL_COLUMN_LABELS = {
    'customer_id': 'ID клієнта',
    'transaction_date': 'Дата транзакції',
    'transaction_id': 'ID транзакції',
    'product_id': 'ID товару',
    'product_name': 'Назва товару',
    'quantity': 'Кількість',
    'price': 'Ціна',
    'country': 'Країна / регіон',
    'churn': 'Churn / target',
}


def is_missing_like(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {'', 'nan', 'none', 'null', 'n/a', 'na', '<na>'}


def clean_text_value(value: Any, default: str = '') -> str:
    if is_missing_like(value):
        return default
    return str(value).strip()


def normalize_category_label(value: Any) -> str:
    text = clean_text_value(value, '')
    if text == '':
        return ''

    if '—' in text:
        text = clean_text_value(text.split('—', 1)[0], '')

    if ' - ' in text:
        text = clean_text_value(text.split(' - ', 1)[0], '')

    if text.lower() in GENERIC_CATEGORY_VALUES:
        return ''

    return text


def format_category_output(value: Any) -> str:
    cleaned = normalize_category_label(value)
    if cleaned:
        return cleaned
    return CATEGORY_OTHER_LABEL


def split_categories(value: Any) -> list[str]:
    if is_missing_like(value):
        return []

    items: list[str] = []
    for part in str(value).split(','):
        cleaned = normalize_category_label(part)
        if cleaned:
            items.append(cleaned)
    return items


def unique_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def sorted_options(series: pd.Series) -> list[str]:
    values = [clean_text_value(value, '') for value in series.tolist()]
    values = [value for value in values if value != '']
    return sorted(pd.Series(values).drop_duplicates().tolist())


def infer_category_from_text(text: str) -> str:
    tokens = set(re.findall(r'[a-z0-9]+', str(text).lower()))
    if not tokens:
        return CATEGORY_OTHER_LABEL

    best_category = CATEGORY_OTHER_LABEL
    best_score = 0

    for category_name, keywords in AUTO_CATEGORY_RULES.items():
        score = len(tokens.intersection(set(keywords)))
        if score > best_score:
            best_score = score
            best_category = category_name

    return best_category


def build_products_for_categories(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    raw_df = load_table_bytes(file_bytes, file_name)
    mapping = infer_column_mapping(raw_df.columns.tolist())

    rename_map: dict[str, str] = {}
    for canonical_name in ['customer_id', 'product_id', 'product_name']:
        source_name = mapping.get(canonical_name)
        if source_name and source_name in raw_df.columns and canonical_name not in raw_df.columns:
            rename_map[source_name] = canonical_name

    products = raw_df.rename(columns=rename_map).copy()

    if 'customer_id' not in products.columns or 'product_name' not in products.columns:
        return pd.DataFrame(columns=['customer_id', 'product_id', 'product_name', 'product_description', 'category'])

    if 'product_id' not in products.columns:
        products['product_id'] = products['product_name'].astype(str)

    products['customer_id'] = (
        products['customer_id']
        .astype('string')
        .str.replace(r'\.0$', '', regex=True)
        .str.strip()
    )
    products['product_id'] = products['product_id'].astype(str).str.strip()
    products['product_name'] = products['product_name'].fillna('').astype(str).str.strip()
    products['product_description'] = products['product_name']

    products = products[
        products['customer_id'].notna() &
        (products['product_name'] != '')
    ].copy()

    text_series = (
        products['product_name'].fillna('').astype(str)
        + ' '
        + products['product_description'].fillna('').astype(str)
    ).str.strip()

    category_results = text_series.apply(categorize_product)
    products['category'] = category_results.apply(lambda result: result[0])
    products['category_source'] = category_results.apply(lambda result: result[1])

    keep_cols = ['customer_id', 'product_id', 'product_name', 'product_description', 'category']
    return products[keep_cols].drop_duplicates().reset_index(drop=True)


def build_products_for_categories_from_dataframe(
    raw_df: pd.DataFrame,
    manual_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    mapping = infer_column_mapping(raw_df.columns.tolist())

    if manual_mapping:
        for canonical_name, source_name in manual_mapping.items():
            if source_name:
                mapping[canonical_name] = source_name

    rename_map: dict[str, str] = {}
    for canonical_name in ['customer_id', 'product_id', 'product_name']:
        source_name = mapping.get(canonical_name)
        if source_name and source_name in raw_df.columns and source_name != canonical_name:
            rename_map[source_name] = canonical_name

    products = raw_df.rename(columns=rename_map).copy()

    if 'customer_id' not in products.columns or 'product_name' not in products.columns:
        return pd.DataFrame(columns=[
            'customer_id',
            'product_id',
            'product_name',
            'product_description',
            'category',
        ])

    if 'product_id' not in products.columns:
        products['product_id'] = products['product_name'].astype(str)

    products['customer_id'] = (
        products['customer_id']
        .astype('string')
        .str.replace(r'\.0$', '', regex=True)
        .str.strip()
    )
    products['product_id'] = products['product_id'].astype(str).str.strip()
    products['product_name'] = products['product_name'].fillna('').astype(str).str.strip()
    products['product_description'] = products['product_name']

    products = products[
        products['customer_id'].notna()
        & (products['product_name'] != '')
    ].copy()

    text_series = (
        products['product_name'].fillna('').astype(str)
        + ' '
        + products['product_description'].fillna('').astype(str)
    ).str.strip()

    category_results = text_series.apply(categorize_product)
    products['category'] = category_results.apply(lambda result: result[0])
    products['category_source'] = category_results.apply(lambda result: result[1])

    keep_cols = [
        'customer_id',
        'product_id',
        'product_name',
        'product_description',
        'category',
    ]
    return products[keep_cols].drop_duplicates().reset_index(drop=True)


def build_auto_category_profile(products: pd.DataFrame) -> pd.DataFrame:
    if products is None or products.empty or 'customer_id' not in products.columns:
        return pd.DataFrame(columns=[
            'customer_id',
            'dominant_category_auto',
            'top_categories_auto',
            'purchased_categories_count',
        ])

    work = products.copy()
    grouped = work.groupby('customer_id')['category'].apply(list).reset_index()

    def choose_top(items: list[str]) -> str:
        cleaned = [normalize_category_label(item) for item in items]
        cleaned = [item for item in cleaned if item]
        if not cleaned:
            return CATEGORY_OTHER_LABEL

        counts = pd.Series(cleaned).value_counts()
        return ', '.join(counts.index.tolist()[:3])

    def choose_dominant(items: list[str]) -> str:
        cleaned = [normalize_category_label(item) for item in items]
        cleaned = [item for item in cleaned if item]
        if not cleaned:
            return CATEGORY_OTHER_LABEL
        return pd.Series(cleaned).value_counts().index[0]

    def count_specific(items: list[str]) -> int:
        cleaned = {normalize_category_label(item) for item in items}
        cleaned.discard('')
        return len(cleaned)

    grouped['top_categories_auto'] = grouped['category'].apply(choose_top)
    grouped['dominant_category_auto'] = grouped['category'].apply(choose_dominant)
    grouped['purchased_categories_count'] = grouped['category'].apply(count_specific)

    return grouped[[
        'customer_id',
        'dominant_category_auto',
        'top_categories_auto',
        'purchased_categories_count',
    ]]


def build_spend_category_profile(latest_df: pd.DataFrame) -> pd.DataFrame:
    category_cols = [col for col in latest_df.columns if col.startswith('category_spend_')]
    if not category_cols:
        return pd.DataFrame(columns=['customer_id', 'dominant_category_spend', 'top_categories_spend'])

    work = latest_df[['customer_id'] + category_cols].copy()
    work[category_cols] = work[category_cols].fillna(0)

    def top_categories(row: pd.Series) -> str:
        pairs: list[tuple[str, float]] = []
        for col in category_cols:
            name = normalize_category_label(col.replace('category_spend_', ''))
            if name:
                pairs.append((name, float(row[col])))

        pairs = sorted(pairs, key=lambda item: item[1], reverse=True)
        selected = [name for name, value in pairs if value > 0][:3]
        if not selected:
            return CATEGORY_OTHER_LABEL
        return ', '.join(selected)

    def dominant_category(row: pd.Series) -> str:
        pairs: list[tuple[str, float]] = []
        for col in category_cols:
            name = normalize_category_label(col.replace('category_spend_', ''))
            if name:
                pairs.append((name, float(row[col])))

        pairs = sorted(pairs, key=lambda item: item[1], reverse=True)
        for name, value in pairs:
            if value > 0:
                return name
        return CATEGORY_OTHER_LABEL

    work['top_categories_spend'] = work.apply(top_categories, axis=1)
    work['dominant_category_spend'] = work.apply(dominant_category, axis=1)
    return work[['customer_id', 'dominant_category_spend', 'top_categories_spend']]


def choose_best_category(row: pd.Series) -> str:
    candidates = [
        row.get('dominant_category_spend', ''),
        *split_categories(row.get('top_categories_spend', '')),
        row.get('dominant_category_auto', ''),
        *split_categories(row.get('top_categories_auto', '')),
    ]

    for value in candidates:
        cleaned = normalize_category_label(value)
        if cleaned:
            return cleaned

    return CATEGORY_OTHER_LABEL


def choose_best_top_categories(row: pd.Series) -> str:
    values: list[str] = []
    values.extend(split_categories(row.get('top_categories_spend', '')))
    values.extend(split_categories(row.get('top_categories_auto', '')))
    values = unique_preserve_order(values)

    if values:
        return ', '.join(values[:3])

    dominant = choose_best_category(row)
    return dominant


def choose_category_source(row: pd.Series) -> str:
    spend = normalize_category_label(row.get('dominant_category_spend', ''))
    auto = normalize_category_label(row.get('dominant_category_auto', ''))

    if spend:
        return 'category_spend_*'
    if auto:
        return 'auto rules'
    return 'fallback'


def build_customer_category_table(latest_df: pd.DataFrame, auto_profile: pd.DataFrame) -> pd.DataFrame:
    base = latest_df[['customer_id']].drop_duplicates().copy()
    spend_profile = build_spend_category_profile(latest_df)

    result = base.merge(spend_profile, on='customer_id', how='left')
    result = result.merge(auto_profile, on='customer_id', how='left')

    result['dominant_category_display'] = result.apply(choose_best_category, axis=1)
    result['top_categories_display'] = result.apply(choose_best_top_categories, axis=1)
    result['category_source'] = result.apply(choose_category_source, axis=1)
    result['purchased_categories_count'] = pd.to_numeric(
        result.get('purchased_categories_count', 0),
        errors='coerce'
    ).fillna(0).astype(int)

    return result[[
        'customer_id',
        'dominant_category_display',
        'top_categories_display',
        'purchased_categories_count',
        'category_source',
    ]]


def auto_recommended_action(row: pd.Series) -> str:
    category_name = format_category_output(row.get('campaign_category', row.get('dominant_category_display', '')))
    risk = clean_text_value(row.get('risk_class', ''), '')
    discount = auto_discount_pct(row)

    if risk == 'High':
        return f'Точкова знижка {discount}% + персональна рекомендація по категорії {category_name}'
    if risk == 'Medium':
        return f'Знижка {discount}% + нагадування і добірка товарів по категорії {category_name}'
    return f'Знижка {discount}% + контентна комунікація по категорії {category_name}'


def _risk_percent(row: pd.Series) -> float:
    """Повертає ризик клієнта у відсотках (0–100). Якщо значення немає — 0.0."""
    try:
        val = float(row.get('churn_probability_percent', 0) or 0)
    except (TypeError, ValueError):
        val = 0.0
    return val


def auto_recommended_channel(row: pd.Series) -> str:
    risk_pct = _risk_percent(row)

    if risk_pct >= 70:
        return 'Push + SMS'
    # 60–69%, 30–59% та <30% — всі через SMS
    return 'SMS'


def auto_discount_pct(row: pd.Series) -> int:
    """Розмір знижки (%) залежно від ризику відтоку у відсотках."""
    risk_pct = _risk_percent(row)

    if risk_pct >= 70:
        return 25
    if risk_pct >= 60:
        return 20
    if risk_pct >= 30:
        return 15
    return 5


def build_manual_action_text(offer_type: str, category_name: str) -> str:
    category_name = format_category_output(category_name)

    if offer_type == 'Discount':
        return f'Знижка на категорію {category_name}'
    if offer_type == 'Bonus points':
        return f'Бонусні бали за покупку в категорії {category_name}'
    if offer_type == 'Personal recommendation':
        return f'Персональна рекомендація товарів у категорії {category_name}'
    if offer_type == 'Reminder':
        return f'Нагадування про категорію {category_name}'
    if offer_type == 'Bundle offer':
        return f'Комплектна пропозиція по категорії {category_name}'

    return auto_recommended_action(pd.Series({'campaign_category': category_name, 'risk_class': ''}))


def build_campaign_table(
    audience: pd.DataFrame,
    campaign_name: str,
    offer_type: str,
    channel_mode: str,
    category_mode: str,
    manual_category: str,
) -> pd.DataFrame:
    if audience.empty:
        return pd.DataFrame(columns=[
            'campaign_name',
            'customer_id',
            'churn_probability_percent',
            'risk_class',
            'rfm_segment',
            'customer_cluster_name',
            'campaign_category',
            'recommended_action',
            'recommended_channel',
            'top_categories_display',
        ])

    work = audience.copy()

    if category_mode == 'Manual' and clean_text_value(manual_category, '') != '':
        work['campaign_category'] = manual_category.strip()
    elif 'campaign_category' in work.columns:
        work['campaign_category'] = work['campaign_category'].apply(format_category_output)
    else:
        work['campaign_category'] = work['dominant_category_display'].apply(format_category_output)

    if offer_type == 'Auto':
        work['recommended_action'] = work.apply(auto_recommended_action, axis=1)
    else:
        work['recommended_action'] = work['campaign_category'].apply(
            lambda category_name: build_manual_action_text(offer_type, category_name)
        )

    if channel_mode == 'Auto':
        work['recommended_channel'] = work.apply(auto_recommended_channel, axis=1)
    else:
        work['recommended_channel'] = channel_mode

    work['campaign_name'] = campaign_name

    columns = [
        'campaign_name',
        'customer_id',
        'churn_probability_percent',
        'recommended_channel',
        'risk_class',
        'rfm_segment',
        'customer_cluster_name',
        'campaign_category',
        'recommended_action',
        'top_categories_display',
    ]
    columns = [col for col in columns if col in work.columns]

    return work[columns].sort_values(by='churn_probability_percent', ascending=False).reset_index(drop=True)


def build_feedback_campaign_table(audience: pd.DataFrame, campaign_name: str) -> pd.DataFrame:
    """Побудувати таблицю кампанії, ВИКОРИСТОВУЮЧИ рекомендації feedback-моделі.

    На відміну від build_campaign_table, тут НЕ перетираємо recommended_action /
    recommended_channel / discount_pct, які вже поставив attach_feedback_recommendations().
    Це і є суть feedback-aware кампанії: знижка масштабується від P(response),
    канал залежить від ризику І від ймовірності реакції, текст оферу включає
    P(response) та dynamic discount %.
    """
    if audience is None or audience.empty:
        return pd.DataFrame(columns=[
            'campaign_name', 'customer_id', 'churn_probability_percent', 'risk_class',
            'campaign_category', 'discount_pct', 'response_prob_pct',
            'offer_type', 'recommended_action', 'recommended_channel',
        ])

    work = audience.copy()
    work['campaign_name'] = campaign_name

    # recommended_message з feedback_model.build_offer_vector() містить готовий
    # текст оферу зі знижкою та категорією — він і стає recommended_action.
    if 'recommended_message' in work.columns:
        work['recommended_action'] = work['recommended_message']

    # campaign_category вже виставлена у викликача (= best_category).
    # Якщо ні — fallback на best_category або dominant_category_display.
    if 'campaign_category' not in work.columns:
        if 'best_category' in work.columns:
            work['campaign_category'] = work['best_category']
        elif 'dominant_category_display' in work.columns:
            work['campaign_category'] = work['dominant_category_display']
        else:
            work['campaign_category'] = 'General Merchandise'

    columns = [
        'campaign_name',
        'customer_id',
        'churn_probability_percent',
        'risk_class',
        'rfm_segment',
        'customer_cluster_name',
        'campaign_category',
        'top_categories_ranked',
        'response_prob_pct',
        'discount_pct',
        'offer_type',
        'recommended_action',
        'recommended_channel',
        'top_categories_display',
    ]
    columns = [col for col in columns if col in work.columns]

    return work[columns].sort_values(by='churn_probability_percent', ascending=False).reset_index(drop=True)


@st.cache_data(
    show_spinner=False,  # динамічний спінер виставляється у місці виклику (див. main())
    hash_funcs={
        # Hash DataFrames by shape + column names + first/last rows checksum
        'pandas.core.frame.DataFrame': lambda df: (
            df.shape,
            tuple(df.columns),
            int(df.iloc[0].astype(str).str.len().sum()) if len(df) > 0 else 0,
            int(df.iloc[-1].astype(str).str.len().sum()) if len(df) > 1 else 0,
        ),
    },
)
def build_app_state(
    raw_df: pd.DataFrame,
    source_name: str,
    manual_mapping: dict[str, str] | None = None,
    _saved_churn_artifacts=None,
):
    df, mapping, notes = load_and_prepare_transactions_from_dataframe(
        raw_df,
        manual_mapping=manual_mapping,
    )
    notes.append(f'Джерело даних: {source_name}')

    required_columns = ['customer_id', 'transaction_date']
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            'Не вдалося привести дані до потрібного формату. '
            f'Не вистачає колонок: {missing}'
        )

    products = build_products_for_categories_from_dataframe(raw_df, manual_mapping)
    auto_profile = build_auto_category_profile(products)

    word2vec_model = train_word2vec(products)
    product_clusters = cluster_products(products, word2vec_model)
    semantic_summary = semantic_cluster_summary(product_clusters)

    featured = add_base_features(df)
    horizon_days = 90

    if _saved_churn_artifacts is None:
        train_featured = build_forward_churn_dataset(featured, horizon_days=horizon_days)

        if len(train_featured) == 0:
            raise ValueError('Не вдалося побудувати train set для forward-looking churn. Замало даних після відсікання horizon window.')

        if train_featured['churn'].nunique() < 2:
            raise ValueError('У forward-looking target вийшов лише один клас churn. Спробуйте інший horizon_days.')

        X_train, y, groups, _ = build_training_matrices(train_featured)
        churn_service = ChurnModelService(use_xgboost=True)
        churn_artifacts = churn_service.train(X_train, y, groups)
    else:
        churn_artifacts = _saved_churn_artifacts
        notes.append('Використано збережену churn-модель.')

    featured['is_currently_active'] = featured['days_since_last_purchase'] <= horizon_days
    X_all, _, _, _ = build_training_matrices(featured.assign(churn=0))
    predicted = attach_predictions(featured, X_all, churn_artifacts)
    predicted['is_currently_active'] = featured['is_currently_active'].values

    rfm = build_rfm_table(featured)
    predicted = predicted.merge(
        rfm[['customer_id', 'rfm_segment', 'RFM_score']],
        on='customer_id',
        how='left',
    )

    customer_clusters = build_customer_clusters(predicted)
    if len(customer_clusters) > 0:
        predicted = predicted.merge(customer_clusters, on='customer_id', how='left')
    else:
        predicted['customer_cluster_name'] = 'Сегмент не визначено'

    latest_customers = get_latest_rows(predicted).copy()
    latest_customers['feature_row_index'] = latest_customers.index
    latest_customers = latest_customers.reset_index(drop=True)

    category_table = build_customer_category_table(latest_customers, auto_profile)
    latest_customers = latest_customers.merge(category_table, on='customer_id', how='left')

    latest_customers['dominant_category_display'] = latest_customers['dominant_category_display'].fillna(CATEGORY_OTHER_LABEL)
    latest_customers['top_categories_display'] = latest_customers['top_categories_display'].fillna(CATEGORY_OTHER_LABEL)
    latest_customers['customer_cluster_name'] = latest_customers['customer_cluster_name'].fillna('Сегмент не визначено')

    importance = feature_importance_table(churn_artifacts)

    risk_order = ['High', 'Medium', 'Low']
    risk_counts = (
        latest_customers['risk_class']
        .value_counts()
        .reindex(risk_order)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    risk_counts.columns = ['risk_class', 'customers']

    rfm_counts = rfm['rfm_segment'].value_counts().reset_index()
    rfm_counts.columns = ['rfm_segment', 'customers']

    active_customers = latest_customers[latest_customers['is_currently_active'] == True].copy()
    top_risk_active = active_customers[
        ['customer_id', 'churn_probability_percent', 'risk_class', 'dominant_category_display']
    ].sort_values(by='churn_probability_percent', ascending=False).head(10)

    category_counts = (
        latest_customers['dominant_category_display']
        .value_counts()
        .reset_index()
    )
    category_counts.columns = ['category', 'customers']

    specific_category_counts = category_counts[category_counts['category'] != CATEGORY_OTHER_LABEL].copy()

    category_coverage_pct = round(
        100 * (latest_customers['dominant_category_display'] != CATEGORY_OTHER_LABEL).mean(),
        1,
    )

    return {
        'notes': notes,
        'mapping': mapping,
        'products': products,
        'word2vec_model': word2vec_model,
        'product_clusters': product_clusters,
        'semantic_summary': semantic_summary,
        'featured': featured,
        'predicted': predicted,
        'latest_customers': latest_customers,
        'rfm': rfm,
        'category_table': latest_customers[[
            'customer_id',
            'dominant_category_display',
            'top_categories_display',
            'purchased_categories_count',
            'category_source',
            'customer_cluster_name',
        ]].sort_values(by='customer_id').reset_index(drop=True),
        'customer_clusters': customer_clusters,
        'churn_artifacts': churn_artifacts,
        'feature_importance': importance,
        'risk_counts': risk_counts,
        'rfm_counts': rfm_counts,
        'category_counts': category_counts,
        'specific_category_counts': specific_category_counts,
        'category_coverage_pct': category_coverage_pct,
        'top_risk_active': top_risk_active,
        'X_all': X_all,
    }


EDA_STOP_WORDS = {
    'and', 'of', 'the', 'a', 'an', 'to', 'for', 'with', 'in', 'on', 'at', 'by',
    'set', 'pack', 'assorted', 'design', 'style', 'single', 'size', 'small',
    'large', 'medium', 'mini', 'jumbo', 'classic', 'retro', 'vintage',
    'red', 'pink', 'white', 'blue', 'green', 'black', 'silver', 'gold',
    'new', 'hot', 'pcs', 'piece', 'pieces', 'шт', 'набір', 'товар', 'для', 'та', 'і'
}


def finish_chart(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def dark_table(data, hide_index: bool = True, height: int | None = 360) -> None:
    """Рендерить DataFrame у нашому темному/світлому стилі.

    height=None або 0 → жодного max-height (таблиця розгортається повністю,
    сторінка прокручується). Корисно для фінальних таблиць (кампанія, rescue queue).
    height=int    → обмежена висота з внутрішнім скролом (для preview/малих таблиць).
    """
    df = pd.DataFrame(data)

    if df.empty:
        st.info('Таблиця порожня.')
        return

    table_html = df.to_html(
        index=not hide_index,
        escape=True,
    )

    height_style = f'max-height:{height}px;' if height else ''
    st.markdown(
        f'''
        <div class="dark-table-box" style="{height_style}">
            {table_html}
        </div>
        ''',
        unsafe_allow_html=True,
    )


def plot_horizontal_counts(counts: pd.Series, title: str, xlabel: str, ylabel: str) -> None:
    counts = counts.dropna()
    counts = counts[counts > 0]

    if counts.empty:
        st.info(f'Немає даних для графіка: {title}')
        return

    plot_data = counts.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_data.index.astype(str), plot_data.values, color=CHART_COLORS[0])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='x', alpha=0.3)

    for i, value in enumerate(plot_data.values):
        ax.text(value, i, f' {value:.0f}', va='center')

    fig.tight_layout()
    finish_chart(fig)


def extract_product_words(products: pd.DataFrame, featured: pd.DataFrame) -> Counter:
    counter: Counter = Counter()
    frames = []

    if products is not None and not products.empty:
        frames.append(products)
    if featured is not None and not featured.empty:
        frames.append(featured)

    for frame in frames:
        text_columns = [
            col for col in ['product_name', 'product_description', 'product_text', 'description']
            if col in frame.columns
        ]

        for column in text_columns:
            for text in frame[column].dropna().astype(str):
                tokens = re.findall(r'[a-zA-Zа-яА-ЯіІїЇєЄ0-9]+', text.lower())
                for token in tokens:
                    if len(token) <= 2:
                        continue
                    if token.isdigit():
                        continue
                    if token in EDA_STOP_WORDS:
                        continue
                    counter[token.upper()] += 1

    return counter


def render_wordcloud_block(products: pd.DataFrame, featured: pd.DataFrame) -> None:
    st.markdown('### WordCloud за назвами товарів')

    counter = extract_product_words(products, featured)
    if not counter:
        st.info('Немає текстових назв товарів для WordCloud. Перевір маппінг колонки product_name / description.')
        return

    if WORDCLOUD_AVAILABLE:
        wordcloud = WordCloud(
            width=1100,
            height=480,
            background_color='#0b0d14' if st.session_state.get('dark_mode', False) else '#ffffff',
            colormap='cool' if st.session_state.get('dark_mode', False) else 'Blues',
            max_words=160,
            random_state=42,
            collocations=False,
        ).generate_from_frequencies(counter)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title('Найпопулярніші слова в назвах товарів')
        ax.axis('off')
        fig.tight_layout()
        finish_chart(fig)
        return

    st.warning('Пакет wordcloud не встановлений. Показую fallback: топ слів. Для справжньої хмари: pip install wordcloud')
    top_words = pd.Series(dict(counter.most_common(25)))
    plot_horizontal_counts(top_words, 'Топ слів у назвах товарів', 'Кількість згадувань', 'Слово')


def render_word2vec_block(product_clusters: pd.DataFrame, semantic_summary: pd.DataFrame) -> None:
    st.markdown('### Word2Vec: семантичні кластери товарів')

    if not WORD2VEC_AVAILABLE:
        st.warning('Word2Vec не запустився, бо не встановлено gensim. Встанови: pip install gensim')
        return

    if product_clusters is None or product_clusters.empty:
        st.info('Недостатньо текстових назв товарів для Word2Vec. Потрібно хоча б кілька непорожніх назв товарів.')
        return

    if {'pca_x', 'pca_y', 'semantic_cluster_name'}.issubset(product_clusters.columns):
        plot_df = product_clusters.copy()
        plot_df['semantic_cluster_name'] = plot_df['semantic_cluster_name'].fillna('Інше').astype(str)

        fig, ax = plt.subplots(figsize=(9, 6))
        for cluster_name, part in plot_df.groupby('semantic_cluster_name'):
            ax.scatter(part['pca_x'], part['pca_y'], label=cluster_name, alpha=0.65, s=28)

        ax.set_title('Word2Vec + PCA: близькі товари розташовані поруч')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        finish_chart(fig)

    if semantic_summary is not None and not semantic_summary.empty:
        st.markdown('### Таблиця семантичних кластерів')
        dark_table(semantic_summary, hide_index=True, height=360)
        download_dataframe_button(semantic_summary, 'word2vec_semantic_clusters.csv', 'Завантажити Word2Vec-кластери')


def render_eda_tab(state: dict) -> None:
    st.markdown('### EDA-графіки')

    featured = state['featured']
    latest_customers = state['latest_customers']
    products = state['products']

    left, right = st.columns(2)

    with left:
        st.markdown('### Дохід по місяцях')
        if {'transaction_date', 'total_sales'}.issubset(featured.columns):
            work = featured[['transaction_date', 'total_sales']].copy()
            work['transaction_date'] = pd.to_datetime(work['transaction_date'], errors='coerce')
            work['total_sales'] = pd.to_numeric(work['total_sales'], errors='coerce').fillna(0)
            work = work.dropna(subset=['transaction_date'])

            if not work.empty:
                monthly = work.groupby(work['transaction_date'].dt.to_period('M'))['total_sales'].sum().sort_index()
                monthly.index = monthly.index.astype(str)

                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(monthly.index, monthly.values, marker='o')
                ax.set_title('Дохід по місяцях')
                ax.set_xlabel('Місяць')
                ax.set_ylabel('Дохід')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(alpha=0.3)
                fig.tight_layout()
                finish_chart(fig)
            else:
                st.info('Немає коректних дат для графіка доходу.')
        else:
            st.info('Потрібні колонки transaction_date і total_sales.')

    with right:
        st.markdown('### Кількість замовлень по місяцях')
        if {'transaction_date', 'transaction_id'}.issubset(featured.columns):
            work = featured[['transaction_date', 'transaction_id']].copy()
            work['transaction_date'] = pd.to_datetime(work['transaction_date'], errors='coerce')
            work = work.dropna(subset=['transaction_date'])

            if not work.empty:
                orders = work.groupby(work['transaction_date'].dt.to_period('M'))['transaction_id'].nunique().sort_index()
                orders.index = orders.index.astype(str)

                fig, ax = plt.subplots(figsize=(9, 5))
                ax.bar(orders.index, orders.values)
                ax.set_title('Кількість замовлень по місяцях')
                ax.set_xlabel('Місяць')
                ax.set_ylabel('Замовлення')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)
                fig.tight_layout()
                finish_chart(fig)
            else:
                st.info('Немає коректних дат для графіка замовлень.')
        else:
            st.info('Потрібні колонки transaction_date і transaction_id.')

    left, right = st.columns(2)
    with left:
        plot_horizontal_counts(
            state['rfm_counts'].set_index('rfm_segment')['customers'],
            'RFM-сегменти',
            'Кількість клієнтів',
            'Сегмент',
        )
    with right:
        plot_horizontal_counts(
            state['risk_counts'].set_index('risk_class')['customers'],
            'Розподіл ризику відтоку',
            'Кількість клієнтів',
            'Клас ризику',
        )

    left, right = st.columns(2)
    with left:
        chart_df = state['specific_category_counts'].copy()
        if len(chart_df) == 0:
            chart_df = state['category_counts'].copy()
        if len(chart_df) > 0:
            plot_horizontal_counts(
                chart_df.head(10).set_index('category')['customers'],
                'Топ категорій товарів',
                'Кількість клієнтів',
                'Категорія',
            )
        else:
            st.info('Немає категорій для графіка.')
    with right:
        if 'customer_cluster_name' in latest_customers.columns:
            plot_horizontal_counts(
                latest_customers['customer_cluster_name'].value_counts(),
                'K-Means кластери клієнтів',
                'Кількість клієнтів',
                'Кластер',
            )
        else:
            st.info('Немає customer_cluster_name для графіка кластерів.')

    left, right = st.columns(2)
    with left:
        if 'churn_probability_percent' in latest_customers.columns:
            st.markdown('### Розподіл ймовірності відтоку')
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(pd.to_numeric(latest_customers['churn_probability_percent'], errors='coerce').dropna(), bins=20)
            ax.set_title('Розподіл churn probability')
            ax.set_xlabel('Ймовірність відтоку, %')
            ax.set_ylabel('Кількість клієнтів')
            ax.grid(axis='y', alpha=0.3)
            fig.tight_layout()
            finish_chart(fig)
    with right:
        if state['feature_importance'] is not None and not state['feature_importance'].empty:
            importance = state['feature_importance'].head(10).copy()
            first_col = importance.columns[0]
            last_col = importance.columns[-1]
            plot_horizontal_counts(
                importance.set_index(first_col)[last_col],
                'Топ-10 факторів відтоку',
                'Важливість',
                'Ознака',
            )
        else:
            st.info('Немає feature importance для графіка.')

    render_wordcloud_block(products, featured)
    render_word2vec_block(state.get('product_clusters'), state.get('semantic_summary'))

def read_local_input_file(file_path: str) -> tuple[bytes, str]:
    path = Path(file_path).expanduser()

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f'Файл не знайдено: {path}')

    if path.suffix.lower() not in {'.csv', '.xlsx', '.xls'}:
        raise ValueError('Підтримуються лише файли CSV, XLSX або XLS.')

    return path.read_bytes(), path.name

@st.cache_data(show_spinner=False)
def get_cached_dataframe(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    return load_table_bytes(file_bytes, file_name)
def main():
    # ── Theme toggle ──────────────────────────────────────────────────────────
    # Базова тема після першого запуску — темна.
    # Version flag потрібен, щоб після оновлення файлу Streamlit один раз
    # виставив темну тему навіть якщо у старій сесії залишився світлий стан.
    # Ініціалізація базового стану теми при першому запуску
    _dark = True
    _apply_mpl_theme(_dark)
    # ── CSS (light / dark) ────────────────────────────────────────────────────
    if _dark:
     _css = (
        '<style>\n'
        '*, *::before, *::after { box-sizing: border-box; }\n'
        '[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{ background:#0b0d14!important; border-bottom:1px solid rgba(255,255,255,.05)!important; }\n'
        'html,body{color-scheme:dark!important}\n'
        'html,body,.main,.block-container,[data-testid="stAppViewContainer"],[data-testid="stApp"],[data-testid="stMainBlockContainer"],[data-testid="stVerticalBlock"],[data-testid="stBottom"],[class*="appview"],[class*="main"]{ background-color:#0b0d14!important; color:#dde3f5!important; }\n'
        '[data-testid="stSidebar"],#stSidebar,section[data-testid="stSidebar"]{ background-color:#0f1219!important; border-right:1px solid rgba(255,255,255,.06)!important; }\n'
        '[data-testid="stSidebar"] *, [data-testid="stSidebar"] label{ color:#c8cfe8!important; }\n'
        'h1{color:#7c9fe6!important;font-weight:800!important}\n'
        'h2{color:#a78bfa!important;font-weight:700!important}\n'
        'h3{color:#c8d3f5!important;font-weight:600!important}\n'
        'h4,h5,h6,p,span,label,div{color:#dde3f5}\n'
        '[data-testid="stTabs"] [role="tablist"]{background:#13151f!important;border-radius:12px!important;padding:4px!important;border:1px solid rgba(255,255,255,.07)!important}\n'
        '[data-testid="stTabs"] button[role="tab"]{background:transparent!important;color:#8891aa!important;border-radius:8px!important;font-weight:500!important;font-size:13px!important;border:none!important}\n'
        '[data-testid="stTabs"] button[role="tab"]:hover{color:#dde3f5!important;background:rgba(124,159,230,.1)!important}\n'
        '[data-testid="stTabs"] button[aria-selected="true"]{background:rgba(124,159,230,.18)!important;color:#7c9fe6!important;font-weight:700!important}\n'
        '[data-testid="stMetric"]{background:#13151f!important;border-radius:12px!important;padding:14px 16px!important;border:1px solid rgba(255,255,255,.07)!important}\n'
        '[data-testid="stMetricLabel"]{color:#8891aa!important;font-size:11px!important;text-transform:uppercase;letter-spacing:.8px}\n'
        '[data-testid="stMetricValue"]{color:#7c9fe6!important;font-weight:800!important}\n'
        '[data-testid="stButton"]>button,[data-testid="stDownloadButton"]>button{background:linear-gradient(135deg,#1e2a48,#2a1e48)!important;color:#a0b4e8!important;border:1px solid rgba(124,159,230,.25)!important;border-radius:9px!important;font-weight:600!important}\n'
        '[data-testid="stButton"]>button:hover,[data-testid="stDownloadButton"]>button:hover{background:linear-gradient(135deg,#263560,#36265a)!important;border-color:rgba(124,159,230,.5)!important;color:#dde3f5!important;transform:translateY(-1px)!important;box-shadow:0 4px 16px rgba(124,159,230,.15)!important}\n'
        '[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{background:#13151f!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:9px!important;color:#dde3f5!important}\n'
        '[data-testid="stSelectbox"]>div>div,[data-testid="stSelectbox"] [role="listbox"],[data-testid="stSelectbox"] [role="option"],[data-baseweb="select"]>div,[data-baseweb="popover"] ul,[data-baseweb="popover"],[data-baseweb="menu"]{background:#13151f!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:9px!important;color:#dde3f5!important}\n'
        '[data-baseweb="option"]:hover,[data-baseweb="option"][aria-selected="true"]{background:rgba(124,159,230,.15)!important;color:#7c9fe6!important}\n'
        '[data-testid="stRadio"] label,[data-testid="stRadio"] div{color:#c8cfe8!important}\n'
        '[data-testid="stFileUploader"],[data-testid="stFileUploader"] section,[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],[data-testid="stFileUploadDropzone"]{background:#0f1219!important;border:1.5px dashed rgba(124,159,230,.3)!important;border-radius:12px!important;color:#8891aa!important}\n'
        '[data-testid="stFileUploader"] *{color:#8891aa!important;background:transparent!important}\n'
        '[data-testid="stAlert"],[data-baseweb="notification"]{border-radius:10px!important;border-left-width:3px!important;background:rgba(19,21,31,.95)!important}\n'
        '[data-testid="stExpander"]{background:#13151f!important;border:1px solid rgba(255,255,255,.07)!important;border-radius:10px!important}\n'
        '[data-testid="stExpander"] summary{color:#8891aa!important;font-size:13px!important}\n'
        '[data-testid="stDataFrame"],[data-testid="stDataFrame"]>div{background:#13151f!important;border:1px solid rgba(255,255,255,.07)!important;border-radius:10px!important;overflow:hidden!important}\n'
        '[data-testid="stDataFrame"],[data-testid="stDataFrame"] *{color:#ffffff!important}\n'
        '[data-testid="stDataFrame"] iframe{background:#13151f!important;color-scheme:dark!important;filter:none!important}\n'
        '.dvn-scroller,.gdg-cell,.gdg-header-cell,.glide-data-grid-svg{background:#13151f!important;color:#ffffff!important;fill:#ffffff!important}\n'
        '.dvn-stack{background:#13151f!important;color:#ffffff!important}\n'
        '.dvn-stack *{color:#ffffff!important;fill:#ffffff!important}\n'
        'canvas{color-scheme:dark!important}\n'
        '[data-testid="stArrowVegaLiteChart"] canvas,[data-testid="stArrowVegaLiteChart"] svg{background:#13151f!important;border-radius:10px!important}\n'
        '[data-testid="stBarChart"],[data-testid="stLineChart"],[data-testid="stAreaChart"]{background:#13151f!important;border-radius:10px!important;padding:8px!important;border:1px solid rgba(255,255,255,.06)!important}\n'
        '[data-testid="stImage"] img{border-radius:10px!important;border:1px solid rgba(255,255,255,.06)!important}\n'
        '[data-testid="stSlider"] [role="slider"]{background:#7c9fe6!important;border:2px solid #1a2040!important;box-shadow:0 0 8px rgba(124,159,230,.5)!important;width:22px!important;height:22px!important}\n'
        '[data-testid="stSlider"] [data-testid="stTickBar"] *{font-size:13px!important;font-weight:800!important}\n'
        '[data-testid="stSlider"] [data-testid="stTickBar"] *:first-child{color:#f87171!important}\n'
        '[data-testid="stSlider"] [data-testid="stTickBar"] *:last-child{color:#34d399!important}\n'
        '.whatif-scale{display:flex!important;justify-content:space-between!important;align-items:center!important;margin-top:6px!important;padding:0 4px!important}\n'
        '.whatif-scale span{font-size:12px!important;font-weight:800!important}\n'
        '.whatif-scale .whatif-min{color:#f87171!important}\n'
        '.whatif-scale .whatif-zero{color:#4a5580!important}\n'
        '.whatif-scale .whatif-max{color:#34d399!important}\n'
        '.whatif-scale .whatif-sep{font-size:11px!important;color:#2a2e44!important;font-weight:700!important}\n'
        '[data-testid="stCaptionContainer"] p{color:#4a5580!important}\n'
        '[data-testid="stpyplot"]>div{background:#13151f!important;border-radius:12px!important;padding:8px!important;border:1px solid rgba(255,255,255,.06)!important}\n'
        '::-webkit-scrollbar{width:6px;height:6px}\n'
        '::-webkit-scrollbar-track{background:#0b0d14}\n'
        '::-webkit-scrollbar-thumb{background:#2a2e44;border-radius:3px}\n'
        '::-webkit-scrollbar-thumb:hover{background:#3a3e5a}\n'
        'hr{border-color:rgba(255,255,255,.07)!important}\n'
        '[data-testid="stDataFrame"] *{color:#ffffff!important;fill:#ffffff!important}\n'
        '[data-testid="stDataFrame"] [role="columnheader"]{background:#1a1e2e!important;color:#7c9fe6!important;font-weight:700!important;border-bottom:1px solid rgba(255,255,255,.1)!important}\n'
        '[data-testid="stDataFrame"] [role="gridcell"]{background:#13151f!important;border-color:rgba(255,255,255,.05)!important}\n'
        '[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"]{background:#1a1e2e!important}\n'
        '[data-testid="stDataFrame"] [role="row"]:nth-child(even) [role="gridcell"]{background:#0f1219!important}\n'
        '[data-testid="stArrowVegaLiteChart"]>div,[data-testid="stVegaLiteChart"]>div{background:#13151f!important;border-radius:12px!important;border:1px solid rgba(255,255,255,.06)!important;padding:8px!important}\n'
        '[data-testid="stImage"] img{border-radius:12px!important}\n'
        'table,thead,tbody,tr,th,td{background:#13151f!important;color:#dde3f5!important;border-color:rgba(255,255,255,.07)!important}\n'
        'th{color:#7c9fe6!important;font-weight:700!important;border-bottom:1px solid rgba(255,255,255,.12)!important}\n'
        'small,caption,[data-testid="stCaptionContainer"] p{color:#4a5580!important}\n'
        '[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,[data-testid="stMarkdownContainer"] span{color:#dde3f5!important}\n'
        '[data-testid="stMarkdownContainer"] b,[data-testid="stMarkdownContainer"] strong{color:#ffffff!important}\n'
        '.dark-table-box{width:100%!important;overflow:auto!important;background:#13151f!important;border:1px solid rgba(255,255,255,.12)!important;border-radius:10px!important;margin:8px 0 16px 0!important}\n'
        '.dark-table-box table{width:100%!important;border-collapse:collapse!important;background:#13151f!important;color:#ffffff!important;font-size:13px!important}\n'
        '.dark-table-box thead th{position:sticky!important;top:0!important;z-index:2!important;background:#1a1e2e!important;color:#ffffff!important;font-weight:700!important;border-bottom:1px solid rgba(255,255,255,.18)!important;padding:8px 10px!important;text-align:left!important;white-space:nowrap!important}\n'
        '.dark-table-box tbody td{background:#13151f!important;color:#ffffff!important;border-bottom:1px solid rgba(255,255,255,.07)!important;padding:8px 10px!important;text-align:left!important;white-space:nowrap!important}\n'
        '.dark-table-box tbody tr:nth-child(even) td{background:#0f1219!important}\n'
        '.dark-table-box tbody tr:hover td{background:#1a1e2e!important}\n'
        '</style>'
     )
   
    # Додатковий фікс для компонентів Streamlit, які не повністю
    # перекриваються основним CSS: expander, spinner/status, selectbox popover,
    # file uploader, alerts. Це прибирає чорні артефакти у світлій темі.
    _theme_bg = '#13151f' if _dark else '#ffffff'
    _theme_bg_alt = '#0f1219' if _dark else '#f7f7f7'
    _theme_header = '#1a1e2e' if _dark else '#eeeeee'
    _theme_border = 'rgba(255,255,255,.10)' if _dark else '#d9d9d9'
    _theme_text = '#dde3f5' if _dark else '#000000'
    _theme_hover = '#1a1e2e' if _dark else '#f2f2f2'
    _theme_color_scheme = 'dark' if _dark else 'light'

    _theme_fix_css = f"""
        html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
        [data-testid="stMain"], [data-testid="stMainBlockContainer"],
        [data-testid="stVerticalBlock"], [data-testid="stBottom"] {{
            background: {_theme_bg}!important;
            color: {_theme_text}!important;
            color-scheme: {_theme_color_scheme}!important;
        }}

        [data-testid="stHeader"], [data-testid="stToolbar"],
        [data-testid="stDecoration"], [data-testid="stStatusWidget"],
        [data-testid="stStatusWidget"] div, [data-testid="stStatusWidget"] span,
        [data-testid="stSpinner"], [data-testid="stSpinner"] div,
        [data-testid="stSpinner"] span, [data-testid="stSpinner"] p {{
            background: {_theme_bg}!important;
            color: {_theme_text}!important;
            fill: {_theme_text}!important;
            color-scheme: {_theme_color_scheme}!important;
        }}

        [data-testid="stExpander"], [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
        [data-testid="stExpander"] [data-testid="stVerticalBlock"],
        [data-testid="stExpanderDetails"], [data-testid="stExpanderDetails"] div {{
            background: {_theme_bg}!important;
            color: {_theme_text}!important;
            border-color: {_theme_border}!important;
            color-scheme: {_theme_color_scheme}!important;
        }}

        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] svg,
        [data-testid="stExpander"] label,
        [data-testid="stExpander"] div,
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] span {{
            color: {_theme_text}!important;
            fill: {_theme_text}!important;
        }}

        [data-testid="stSelectbox"], [data-testid="stSelectbox"] div,
        [data-testid="stSelectbox"] label,
        [data-baseweb="select"], [data-baseweb="select"] div,
        [data-baseweb="popover"], [data-baseweb="popover"] div,
        [data-baseweb="popover"] ul, [data-baseweb="popover"] li,
        [data-baseweb="menu"], [data-baseweb="menu"] div,
        [data-baseweb="option"] {{
            background: {_theme_bg}!important;
            color: {_theme_text}!important;
            border-color: {_theme_border}!important;
            color-scheme: {_theme_color_scheme}!important;
        }}

        [data-baseweb="option"]:hover,
        [data-baseweb="option"][aria-selected="true"] {{
            background: {_theme_hover}!important;
            color: {_theme_text}!important;
        }}

        [data-testid="stFileUploader"], [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"],
        [data-testid="stFileUploader"] div,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span {{
            background: {_theme_bg}!important;
            color: {_theme_text}!important;
            border-color: {_theme_border}!important;
            color-scheme: {_theme_color_scheme}!important;
        }}

        [data-testid="stAlert"], [data-testid="stAlert"] div,
        [data-baseweb="notification"], [data-baseweb="notification"] div {{
            background: {_theme_bg_alt}!important;
            color: {_theme_text}!important;
            border-color: {_theme_border}!important;
        }}

        [data-testid="stDataFrame"] iframe,
        iframe, canvas {{
            background: {_theme_bg}!important;
            color-scheme: {_theme_color_scheme}!important;
        }}

        [data-testid="stHeader"] button,
        [data-testid="stToolbar"] button,
        [data-testid="stStatusWidget"] button,
        [data-testid="stHeader"] a,
        [data-testid="stToolbar"] a,
        [data-testid="stStatusWidget"] a {{
            background: {_theme_bg}!important;
            color: {_theme_text}!important;
            border: 1px solid {_theme_border}!important;
            border-radius: 8px!important;
            box-shadow: none!important;
        }}

        [data-testid="stHeader"] button *,
        [data-testid="stToolbar"] button *,
        [data-testid="stStatusWidget"] button *,
        [data-testid="stHeader"] a *,
        [data-testid="stToolbar"] a *,
        [data-testid="stStatusWidget"] a *,
        [data-testid="stHeader"] svg,
        [data-testid="stToolbar"] svg,
        [data-testid="stStatusWidget"] svg {{
            background: transparent!important;
            color: {_theme_text}!important;
            fill: {_theme_text}!important;
            stroke: {_theme_text}!important;
        }}

        [data-testid="stImage"],
        [data-testid="stImage"] > div,
        [data-testid="stImage"] figure,
        [data-testid="stpyplot"],
        [data-testid="stpyplot"] > div,
        [data-testid="stpyplot"] figure,
        [data-testid="stArrowVegaLiteChart"],
        [data-testid="stArrowVegaLiteChart"] > div,
        [data-testid="stVegaLiteChart"],
        [data-testid="stVegaLiteChart"] > div {{
            background: {_theme_bg}!important;
            color: {_theme_text}!important;
            border: 1px solid {_theme_border}!important;
            border-radius: 12px!important;
            padding: 12px!important;
            color-scheme: {_theme_color_scheme}!important;
        }}

        [data-testid="stImage"] img,
        [data-testid="stpyplot"] img,
        [data-testid="stArrowVegaLiteChart"] canvas,
        [data-testid="stVegaLiteChart"] canvas {{
            background: {_theme_bg}!important;
            border-radius: 10px!important;
        }}

        table, thead, tbody, tr, th, td {{
            background: {_theme_bg}!important;
            color: {_theme_text}!important;
            border-color: {_theme_border}!important;
        }}

        th {{
            background: {_theme_header}!important;
        }}
    """

    _css = _css.replace('</style>', _theme_fix_css + '\n</style>')

    st.markdown(_css, unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* 1. Повністю прибираємо верхню службову панель Streamlit */
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        overflow: hidden !important;
    }

    /* 2. Прибираємо відступ, який залишався після header */
    .block-container {
        padding-top: 1rem !important;
    }

    /* 3. Забираємо чорні артефакти у верхніх fixed/portal контейнерах Streamlit */
    div[data-baseweb="popover"],
    div[data-baseweb="tooltip"],
    div[data-baseweb="menu"],
    div[data-baseweb="modal"],
    div[data-baseweb="drawer"] {
        background: transparent !important;
    }

    /* 4. Прибираємо чорне окантування навколо всіх картинок і графіків */
    [data-testid="stImage"],
    [data-testid="stImage"] > div,
    [data-testid="stImage"] figure,
    [data-testid="stpyplot"],
    [data-testid="stpyplot"] > div,
    [data-testid="stpyplot"] figure,
    [data-testid="stVegaLiteChart"],
    [data-testid="stVegaLiteChart"] > div,
    [data-testid="stArrowVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"] > div {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
    }
    
    /* 5. Прибираємо фон у вкладених div біля картинок/графіків */
    [data-testid="stImage"] div,
    [data-testid="stpyplot"] div,
    [data-testid="stVegaLiteChart"] div,
    [data-testid="stArrowVegaLiteChart"] div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* 6. Самі картинки, matplotlib, canvas, iframe без чорного фону */
    [data-testid="stImage"] img,
    [data-testid="stpyplot"] img,
    [data-testid="stVegaLiteChart"] canvas,
    [data-testid="stArrowVegaLiteChart"] canvas,
    iframe,
    canvas {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* 7. Прибираємо кнопку "на повний екран" біля кожного графіка/картинки */
    [data-testid="StyledFullScreenButton"],
    button[title="View fullscreen"],
    button[aria-label="View fullscreen"],
    button[title="Fullscreen"],
    button[aria-label="Fullscreen"],
    [data-testid="stImage"] button,
    [data-testid="stpyplot"] button,
    [data-testid="stVegaLiteChart"] button,
    [data-testid="stArrowVegaLiteChart"] button {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
                /* 8. Повністю блокуємо можливість згорнути сайдбар */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"] button[kind="header"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _title_color = '#000000' if not _dark else '#7c9fe6'
    _sub_color = '#000000' if not _dark else '#4a5580'
    st.markdown(
        f'<h1 style="font-size:32px;font-weight:900;color:{_title_color};letter-spacing:-1px;margin-bottom:2px">'
        f'🛡️ LoyaltyGuard</h1>'
        f'<p style="color:{_sub_color};font-size:14px;margin-top:0;margin-bottom:20px">'
        f'Система прогнозу відтоку клієнтів та підтримки рішень для retention-кампаній у рітейлі</p>',
        unsafe_allow_html=True,
    )

    raw_df = None
    source_name = ''

    with st.sidebar:
        st.header('Дані')

        # Одразу показуємо завантажувач без зайвих селектбоксів
        transactions_file = st.file_uploader(
            'Завантажте транзакції (CSV або XLSX)',
            type=['csv', 'xlsx', 'xls'],
            key='transactions_file',
        )

        if transactions_file is not None:
            try:
                raw_df = get_cached_dataframe(
                    transactions_file.getvalue(),
                    transactions_file.name,
                )
                source_name = transactions_file.name
                
                # Invalidate saved mapping if schema changed
                prev_cols = st.session_state.get('_prev_raw_columns')
                cur_cols = raw_df.columns.tolist()
                if prev_cols is not None and prev_cols != cur_cols:
                    st.session_state.pop('loaded_mapping_template', None)
                    st.session_state.pop('_mapping_auto_applied', None)
                st.session_state['_prev_raw_columns'] = cur_cols
            except Exception as error:
                st.error(str(error))
                st.stop()

        feedback_file = st.file_uploader(
            'Зовнішній feedback кампаній',
            type=['csv', 'xlsx', 'xls'],
            key='feedback_file',
        )

    if raw_df is None:
        st.info('Спочатку завантажте файл з транзакціями (CSV або XLSX).')
        st.stop()

    # ── Mapping section ────────────────────────────────────────────────────────
    detected_mapping = infer_column_mapping(raw_df.columns.tolist())
    column_options = [''] + list(raw_df.columns)
    all_templates = list_mapping_templates()

    # Template selector row (always visible, above expander)
    # ── Завантаження шаблону маппінгу ──
    st.markdown('**Завантажте збережений шаблон (.json) або залиште порожнім для автовизначення:**')
    uploaded_map = st.file_uploader('Файл маппінгу', type=['json'], label_visibility='collapsed', key='map_uploader')
    
    if uploaded_map is not None and uploaded_map.name != st.session_state.get('_last_map_filename'):
        import json
        try:
            loaded = json.loads(uploaded_map.getvalue().decode('utf-8'))
            st.session_state['loaded_mapping_template'] = loaded
            st.session_state['_template_name_to_save'] = uploaded_map.name.replace('.json', '')
            st.session_state['_mapping_auto_applied'] = False
            st.session_state['_last_map_filename'] = uploaded_map.name
            st.success('Маппінг успішно завантажено!')
            st.rerun()
        except Exception as e:
            st.error(f'Не вдалося прочитати JSON файл: {e}')

    loaded_template = st.session_state.get('loaded_mapping_template', {})
    schema_ok = template_matches_schema(loaded_template, raw_df.columns.tolist())

    # Auto-apply once if schema fully matches — hide mapping UI
    if schema_ok and not st.session_state.get('_mapping_auto_applied'):
        st.session_state['_mapping_auto_applied'] = True

    mapping_needed = not schema_ok or not st.session_state.get('_mapping_auto_applied', False)

    if schema_ok:
        st.success(
            'Схема даних збігається з шаблоном — маппінг застосовано автоматично. '
            'Розгорніть нижче, щоб переглянути або змінити.'
        )

    with st.expander(
        'Перегляд колонок та маппінг' + ('' if mapping_needed else ' (застосовано автоматично)'),
        expanded=mapping_needed,
    ):
        # Column preview table with examples
        from src.data_loader import preview_column_info
        col_info = preview_column_info(raw_df, n_examples=3)
        preview_rows = []
        for info in col_info:
            auto_match = next(
                (can for can, src in detected_mapping.items() if src == info['column']),
                '—',
            )
            tmpl_match = next(
                (can for can, src in loaded_template.items() if src == info['column']),
                None,
            )
            preview_rows.append({
                'Колонка': info['column'],
                'Тип': info['dtype'],
                'Null%': f"{info['null_pct']}%",
                'Приклади значень': ' | '.join(str(e) for e in info['examples']) or '—',
                'Маппінг': tmpl_match or auto_match,
            })
        dark_table(preview_rows, hide_index=True, height=320)

        st.markdown('#### Призначення колонок')
        st.caption('Система заповнила поля автоматично на основі шаблону або назв колонок. Виправте при потребі.')

        manual_mapping: dict[str, str] = {}
        req_cols = ['customer_id', 'transaction_date', 'transaction_id', 'product_id', 'product_name']
        opt_cols = [c for c in CANONICAL_TRANSACTION_COLUMNS if c not in req_cols]

        def _sample_caption(source_col: str) -> str:
            """Show first 3 non-null values of selected source column, або '—' якщо нічого."""
            if not source_col or source_col not in raw_df.columns:
                return '—'
            try:
                samples = (
                    raw_df[source_col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .replace('', pd.NA)
                    .dropna()
                    .head(3)
                    .tolist()
                )
            except Exception:
                return '—'
            if not samples:
                return '—'
            # Обрізаємо довгі значення, щоб caption не розпухав
            samples = [s if len(s) <= 24 else s[:21] + '…' for s in samples]
            return 'Приклад: ' + ', '.join(samples)

        st.markdown('**Обов\'язкові поля**')
        req_grid = st.columns(len(req_cols))
        for idx, canonical_name in enumerate(req_cols):
            label = CANONICAL_COLUMN_LABELS.get(canonical_name, canonical_name)
            default_col = loaded_template.get(canonical_name) or detected_mapping.get(canonical_name, '')
            default_index = column_options.index(default_col) if default_col in column_options else 0
            with req_grid[idx]:
                selected = st.selectbox(
                    label,
                    column_options,
                    index=default_index,
                    key=f'mapping_{canonical_name}',
                )
                manual_mapping[canonical_name] = selected
                st.caption(_sample_caption(selected))

        st.markdown('**Додаткові поля**')
        opt_grid = st.columns(min(len(opt_cols), 4))
        for idx, canonical_name in enumerate(opt_cols):
            label = CANONICAL_COLUMN_LABELS.get(canonical_name, canonical_name)
            default_col = loaded_template.get(canonical_name) or detected_mapping.get(canonical_name, '')
            default_index = column_options.index(default_col) if default_col in column_options else 0
            with opt_grid[idx % 4]:
                selected = st.selectbox(
                    label,
                    column_options,
                    index=default_index,
                    key=f'mapping_{canonical_name}',
                )
                manual_mapping[canonical_name] = selected
                st.caption(_sample_caption(selected))
        # ── Ізольований блок збереження файлу маппінгу на комп'ютер ──
        @st.fragment
        def render_download_mapping_block(mapping_data):
            import json
            
            st.markdown('**Зберегти налаштований маппінг на комп\'ютер:**')
            save_col1, save_col2 = st.columns([3, 5])
            
            with save_col1:
                # Введення тексту тут тепер НЕ перемальовує всю сторінку!
                file_base_name = st.text_input(
                    'Назва файлу',
                    value=st.session_state.get('_template_name_to_save', 'loyalty_mapping'),
                    label_visibility='collapsed',
                    placeholder='Назва шаблону...'
                )
                st.session_state['_template_name_to_save'] = file_base_name
            
            with save_col2:
                # Формуємо правильну назву файлу з розширенням
                clean_name = file_base_name.strip()
                filename = f"{clean_name}.json" if not clean_name.endswith('.json') else clean_name
                
                # Конвертуємо наш словник маппінгу в JSON-рядок
                json_string = json.dumps(mapping_data, ensure_ascii=False, indent=2)
                
                # Використовуємо download_button, який віддає файл локально без ре-рану
                st.download_button(
                    label='Скачати шаблон (.json)',
                    data=json_string,
                    file_name=filename,
                    mime='application/json',
                    use_container_width=True
                )

        # Викликаємо фрагмент і передаємо йому поточний стан маппінгу
        render_download_mapping_block(manual_mapping)
    # When schema matched and expander is collapsed, build mapping silently
    if not mapping_needed and 'manual_mapping' not in dir():
        manual_mapping = {
            canonical_name: (
                loaded_template.get(canonical_name)
                or detected_mapping.get(canonical_name, '')
            )
            for canonical_name in CANONICAL_TRANSACTION_COLUMNS
        }
    with st.sidebar:
        st.header('Модель')
        st.info('⚙️ Завантажте файл існуючої моделі (.pkl). Якщо поле порожнє — буде навчена нова модель.')
        
        uploaded_model = st.file_uploader(
            'Оберіть файл моделі (.pkl)',
            type=['pkl'],
            label_visibility='collapsed',
            key='model_uploader'
        )

    saved_churn_artifacts = None

    if uploaded_model is not None:
        import io
        try:
            # Читаємо модель прямо з байтів
            saved_churn_artifacts = joblib.load(io.BytesIO(uploaded_model.getvalue()))
            st.sidebar.success('Модель завантажено з файлу.')
        except Exception as e:
            st.sidebar.error(f'Помилка читання моделі: {e}')
            st.stop()

    # ── Cache key: only retrain when data / mapping / uploaded model actually change ──
    import hashlib as _hl, json as _json

    _data_hash = f"{source_name}_{raw_df.shape}_{list(raw_df.columns)}"

    _mapping_hash = _hl.md5(
        _json.dumps(
            {k: v for k, v in sorted((manual_mapping or {}).items())},
            ensure_ascii=False,
            sort_keys=True,
        ).encode('utf-8')
    ).hexdigest()

    # Хешуємо за наявністю завантаженого файлу моделі
    _saved_model_hash = f"{uploaded_model.name}_{uploaded_model.size}" if uploaded_model is not None else "train_new"

    _cache_key = _hl.md5(
        (
            str(source_name)
            + str(raw_df.shape)
            + _json.dumps(list(raw_df.columns), ensure_ascii=False)
            + _data_hash
            + _mapping_hash
            + _saved_model_hash
        ).encode('utf-8')
    ).hexdigest()

    # ── Єдина логіка запуску (навчання або завантаження) ──
    if st.session_state.get('_model_signature') != _cache_key:
        st.session_state['_model_built'] = False
        st.session_state['_model_signature'] = _cache_key

    _cached = st.session_state.get('_app_state_cache')
    
    if _cached is not None and _cached.get('key') == _cache_key:
        state = _cached['state']
    else:
        # Визначаємо тексти залежно від режиму
        if saved_churn_artifacts is not None:
            prompt_msg = 'ℹ️ Файл моделі готовий. Натисніть кнопку, щоб застосувати її та згенерувати дашборд.'
            btn_text = '⚙️ Застосувати збережену модель'
            spinner_text = '🔄 Розрахунок метрик та підготовка даних...'
        else:
            prompt_msg = '📋 Дані та маппінг готові. Натисніть кнопку нижче, щоб розпочати навчання моделі XGBoost.'
            btn_text = '🚀 Розпочати навчання моделі'
            spinner_text = '🔄 Навчання моделі та розрахунок метрик (це може зайняти деякий час)...'

        _build_model_clicked = st.button(btn_text, type='primary', use_container_width=True)
        
        if _build_model_clicked:
            st.session_state['_model_built'] = True

        if not st.session_state.get('_model_built', False):
            st.info(prompt_msg)
            st.stop()
            
        try:
            with st.spinner(spinner_text):
                new_state = build_app_state(
                    raw_df=raw_df,
                    source_name=source_name,
                    manual_mapping=manual_mapping,
                    _saved_churn_artifacts=saved_churn_artifacts,
                )
            st.session_state['_app_state_cache'] = {'key': _cache_key, 'state': new_state}
            st.rerun()
        except Exception as error:
            st.error(str(error))
            # Якщо сталася помилка, скидаємо статус, щоб користувач міг спробувати знову
            st.session_state['_model_built'] = False
            st.stop()

    with st.sidebar:
        # ── Ізольований блок збереження файлу моделі (.pkl) на комп'ютер ──
        @st.fragment
        def render_download_model_block(model_artifacts):
            import io
            import joblib

            st.markdown('**Зберегти модель на комп\'ютер:**')
            
            # Текстове поле для назви файлу (не перемальовує весь додаток завдяки фрагменту)
            model_file_base = st.text_input(
                'Назва файлу моделі',
                value=st.session_state.get('_model_filename_to_save', 'churn_model'),
                label_visibility='collapsed',
                placeholder='Назва моделі...'
            )
            st.session_state['_model_filename_to_save'] = model_file_base

            # Формуємо правильне розширення
            clean_name = model_file_base.strip()
            filename = f"{clean_name}.pkl" if not clean_name.endswith('.pkl') else clean_name

            # Запаковуємо модель у бінарний буфер в оперативній пам'яті
            buffer = io.BytesIO()
            joblib.dump(model_artifacts, buffer)
            model_bytes = buffer.getvalue()

            # Кнопка скачування віддає файл локально у браузер без повного ре-рану
            st.download_button(
                label='Скачати модель (.pkl)',
                data=model_bytes,
                file_name=filename,
                mime='application/octet-stream',
                use_container_width=True
            )

        # Викликаємо фрагмент і передаємо йому навчені артефакти моделі
        render_download_model_block(state['churn_artifacts'])

    latest_customers = state['latest_customers']
    rfm = state['rfm']
    churn_artifacts = state['churn_artifacts']
    feature_importance = state['feature_importance']
    category_table = state['category_table']
    top_risk_active = state['top_risk_active']
    X_all = state['X_all']

    # Створюємо нативні вкладки (перемикання відбувається миттєво без ре-рану)
    tab_overview, tab_campaign, tab_eda = st.tabs(['Overview', 'Campaign builder', 'EDA'])

    with tab_overview:
        st.markdown('### Огляд')
        c1, c2, c3 = st.columns(3)

        metric_row(
            c1, 'Клієнтів', latest_customers['customer_id'].nunique(),
            c2, 'Рядків у датасеті', len(state['predicted']),
            c3, 'ROC-AUC', f"{getattr(churn_artifacts, 'roc_auc', float('nan')):.4f}",
        )

        st.markdown('### Розподіл клієнтів за ризиком')
        _risk_s = state['risk_counts'].set_index('risk_class')['customers']
        plot_horizontal_counts(_risk_s, 'Розподіл клієнтів за ризиком', 'Кількість клієнтів', 'Клас ризику')

        st.markdown('### Статистика та стратегії по кластерах')

        # Build enriched cluster stats
        cluster_stats = (
            latest_customers
            .groupby('customer_cluster_name')
            .agg(
                customers=('customer_id', 'nunique'),
                avg_churn_risk_pct=('churn_probability_percent', 'mean'),
                high_risk_count=('risk_class', lambda v: int((v == 'High').sum())),
                medium_risk_count=('risk_class', lambda v: int((v == 'Medium').sum())),
                low_risk_count=('risk_class', lambda v: int((v == 'Low').sum())),
            )
            .reset_index()
        )
        for _num_col in ['avg_ticket_size', 'total_sales', 'online_purchases', 'promo_response_rate']:
            if _num_col in latest_customers.columns:
                _agg = (
                    latest_customers
                    .groupby('customer_cluster_name')[_num_col]
                    .mean()
                    .reset_index(name=_num_col)
                )
                cluster_stats = cluster_stats.merge(_agg, on='customer_cluster_name', how='left')
                cluster_stats[_num_col] = cluster_stats[_num_col].round(2)

        if 'rfm_segment' in latest_customers.columns:
            _top_seg = (
                latest_customers
                .groupby(['customer_cluster_name', 'rfm_segment'])
                .size()
                .reset_index(name='_cnt')
                .sort_values('_cnt', ascending=False)
                .groupby('customer_cluster_name')
                .first()
                .reset_index()[['customer_cluster_name', 'rfm_segment']]
                .rename(columns={'rfm_segment': 'top_rfm_segment'})
            )
            cluster_stats = cluster_stats.merge(_top_seg, on='customer_cluster_name', how='left')

        cluster_stats['avg_churn_risk_pct'] = cluster_stats['avg_churn_risk_pct'].round(1)

        for _, _row in cluster_stats.iterrows():
            _cname = _row['customer_cluster_name']
            _strat_title, _strat_text = cluster_strategy_text(_cname)
            _bg = '#1a2a3a'
            if not _dark:
                _bg = '#ffffff'
            _risk_pct = float(_row['avg_churn_risk_pct'])
            _risk_color = '#000000' if not _dark else '#f87171' if _risk_pct > 60 else '#fbbf24' if _risk_pct > 30 else '#34d399'
            _card_border = '#d9d9d9' if not _dark else 'rgba(255,255,255,0.08)'
            _muted = '#000000' if not _dark else '#8891aa'
            _strong = '#000000' if not _dark else '#e8eaf0'
            _accent = '#000000' if not _dark else '#7c9fe6'
            _strategy_bg = '#f2f2f2' if not _dark else 'rgba(255,255,255,0.04)'
            _pill_bg = '#eeeeee' if not _dark else None
            _pill_high = _pill_med = _pill_low = '#000000'
            if _dark:
                _pill_high, _pill_med, _pill_low = '#f87171', '#fbbf24', '#34d399'
            _ticket_html = ''
            if 'avg_ticket_size' in _row and _row['avg_ticket_size'] == _row['avg_ticket_size']:
                _ticket_html = f'<span style="color:{_muted};font-size:13px">Сер. чек: <b style="color:{_strong}">£{_row["avg_ticket_size"]:.2f}</b></span>'
            _rfm_html = ''
            if 'top_rfm_segment' in _row:
                _rfm_html = f'<span style="color:{_muted};font-size:13px">Топ RFM: <b style="color:{_strong}">{_row["top_rfm_segment"]}</b></span>'

            st.markdown(
                f'<div style="background:{_bg};border-radius:10px;padding:16px 20px;margin-bottom:12px;border:1px solid {_card_border}">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">'
                f'<div><span style="font-size:18px;font-weight:700;color:{_strong}">{_cname}</span>'
                f'<span style="margin-left:12px;font-size:13px;color:{_muted}">{int(_row["customers"])} клієнтів</span></div>'
                f'<div style="display:flex;gap:10px;flex-wrap:wrap">'
                f'<span style="background:{_pill_bg or "rgba(248,113,113,0.15)"};color:{_pill_high};padding:3px 10px;border-radius:20px;font-size:12px">High: {int(_row["high_risk_count"])}</span>'
                f'<span style="background:{_pill_bg or "rgba(251,191,36,0.15)"};color:{_pill_med};padding:3px 10px;border-radius:20px;font-size:12px">Med: {int(_row["medium_risk_count"])}</span>'
                f'<span style="background:{_pill_bg or "rgba(52,211,153,0.15)"};color:{_pill_low};padding:3px 10px;border-radius:20px;font-size:12px">Low: {int(_row["low_risk_count"])}</span>'
                f'</div></div>'
                f'<div style="display:flex;gap:20px;margin-top:10px;flex-wrap:wrap">'
                f'<span style="color:{_muted};font-size:13px">Сер. ризик: <b style="color:{_risk_color}">{_risk_pct:.1f}%</b></span>'
                f'{_ticket_html}{_rfm_html}'
                f'</div>'
                f'<div style="margin-top:12px;padding:10px 14px;background:{_strategy_bg};border-radius:8px">'
                f'<span style="font-weight:600;color:{_accent}">{_strat_title}</span>'
                f'<p style="margin:4px 0 0;color:{_muted};font-size:13px;line-height:1.55">{_strat_text}</p>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        download_dataframe_button(cluster_stats, 'cluster_statistics.csv', 'Завантажити статистику по кластерах')

        st.markdown('### ТОП-10 активних клієнтів з найвищим ризиком')
        dark_table(top_risk_active, hide_index=True, height=None)
        download_dataframe_button(top_risk_active, 'top_risk_active.csv', 'Завантажити TOP-10')

        if feedback_file is not None:
            st.markdown('---')
            st.markdown('### Зовнішній feedback кампаній')
            st.caption(
                'Файл зі зворотнім зв’язком з CRM/пошти або іншої системи...'
            )
            
            @st.fragment
            def render_feedback_analysis():
                try:
                    _fb_df = get_cached_dataframe(feedback_file.getvalue(), feedback_file.name)
                    # ── Auto-detect customer_id column by common aliases ─────────
                    _FB_CID_ALIASES = {
                        'customer_id', 'customerid', 'customer id',
                        'user_id', 'userid', 'user id',
                        'client_id', 'clientid', 'client id',
                        'cust_id', 'custid',
                    }
                    _fb_cid_col = None
                    for _col in _fb_df.columns:
                        if _col.strip().lower() in _FB_CID_ALIASES:
                            _fb_cid_col = _col
                            break
                    if _fb_cid_col and _fb_cid_col != 'customer_id':
                        _fb_df = _fb_df.rename(columns={_fb_cid_col: 'customer_id'})

                    _fb_n_rows = len(_fb_df)
                    _fb_n_cols = len(_fb_df.columns)

                    # ── Quick stats row ──────────────────────────────────────────
                    _fc1, _fc2, _fc3, _fc4 = st.columns(4)
                    with _fc1:
                        st.metric('Рядків у feedback', f'{_fb_n_rows:,}')
                    with _fc2:
                        st.metric('Колонок', _fb_n_cols)
                    with _fc3:
                        _has_cid = 'customer_id' in _fb_df.columns
                        _cid_label = (
                            f'є ({_fb_cid_col})' if (_has_cid and _fb_cid_col and _fb_cid_col != 'customer_id')
                            else ('є' if _has_cid else 'немає')
                        )
                        st.metric('customer_id', _cid_label)
                    with _fc4:
                        if _has_cid:
                            _match_n = _fb_df['customer_id'].astype(str).str.strip().isin(
                                latest_customers['customer_id'].astype(str)
                            ).sum()
                            st.metric('Збіг з клієнтами', f'{_match_n:,}')
                        else:
                            st.metric('Збіг', '—')

                    # ── Column preview ───────────────────────────────────────────
                    with st.expander('Перегляд файлу feedback'):
                        dark_table(_fb_df.head(50), hide_index=True, height=360)

                    if not _has_cid:
                        st.warning(
                            'У feedback-файлі не знайдено колонки з ID клієнта. '
                            f'Знайдені колонки: {list(_fb_df.columns)}. '
                            'Очікується одна з: customer_id, user_id, User ID тощо.'
                        )
                        dark_table(_fb_df.head(100), hide_index=True, height=420)
                    else:
                        # Detect status/response column automatically
                        _NON_STATUS_COLS = {
                            'customer_id', 'customerid', 'id',
                            'suggested item', 'suggested_item', 'item', 'product',
                        }
                        _status_candidates = [
                            c for c in _fb_df.columns
                            if c.lower() not in _NON_STATUS_COLS
                            and _fb_df[c].nunique() <= 20
                        ]

                        # Detect item/product column (e.g. "Suggested Item")
                        _ITEM_ALIASES = {'suggested item', 'suggested_item', 'item', 'product', 'product_name', 'товар'}
                        _fb_item_col = next(
                            (c for c in _fb_df.columns if c.strip().lower() in _ITEM_ALIASES),
                            None,
                        )

                        _status_col = None
                        if _status_candidates:
                            _status_col = st.selectbox(
                                'Колонка статусу / реакції',
                                ['— не вибрано'] + _status_candidates,
                                key='feedback_status_col',
                            )
                            if _status_col == '— не вибрано':
                                _status_col = None

                        # Merge feedback with predicted customers
                        _fb_clean = _fb_df.copy()
                        _fb_clean['customer_id'] = _fb_clean['customer_id'].astype(str).str.strip()
                        _lc_copy = latest_customers[['customer_id', 'churn_probability_percent', 'risk_class',
                                                    'rfm_segment', 'customer_cluster_name']].copy()
                        _lc_copy['customer_id'] = _lc_copy['customer_id'].astype(str).str.strip()
                        _merged = _fb_clean.merge(_lc_copy, on='customer_id', how='inner')

                        if len(_merged) == 0:
                            st.info(
                                'Жодного перетину між feedback і предікціями. '
                                f'ID з feedback (приклади): {_fb_clean["customer_id"].head(5).tolist()}. '
                                f'ID в моделі (приклади): {_lc_copy["customer_id"].head(5).tolist()}.'
                            )
                        else:
                            st.markdown(f'**Перетин: {len(_merged):,} клієнтів**')

                            if _status_col:
                                st.markdown('#### Розподіл реакцій за рівнем ризику')
                                _pivot = (
                                    _merged
                                    .groupby(['risk_class', _status_col])
                                    .size()
                                    .reset_index(name='клієнтів')
                                )
                                dark_table(_pivot, hide_index=True, height=320)

                                st.markdown('#### Сер. ризик відтоку за статусом')
                                _avg_risk = (
                                    _merged
                                    .groupby(_status_col)['churn_probability_percent']
                                    .mean()
                                    .round(1)
                                    .reset_index()
                                )
                                _avg_risk.columns = [_status_col, 'Сер. ризик, %']
                                dark_table(_avg_risk, hide_index=True, height=320)

                                # ── Item-level analysis (if Suggested Item present) ──
                                if _fb_item_col and _fb_item_col in _merged.columns:
                                    st.markdown(f'#### Реакція за товаром (`{_fb_item_col}`)')
                                    _item_pivot = (
                                        _merged
                                        .groupby([_fb_item_col, _status_col])
                                        .size()
                                        .reset_index(name='клієнтів')
                                        .sort_values('клієнтів', ascending=False)
                                    )
                                    dark_table(_item_pivot.head(50), hide_index=True, height=360)

                                    _negative_reactions = {'not interested', 'no action', 'unsubscribed', 'ignored'}
                                    _positive = _merged[
                                        ~_merged[_status_col].str.lower().isin(_negative_reactions)
                                    ]
                                    if len(_positive) > 0:
                                        _top_items = (
                                            _positive[_fb_item_col]
                                            .value_counts()
                                            .head(10)
                                            .reset_index()
                                        )
                                        _top_items.columns = ['Товар', 'Позитивних реакцій']
                                        st.markdown('#### ТОП-10 товарів з позитивними реакціями')
                                        dark_table(_top_items, hide_index=True, height=320)

                            st.markdown('#### Об\u2019єднана таблиця (feedback + предікції)')
                            dark_table(_merged.head(100), hide_index=True, height=420)
                            download_dataframe_button(_merged, 'feedback_enriched.csv', 'Завантажити feedback + предікції')
                except Exception as _fb_error:
                    st.error(f'Не вдалося зчитати feedback-файл: {_fb_error}')
            render_feedback_analysis()

        with st.expander('Нотатки автопідготовки'):
            if state['notes']:
                for note in state['notes']:
                    st.write(f'- {note}')
            else:
                st.write('Додаткових нотаток немає.')

    # elif active_page == '2. Churn & RFM':
    #     st.markdown('### 📊 Churn-модель: метрики')
    #
    #     def _stat_card(col, label, value, hint='', color='#7c9fe6'):
    #         with col:
    #             _bg = '#ffffff' if not _dark else '#1a1e2e'
    #             _border = '#d9d9d9' if not _dark else 'rgba(255,255,255,0.07)'
    #             _text = '#000000' if not _dark else '#4a5580'
    #             _value = '#000000' if not _dark else color
    #             st.markdown(
    #                 f'<div style="background:{_bg};border-radius:10px;padding:16px 14px;'
    #                 f'text-align:center;border:1px solid {_border};margin-bottom:8px">'
    #                 f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:1.2px;'
    #                 f'color:{_text};margin-bottom:6px">{label}</div>'
    #                 f'<div style="font-size:38px;font-weight:800;color:{_value};line-height:1.1">{value}</div>'
    #                 f'<div style="font-size:11px;color:{_text};margin-top:4px">{hint}</div>'
    #                 f'</div>',
    #                 unsafe_allow_html=True,
    #             )
    #
    #     def _fmt4(v):
    #         try:
    #             f = float(v)
    #             return f'{f:.4f}'
    #         except Exception:
    #             return 'n/a'
    #
    #     _ca = churn_artifacts
    #     _auc  = getattr(_ca, 'roc_auc',   float('nan'))
    #     _acc  = getattr(_ca, 'accuracy',   float('nan'))
    #     _prec = getattr(_ca, 'precision',  float('nan'))
    #     _rec  = getattr(_ca, 'recall',     float('nan'))
    #     _f1   = getattr(_ca, 'f1',         float('nan'))
    #     _ll   = getattr(_ca, 'logloss',    float('nan'))
    #     _algo = getattr(_ca, 'algorithm_name', 'n/a')
    #     _nt   = len(getattr(_ca, 'test_index', []))
    #
    #     # Row 1 — primary metrics
    #     _r1 = st.columns(4)
    #     _stat_card(_r1[0], 'ROC-AUC',  _fmt4(_auc),  'де 0.5 = випадково', '#7c9fe6')
    #     _stat_card(_r1[1], 'Accuracy', _fmt4(_acc),  f'Test rows: {_nt}',             '#a78bfa')
    #     _stat_card(_r1[2], 'F1-score', _fmt4(_f1),   'Prec / Rec баланс',          '#34d399')
    #     _stat_card(_r1[3], 'LogLoss',  _fmt4(_ll),   'нижче = краще',                    '#fbbf24')
    #
    #     # Row 2 — secondary metrics
    #     _r2 = st.columns(4)
    #     _stat_card(_r2[0], 'Precision', _fmt4(_prec), 'TP / (TP+FP)', '#38bdf8')
    #     _stat_card(_r2[1], 'Recall',    _fmt4(_rec),  'TP / (TP+FN)', '#38bdf8')
    #     _stat_card(_r2[2], 'Алгоритм',    _algo,         '',              '#e8eaf0')
    #     _stat_card(_r2[3], 'Test rows', str(_nt),     'рядків в тестовій вибірці', '#e8eaf0')
    #
    #     st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    #
    #     # ROC + Confusion side by side
    #     _rcol, _ccol = st.columns([3, 2])
    #     with _rcol:
    #         st.markdown('#### ROC-крива')
    #         plot_roc_curve(_ca.fpr, _ca.tpr, _ca.roc_auc)
    #     with _ccol:
    #         st.markdown('#### Confusion Matrix')
    #         _cm = getattr(_ca, 'confusion_matrix_table', None)
    #         if _cm is not None:
    #             dark_table(_cm, hide_index=False, height=260)
    #             _tn = int(_cm.iloc[0, 0]); _fp = int(_cm.iloc[0, 1])
    #             _fn = int(_cm.iloc[1, 0]); _tp = int(_cm.iloc[1, 1])
    #             _cm_bg = '#ffffff' if not _dark else '#1a1e2e'
    #             _cm_text = '#000000' if not _dark else '#8891aa'
    #             _cm_border = '#d9d9d9' if not _dark else 'rgba(255,255,255,0.07)'
    #             _cm_mark = '#000000' if not _dark else None
    #             st.markdown(
    #                 f'<div style="background:{_cm_bg};border-radius:10px;padding:10px 14px;'
    #                 f'font-size:13px;color:{_cm_text};margin-top:8px;border:1px solid {_cm_border}">'
    #                 f'<b style="color:{_cm_mark or "#34d399"}">TP</b> {_tp}  '
    #                 f'<b style="color:{_cm_mark or "#34d399"}">TN</b> {_tn}  '
    #                 f'<b style="color:{_cm_mark or "#f87171"}">FP</b> {_fp}  '
    #                 f'<b style="color:{_cm_mark or "#fbbf24"}">FN</b> {_fn}'
    #                 f'</div>',
    #                 unsafe_allow_html=True,
    #             )
    #
    #     with st.expander('📊 Важливість ознак (Top-20)'):
    #         importance_plot = feature_importance.head(20).copy()
    #
    #         if importance_plot.empty:
    #             st.info('Немає даних для графіка важливості ознак.')
    #         else:
    #             feature_col = importance_plot.columns[0]
    #             importance_col = importance_plot.columns[-1]
    #
    #             importance_plot[importance_col] = pd.to_numeric(
    #                 importance_plot[importance_col],
    #                 errors='coerce'
    #             ).fillna(0)
    #
    #             importance_plot = importance_plot.sort_values(
    #                 by=importance_col,
    #                 ascending=True
    #             )
    #
    #             fig, ax = plt.subplots(figsize=(10, 7))
    #
    #             ax.barh(
    #                 importance_plot[feature_col].astype(str),
    #                 importance_plot[importance_col],
    #                 color=CHART_COLORS[0]
    #             )
    #
    #             ax.set_title('Важливість ознак моделі')
    #             ax.set_xlabel('Важливість')
    #             ax.set_ylabel('Ознака')
    #             ax.grid(axis='x', alpha=0.3)
    #
    #             for i, value in enumerate(importance_plot[importance_col]):
    #                 ax.text(
    #                     value,
    #                     i,
    #                     f' {value:.4f}',
    #                     va='center',
    #                     fontsize=9
    #                 )
    #
    #             fig.tight_layout()
    #             finish_chart(fig)
    #
    #     st.markdown('### 📋 RFM-сегменти')
    #     _rfm_s = state['rfm_counts'].set_index('rfm_segment')['customers']
    #     plot_horizontal_counts(_rfm_s, 'RFM-сегменти', 'Кількість клієнтів', 'Сегмент')
    #
    #     st.markdown('### RFM-таблиця')
    #     dark_table(rfm.head(50), hide_index=True, height=None)
    #     download_dataframe_button(rfm, 'rfm_table.csv', 'Завантажити RFM таблицю')

    # elif active_page == '3. Categories':
    #     st.markdown('### Найчастіші категорії клієнтів')
        # c1, c2, c3 = st.columns(3)
        # metric_row(
        #     c1, 'Клієнтів із конкретною категорією, %', state['category_coverage_pct'],
        #     c2, 'Клієнтів з fallback-категорією', int((latest_customers['dominant_category_display'] == CATEGORY_OTHER_LABEL).sum()),
        #     c3, 'Товарних рядків для автокатегоризації', len(state['products']),
        # )

        # chart_df = state['specific_category_counts'].copy()
        # if len(chart_df) == 0:
        #     chart_df = state['category_counts'].copy()
        #
        # _cat_s = chart_df.head(10).set_index('category')['customers']
        # plot_horizontal_counts(_cat_s, 'Топ категорій', 'Кількість клієнтів', 'Категорія')

        # st.markdown('### Узгоджений профіль категорій клієнтів')
        # dark_table(category_table.head(100), hide_index=True, height=420)
        # download_dataframe_button(category_table, 'customer_category_profile.csv', 'Завантажити профіль категорій')

        # with st.expander('Показати приклади автокатегоризації товарів'):
        #     preview_cols = ['customer_id', 'product_id', 'product_name', 'category']
        #     preview_cols = [col for col in preview_cols if col in state['products'].columns]
        #     dark_table(state['products'][preview_cols].head(100), hide_index=True, height=420)

    with tab_campaign:
        st.markdown('### Конструктор кампаній')

        # Створюємо ізольований фрагмент. Усе, що всередині, оновлюватиметься окремо від усієї сторінки!
        @st.fragment
        @st.fragment
        def render_campaign_tab():
            st.markdown('Налаштування цільової аудиторії')
            
            # Розміщуємо три фільтри в один гарний ряд
            col1, col2, col3 = st.columns(3)
            
            with col1:
                available_rfm = sorted(latest_customers['rfm_segment'].dropna().unique().tolist()) if 'rfm_segment' in latest_customers.columns else []
                selected_rfm = st.multiselect('RFM Сегменти', options=available_rfm, placeholder='Напр. VIP')
                
            with col2:
                available_risk = sorted(latest_customers['risk_class'].dropna().unique().tolist()) if 'risk_class' in latest_customers.columns else []
                selected_risk = st.multiselect('Клас ризику', options=available_risk, placeholder='Напр. High')
                
            with col3:
                available_clusters = sorted(latest_customers['customer_cluster_name'].dropna().unique().tolist()) if 'customer_cluster_name' in latest_customers.columns else []
                selected_clusters = st.multiselect('ML Кластери', options=available_clusters, placeholder='Оберіть...')

            audience_limit = st.slider('Скільки рядків показувати в таблиці', 10, 500, 100, 10)
            
            campaign_name = 'Retention campaign'
            offer_type = 'Auto'
            category_mode = 'Auto'
            manual_category = ''
            channel_mode = 'Auto'

            _campaign_signature = (
                len(latest_customers),
                id(churn_artifacts),
                feedback_file.name if feedback_file is not None else None,
                feedback_file.size if feedback_file is not None else 0,
            )
            if st.session_state.get('_campaign_signature') != _campaign_signature:
                st.session_state['_campaign_built'] = False
                st.session_state['_campaign_signature'] = _campaign_signature

            _build_clicked = st.button(
                'Побудувати кампанію',
                type='primary',
                help='Аналізує аудиторію, навчає feedback-модель та формує таблицю.',
            )
            if _build_clicked:
                st.session_state['_campaign_built'] = True

            if not st.session_state.get('_campaign_built', False):
                st.info('ℹНалаштуйте фільтри та натисніть **Побудувати кампанію**.')
                return  

            # --- Логіка побудови кампанії ---
            audience = latest_customers.copy()
            
            # Тільки жорстка логіка (класичний перетин AND)
            if selected_rfm:
                audience = audience[audience['rfm_segment'].isin(selected_rfm)]
            if selected_risk:
                audience = audience[audience['risk_class'].isin(selected_risk)]
            if selected_clusters:
                audience = audience[audience['customer_cluster_name'].isin(selected_clusters)]
                
            if len(audience) == 0:
                st.warning('Аудиторія порожня. Жоден клієнт не підпадає під усі обрані фільтри одночасно.')
                return
            fb_artifacts = None
            if feedback_file is not None:
                try:
                    _fb_raw = get_cached_dataframe(feedback_file.getvalue(), feedback_file.name)
                    detected = autodetect_feedback_columns(_fb_raw)
                    cid_col = detected['customer_id']
                    item_col = detected['item']
                    reaction_col = detected['reaction']

                    if cid_col and item_col and reaction_col:
                        _fb_raw[cid_col] = _fb_raw[cid_col].astype(str).str.strip()

                        _feedback_cache_key = (
                            feedback_file.name,
                            len(feedback_file.getvalue()),
                            tuple(_fb_raw.columns),
                            len(_fb_raw),
                            len(latest_customers),
                        )
                        _cached_feedback = st.session_state.get('_feedback_response_cache')

                        if (
                            _cached_feedback is not None
                            and _cached_feedback.get('key') == _feedback_cache_key
                        ):
                            fb_artifacts = _cached_feedback.get('artifacts')
                        else:
                            with st.spinner('Навчання feedback-моделі реакції...'):
                                X_fb, y_fb, groups_fb, fcols, ccols = prepare_feedback_training_set(
                                    feedback_df=_fb_raw,
                                    customer_features=latest_customers,
                                    categorizer=lambda t: categorize_product(t)[0],
                                    cid_col=cid_col,
                                    item_col=item_col,
                                    reaction_col=reaction_col,
                                )
                                fb_service = FeedbackResponseService(use_xgboost=True)
                                fb_artifacts = fb_service.train(X_fb, y_fb, groups_fb, fcols, ccols)
                                st.session_state['_feedback_response_cache'] = {
                                    'key': _feedback_cache_key,
                                    'artifacts': fb_artifacts,
                                }

                        st.success(
                            f'Feedback-модель навчена ({fb_artifacts.algorithm_name}). '
                            f'ROC-AUC = {fb_artifacts.roc_auc:.4f}. '
                            f'Категорій у моделі: {len(fb_artifacts.known_categories)}.'
                        )
                    else:
                        st.warning(
                            f'Feedback файл не містить усіх потрібних колонок. '
                            f'Знайдено: {detected}. Очікується customer_id + item + reaction.'
                        )
                except Exception as _err:
                    st.error(f'Помилка тренування feedback-моделі: {_err}')

            if fb_artifacts is not None and len(audience) > 0:
                audience = attach_feedback_recommendations(audience, fb_artifacts)
                audience['campaign_category'] = audience['best_category']
                final_campaign = build_feedback_campaign_table(audience, campaign_name)
            else:
                final_campaign = build_campaign_table(
                    audience=audience,
                    campaign_name=campaign_name,
                    offer_type=offer_type,
                    channel_mode=channel_mode,
                    category_mode=category_mode,
                    manual_category=manual_category,
                )

            st.markdown(f'### Розмір аудиторії: {len(final_campaign)}')

            st.markdown('### What-if сценарій')
            st.caption('Змоделюйте, як зміна поведінки клієнтів вплине на ймовірність відтоку.')

            def _slider_html(val: int, unit: str = '%') -> str:
                if val > 0:
                    color, glow, arrow, sign = '#34d399', 'rgba(52,211,153,.30)', '↑', '+'
                elif val < 0:
                    color, glow, arrow, sign = '#f87171', 'rgba(248,113,113,.30)', '↓', ''
                else:
                    color, glow, arrow, sign = '#4a5580', 'transparent', '→', ''
                return (
                    f'<div style="text-align:center;margin:6px 0 4px">'
                    f'<div style="display:inline-block;background:rgba(0,0,0,.35);'
                    f'border:2px solid {color};border-radius:16px;padding:10px 32px;'
                    f'box-shadow:0 0 22px {glow},0 0 6px {glow}">'
                    f'<span style="font-size:48px;font-weight:900;color:{color}!important;'
                    f'letter-spacing:-2px;line-height:1;text-shadow:0 0 20px {glow}">'
                    f'{arrow}&nbsp;{sign}{val}{unit}</span>'
                    f'</div>'
                    f'<div class="whatif-scale">'
                    f'<span class="whatif-min">−50%</span>'
                    f'<span class="whatif-sep">|||||||||</span>'
                    f'<span class="whatif-zero">0</span>'
                    f'<span class="whatif-sep">|||||||||</span>'
                    f'<span class="whatif-max">+50%</span>'
                    f'</div>'
                    f'</div>'
                )

            _sl1, _sl2, _sl3 = st.columns(3)

            with _sl1:
                st.markdown('**Частота покупок**')
                freq_change_percent = st.slider(
                    'Частота, %',
                    min_value=-50, max_value=50, value=0, step=5, key='sl_freq',
                )
                st.markdown(_slider_html(freq_change_percent), unsafe_allow_html=True)

            with _sl2:
                st.markdown('**Середній чек**')
                ticket_change_percent = st.slider(
                    'Чек, %',
                    min_value=-50, max_value=50, value=0, step=5, key='sl_ticket',
                )
                st.markdown(_slider_html(ticket_change_percent), unsafe_allow_html=True)

            with _sl3:
                st.markdown('**Реакція на акції**')
                promo_change_percent = st.slider(
                    'Промо, п.п.',
                    min_value=-50, max_value=50, value=0, step=5, key='sl_promo',
                )
                st.markdown(_slider_html(promo_change_percent, ' п.п.'), unsafe_allow_html=True)

            ticket_change = ticket_change_percent / 100
            freq_change = freq_change_percent / 100
            promo_change = promo_change_percent / 100

            if len(audience) > 0:
                audience_X = X_all.loc[audience['feature_row_index']].copy().reset_index(drop=True)
                effect_summary, effect_details = estimate_campaign_effect(
                    audience_df=audience,
                    audience_X=audience_X,
                    artifacts=churn_artifacts,
                    freq_change_pct=freq_change,
                    ticket_change_pct=ticket_change,
                    promo_change_pct=promo_change,
                )

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_row(
                    metric_col1, 'Аудиторія', int(effect_summary['audience_size'].iloc[0]),
                    metric_col2, 'Сер. ризик до, %', effect_summary['avg_risk_before_pct'].iloc[0],
                    metric_col3, 'Сер. ризик після, %', effect_summary['avg_risk_after_pct'].iloc[0],
                )

                st.markdown('### Фінальна таблиця кампанії')
                dark_table(final_campaign.head(audience_limit), hide_index=True, height=None)
                download_dataframe_button(final_campaign, 'campaign_final.csv', 'Завантажити фінальну кампанію')
        # Важливо: викликаємо наш фрагмент!
        render_campaign_tab()

    with tab_eda:
        render_eda_tab(state)

    # elif active_page == 'ℹ️ Про модель':
    #     st.markdown('## 📊 Про модель')
    #
    #     _ca = churn_artifacts
    #
    #     # ── Big metric cards ──────────────────────────────────────────────────
    #     _mc = st.columns(4)
    #
    #     def _big_metric(col, label, value, sub=''):
    #         with col:
    #             _bg   = '#ffffff' if not _dark else '#1a1e2e'
    #             _bdr  = '1px solid #d9d9d9' if not _dark else '1px solid rgba(255,255,255,0.07)'
    #             _lclr = '#000000' if not _dark else '#8891aa'
    #             _vclr = '#000000' if not _dark else '#7c9fe6'
    #             _sclr = '#000000' if not _dark else '#4a5580'
    #             st.markdown(
    #                 f'<div style="background:{_bg};border-radius:10px;padding:18px 16px;text-align:center;border:{_bdr};box-shadow:none">'
    #                 f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:{_lclr};margin-bottom:4px">{label}</div>'
    #                 f'<div style="font-size:40px;font-weight:800;color:{_vclr};line-height:1">{value}</div>'
    #                 f'<div style="font-size:12px;color:{_sclr};margin-top:4px">{sub}</div>'
    #                 f'</div>',
    #                 unsafe_allow_html=True,
    #             )
    #
    #     _auc_val  = getattr(_ca, 'roc_auc',  float('nan'))
    #     _acc_val  = getattr(_ca, 'accuracy',  float('nan'))
    #     _prec_val = getattr(_ca, 'precision', float('nan'))
    #     _rec_val  = getattr(_ca, 'recall',    float('nan'))
    #     _f1_val   = getattr(_ca, 'f1',        float('nan'))
    #     _ll_val   = getattr(_ca, 'logloss',   float('nan'))
    #     _algo     = getattr(_ca, 'algorithm_name', 'n/a')
    #     _n_test   = len(getattr(_ca, 'test_index', []))
    #
    #     def _fmt(v):
    #         try:
    #             return f'{v:.4f}'
    #         except Exception:
    #             return 'n/a'
    #
    #     _big_metric(_mc[0], 'ROC-AUC',   _fmt(_auc_val),  'де 0.5 = випадково, 1.0 = ідеал')
    #     _big_metric(_mc[1], 'Accuracy',  _fmt(_acc_val),  f'Test rows: {_n_test}')
    #     _big_metric(_mc[2], 'F1-score',  _fmt(_f1_val),   'Precision / Recall')
    #     _big_metric(_mc[3], 'LogLoss',   _fmt(_ll_val),   'нижче = краще')
    #
    #     st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    #
    #     _mc2 = st.columns(4)
    #     _big_metric(_mc2[0], 'Precision', _fmt(_prec_val), 'TP / (TP+FP)')
    #     _big_metric(_mc2[1], 'Recall',    _fmt(_rec_val),  'TP / (TP+FN)')
    #     _big_metric(_mc2[2], 'Алгоритм',   _algo,           '')
    #     _big_metric(_mc2[3], 'Test size', f'{_n_test}',    'рядків')
    #
    #     st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    #
    #     # ── ROC curve + Confusion matrix side by side ─────────────────────────
    #     _rcol, _ccol = st.columns([3, 2])
    #     with _rcol:
    #         st.markdown('#### ROC-крива')
    #         plot_roc_curve(_ca.fpr, _ca.tpr, _ca.roc_auc)
    #
    #     with _ccol:
    #         st.markdown('#### Confusion Matrix')
    #         _cm = getattr(_ca, 'confusion_matrix_table', None)
    #         if _cm is not None:
    #             dark_table(_cm, hide_index=False, height=260)
    #             _tn = int(_cm.iloc[0, 0])
    #             _fp = int(_cm.iloc[0, 1])
    #             _fn = int(_cm.iloc[1, 0])
    #             _tp = int(_cm.iloc[1, 1])
    #             _cm_bg  = '#ffffff' if not _dark else '#1a1e2e'
    #             _cm_txt = '#000000' if not _dark else '#8891aa'
    #             _tp_c   = '#000000' if not _dark else '#e8eaf0'
    #             _err_c  = '#000000' if not _dark else '#dc2626'
    #             _warn_c = '#000000' if not _dark else '#d97706'
    #             st.markdown(
    #                 f'<div style="margin-top:12px;background:{_cm_bg};border-radius:10px;padding:12px 16px;font-size:13px;color:{_cm_txt};border:1px solid {"#d9d9d9" if not _dark else "rgba(255,255,255,.07)"}">'
    #                 f'<b style="color:{_tp_c}">TP</b> {_tp} &nbsp; '
    #                 f'<b style="color:{_tp_c}">TN</b> {_tn} &nbsp; '
    #                 f'<b style="color:{_err_c}">FP</b> {_fp} &nbsp; '
    #                 f'<b style="color:{_warn_c}">FN</b> {_fn}'
    #                 f'</div>',
    #                 unsafe_allow_html=True,
    #             )
    #         else:
    #             st.info('Confusion matrix недоступна.')
    #
    #     # ── Feature importance ────────────────────────────────────────────────
    #     st.markdown('#### Важливість ознак (Top-20)')
    #     _fi = feature_importance.head(20).copy()
    #     if not _fi.empty:
    #         import matplotlib.pyplot as _plt
    #         _fig, _ax = _plt.subplots(figsize=(8, max(4, len(_fi) * 0.32)))
    #         _colors = ['#000000' if i < 5 else '#777777' for i in range(len(_fi))] if not _dark else ['#7c9fe6' if i < 5 else '#4a5580' for i in range(len(_fi))]
    #         _ax.barh(_fi['feature'][::-1], _fi['importance'][::-1], color=_colors[::-1])
    #         _ax.set_xlabel('Importance')
    #         _ax.set_title('Feature importances (top 20)', fontsize=11)
    #         _ax.spines[['top', 'right']].set_visible(False)
    #         _fig.tight_layout()
    #         st.pyplot(_fig)
    #
    #     from src.config import LOW_RISK_MAX, MEDIUM_RISK_MAX
    #     _r_val_c = '#000000' if not _dark else '#e8eaf0'
    #     _risk_label_c = '#000000' if not _dark else None
    #     _low_bg  = '#ffffff' if not _dark else '#1a3a1a'
    #     _mid_bg  = '#ffffff' if not _dark else '#2a2a1a'
    #     _hi_bg   = '#ffffff' if not _dark else '#3a1a1a'
    #     _risk_border = '#d9d9d9' if not _dark else 'transparent'
    #     st.markdown(
    #         f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:4px">'
    #         f'<div style="background:{_low_bg};border:1px solid {_risk_border};border-radius:10px;padding:10px 18px;text-align:center">'
    #         f'<div style="color:{_risk_label_c or "#059669"};font-size:13px;font-weight:700">Low risk</div>'
    #         f'<div style="color:{_r_val_c};font-size:18px;font-weight:800">≤ {LOW_RISK_MAX*100:.0f}%</div></div>'
    #         f'<div style="background:{_mid_bg};border:1px solid {_risk_border};border-radius:10px;padding:10px 18px;text-align:center">'
    #         f'<div style="color:{_risk_label_c or "#d97706"};font-size:13px;font-weight:700">Medium risk</div>'
    #         f'<div style="color:{_r_val_c};font-size:18px;font-weight:800">{LOW_RISK_MAX*100:.0f}–{MEDIUM_RISK_MAX*100:.0f}%</div></div>'
    #         f'<div style="background:{_hi_bg};border:1px solid {_risk_border};border-radius:10px;padding:10px 18px;text-align:center">'
    #         f'<div style="color:{_risk_label_c or "#dc2626"};font-size:13px;font-weight:700">High risk</div>'
    #         f'<div style="color:{_r_val_c};font-size:18px;font-weight:800">&gt; {MEDIUM_RISK_MAX*100:.0f}%</div></div>'
    #         f'</div>',
    #         unsafe_allow_html=True,
    #     )


if __name__ == '__main__':
    main()
