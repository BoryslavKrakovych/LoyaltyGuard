from __future__ import annotations

from collections import Counter
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.churn_model import ChurnModelService, attach_predictions, feature_importance_table
from src.clustering import build_customer_clusters
from src.data_loader import (
    infer_column_mapping,
    load_and_prepare_transactions,
    load_table_file,
)
from src.feature_engineering import (
    add_base_features,
    build_forward_churn_dataset,
    build_training_matrices,
)
from src.product_analytics import (
    cluster_products,
    customer_category_profile,
    semantic_cluster_summary,
    train_word2vec,
)
from src.retention import (
    build_individual_triggers,
    build_rescue_queue,
    compute_cooling_triggers,
    estimate_campaign_effect,
    simulate_what_if,
)
from src.rfm import build_rfm_table
from src.ui_helpers import download_dataframe_button, metric_row, plot_roc_curve

st.set_page_config(page_title='LoyaltyGuard', layout='wide')

CATEGORY_FALLBACK_LABEL = 'Невизначена категорія'
CHANNEL_FALLBACK_LABEL = 'push + e-mail'

GENERIC_CATEGORY_VALUES = {
    '',
    'unknown',
    'general merchandise',
    'загальний асортимент',
    'невизначена категорія',
    'nan',
    'none',
    'null',
    'n/a',
    '<na>',
    'інший кластер',
}

AUTO_CATEGORY_RULES = {
    'Seasonal & Holiday': [
        'christmas', 'xmas', 'santa', 'snowman', 'snowflake', 'reindeer',
        'bauble', 'stocking', 'advent', 'noel', 'holiday', 'festive'
    ],
    'Home Decor': [
        'decor', 'decoration', 'ornament', 'vase', 'frame', 'mirror',
        'clock', 'wreath', 'sign', 'plaque', 'hanger', 'hook', 'wall',
        'ceramic', 'porcelain', 'trinket', 'lantern', 'candle', 'candles',
        'holder', 'tealight', 'lamp', 'heart'
    ],
    'Kitchen & Dining': [
        'plate', 'bowl', 'dish', 'tray', 'fork', 'knife', 'spoon',
        'cutlery', 'teapot', 'saucer', 'jar', 'mug', 'cup',
        'glass', 'bottle', 'flask', 'tumbler', 'goblet', 'kitchen'
    ],
    'Textiles & Soft Furnishings': [
        'blanket', 'throw', 'towel', 'rug', 'mat', 'curtain', 'linen',
        'fabric', 'knit', 'apron', 'cloth', 'cover', 'cushion',
        'pillow', 'doormat', 'textile'
    ],
    'Bags & Accessories': [
        'bag', 'handbag', 'purse', 'wallet', 'case', 'pouch', 'backpack',
        'necklace', 'bracelet', 'ring', 'earring', 'brooch',
        'scarf', 'hat', 'glove', 'jewellery', 'jewelry'
    ],
    'Stationery & Gift': [
        'card', 'cards', 'paper', 'notebook', 'journal', 'pen', 'pencil',
        'sticker', 'craft', 'ribbon', 'wrap', 'wrapping', 'tag',
        'gift', 'present', 'party', 'celebration', 'balloon',
        'cake', 'birthday'
    ],
    'Garden & Outdoor': [
        'garden', 'plant', 'flower', 'pot', 'outdoor', 'bird',
        'feeder', 'watering', 'planter'
    ],
    'Kids & Toys': [
        'toy', 'doll', 'kids', 'child', 'children', 'baby', 'game', 'teddy'
    ],
}

OFFER_TYPE_LABELS = {
    'Auto (recommended)': 'Auto (recommended)',
    'Discount': 'Знижка',
    'Bonus points': 'Бонусні бали',
    'Personal recommendation': 'Персональна рекомендація',
    'Reminder': 'Нагадування',
    'Bundle offer': 'Комплектна пропозиція',
}

CHANNEL_LABELS = {
    'Auto (recommended)': 'Auto (recommended)',
    'SMS': 'SMS',
    'E-mail': 'E-mail',
    'Push': 'Push',
    'App': 'App',
    'SMS + e-mail': 'SMS + e-mail',
    'SMS + push': 'SMS + push',
    'Push + e-mail': 'Push + e-mail',
}


def normalize_text(text: str) -> str:
    return re.sub(r'[^a-z0-9\s]+', ' ', str(text).lower()).strip()


def is_missing_like(value) -> bool:
    if pd.isna(value):
        return True

    text = str(value).strip().lower()
    return text in {'', 'nan', 'none', 'null', 'n/a', 'na', '<na>'}


def clean_text_value(value, default: str = '') -> str:
    if is_missing_like(value):
        return default
    return str(value).strip()


def clean_category_value(value, default: str = '') -> str:
    return clean_text_value(value, default)


def is_generic_category(value) -> bool:
    text = clean_text_value(value, '').strip().lower()
    return text in GENERIC_CATEGORY_VALUES


def sorted_options(series: pd.Series) -> list[str]:
    values = []
    for value in series.tolist():
        text = clean_text_value(value, '')
        if text != '':
            values.append(text)
    return sorted(pd.Series(values).drop_duplicates().tolist())


def first_valid_category_from_text(value: str) -> str:
    if is_missing_like(value):
        return ''

    parts = []
    for item in str(value).split(','):
        cleaned = clean_text_value(item, '')
        if cleaned != '':
            parts.append(cleaned)

    if not parts:
        return ''
    return parts[0]


def build_category_profile_from_main_csv(df: pd.DataFrame) -> pd.DataFrame:
    if 'customer_id' not in df.columns:
        return pd.DataFrame(columns=['customer_id', 'top_categories', 'dominant_category'])

    category_cols = [col for col in df.columns if col.startswith('category_spend_')]
    if not category_cols:
        return pd.DataFrame(columns=['customer_id', 'top_categories', 'dominant_category'])

    work = df.copy()

    if 'transaction_date' in work.columns:
        work = work.sort_values(by=['customer_id', 'transaction_date'])

    latest = work.groupby('customer_id').tail(1).copy()
    latest[category_cols] = latest[category_cols].fillna(0)

    def get_top_categories(row: pd.Series) -> str:
        pairs = []
        for col in category_cols:
            cat_name = col.replace('category_spend_', '')
            pairs.append((cat_name, float(row[col])))

        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        non_zero = [name for name, value in pairs if value > 0][:3]

        if not non_zero:
            return ''
        return ', '.join(non_zero)

    def get_dominant_category(row: pd.Series) -> str:
        if float(row[category_cols].sum()) <= 0:
            return ''
        best_col = row[category_cols].idxmax()
        return best_col.replace('category_spend_', '')

    latest['top_categories'] = latest.apply(get_top_categories, axis=1)
    latest['dominant_category'] = latest.apply(get_dominant_category, axis=1)

    result_cols = ['customer_id', 'top_categories', 'dominant_category'] + category_cols
    return latest[result_cols].copy()


def build_category_clusters_from_main_csv(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    category_profile = build_category_profile_from_main_csv(df)
    category_cols = [col for col in category_profile.columns if col.startswith('category_spend_')]

    if len(category_profile) == 0 or len(category_cols) < 2:
        return pd.DataFrame(columns=['customer_id', 'category_cluster', 'top_categories', 'dominant_category'])

    model_df = category_profile[['customer_id', 'top_categories', 'dominant_category'] + category_cols].copy()
    model_df[category_cols] = model_df[category_cols].fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(model_df[category_cols])

    n_clusters = max(2, min(n_clusters, len(model_df)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model_df['category_cluster'] = kmeans.fit_predict(X)

    return model_df[['customer_id', 'category_cluster', 'top_categories', 'dominant_category']]


def infer_category_from_text(text: str) -> str:
    tokens = set(re.findall(r'[a-z0-9]+', str(text).lower()))

    if not tokens:
        return CATEGORY_FALLBACK_LABEL

    best_category = CATEGORY_FALLBACK_LABEL
    best_score = 0

    for category_name, keywords in AUTO_CATEGORY_RULES.items():
        score = len(tokens.intersection(set(keywords)))
        if score > best_score:
            best_score = score
            best_category = category_name

    return best_category


def assign_auto_categories(products: pd.DataFrame) -> pd.DataFrame:
    if products is None or products.empty:
        return pd.DataFrame(columns=['customer_id', 'product_id', 'product_name', 'product_description', 'category'])

    work = products.copy()

    text_series = (
        work.get('product_name', pd.Series('', index=work.index)).fillna('').astype(str)
        + ' '
        + work.get('product_description', pd.Series('', index=work.index)).fillna('').astype(str)
    ).str.strip()

    work['category'] = text_series.apply(infer_category_from_text)
    work['category'] = work['category'].apply(lambda x: clean_category_value(x, CATEGORY_FALLBACK_LABEL))
    return work


def build_products_for_word2vec(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=['customer_id', 'product_id', 'product_name', 'product_description'])

    if hasattr(uploaded_file, 'seek'):
        uploaded_file.seek(0)

    raw_df = load_table_file(uploaded_file)
    detected_mapping = infer_column_mapping(raw_df.columns.tolist())

    rename_map = {}
    for canonical_name in ['customer_id', 'product_id', 'product_name']:
        source_name = detected_mapping.get(canonical_name)
        if source_name and source_name in raw_df.columns and canonical_name not in raw_df.columns:
            rename_map[source_name] = canonical_name

    products = raw_df.rename(columns=rename_map).copy()

    if 'customer_id' not in products.columns or 'product_name' not in products.columns:
        return pd.DataFrame(columns=['customer_id', 'product_id', 'product_name', 'product_description'])

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

    keep_cols = ['customer_id', 'product_id', 'product_name', 'product_description']
    products = products[keep_cols].copy()
    products = products[
        products['customer_id'].notna() &
        (products['product_name'] != '')
    ].drop_duplicates().reset_index(drop=True)

    return products


def build_customer_semantic_profile(product_clusters: pd.DataFrame) -> pd.DataFrame:
    if (
        product_clusters is None
        or product_clusters.empty
        or 'customer_id' not in product_clusters.columns
        or 'semantic_cluster_name' not in product_clusters.columns
    ):
        return pd.DataFrame(columns=['customer_id', 'top_categories_semantic', 'dominant_category_semantic'])

    work = product_clusters.copy()
    work['semantic_cluster_name'] = work['semantic_cluster_name'].apply(lambda x: clean_text_value(x, ''))
    work = work[work['semantic_cluster_name'] != ''].copy()

    if len(work) == 0:
        return pd.DataFrame(columns=['customer_id', 'top_categories_semantic', 'dominant_category_semantic'])

    def top_three(values: list[str]) -> str:
        counter = Counter([v for v in values if clean_text_value(v, '') != ''])
        if not counter:
            return ''
        return ', '.join([name for name, _ in counter.most_common(3)])

    grouped = work.groupby('customer_id')['semantic_cluster_name'].apply(list).reset_index()
    grouped['top_categories_semantic'] = grouped['semantic_cluster_name'].apply(top_three)
    grouped['dominant_category_semantic'] = grouped['top_categories_semantic'].apply(first_valid_category_from_text)

    return grouped[['customer_id', 'top_categories_semantic', 'dominant_category_semantic']]


def choose_best_category(row: pd.Series) -> str:
    candidates = [
        row.get('suggested_category', ''),
        row.get('dominant_category', ''),
        first_valid_category_from_text(row.get('top_categories', '')),
        row.get('dominant_category_products', ''),
        first_valid_category_from_text(row.get('top_categories_products', '')),
        row.get('dominant_category_semantic', ''),
        first_valid_category_from_text(row.get('top_categories_semantic', '')),
    ]

    for value in candidates:
        text = clean_category_value(value, '')
        if text != '' and not is_generic_category(text):
            return text

    for value in candidates:
        text = clean_category_value(value, '')
        if text != '':
            return text

    return CATEGORY_FALLBACK_LABEL


def choose_best_top_categories(row: pd.Series) -> str:
    candidates = [
        row.get('top_categories', ''),
        row.get('top_categories_products', ''),
        row.get('top_categories_semantic', ''),
    ]

    for value in candidates:
        if is_missing_like(value):
            continue

        parts = []
        for item in str(value).split(','):
            cleaned = clean_text_value(item, '')
            if cleaned != '':
                parts.append(cleaned)

        non_generic = [x for x in parts if not is_generic_category(x)]
        if non_generic:
            return ', '.join(non_generic[:3])

        if parts:
            return ', '.join(parts[:3])

    return CATEGORY_FALLBACK_LABEL


def build_auto_recommended_action(row: pd.Series) -> str:
    risk = clean_text_value(row.get('risk_class', ''), '')
    segment = clean_text_value(row.get('rfm_segment', ''), '')
    cooling = bool(row.get('cooling_flag', False))
    category_name = choose_best_category(row)

    if risk == 'High' and 'At Risk' in segment and cooling:
        return f'Сильна retention-пропозиція + персональний контакт по категорії {category_name}'
    if risk == 'High':
        return f'Точкова знижка + нагадування по категорії {category_name}'
    if risk == 'Medium' and cooling:
        return f'М’яка акція + персональна рекомендація по категорії {category_name}'
    if risk == 'Medium':
        return 'М’яке нагадування + добірка релевантних товарів'
    return 'Лояльність / контентне нагадування'


def build_auto_recommended_channel(row: pd.Series) -> str:
    risk = clean_text_value(row.get('risk_class', ''), '')
    cooling = bool(row.get('cooling_flag', False))

    if risk == 'High' and cooling:
        return 'SMS + push'
    if risk == 'High':
        return 'SMS + e-mail'
    if risk == 'Medium':
        return 'Push + e-mail'
    return 'App'


def build_manual_action_text(offer_type: str, category_name: str) -> str:
    category_name = clean_category_value(category_name, CATEGORY_FALLBACK_LABEL)

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

    return f'Комунікація по категорії {category_name}'


def main():
    st.title('LoyaltyGuard')
    st.subheader('Система прогнозу відтоку клієнтів та підтримки рішень для retention-кампаній у рітейлі')

    with st.sidebar:
        st.header('Дані')
        transactions_file = st.file_uploader(
            'Завантажте CSV або XLSX',
            type=['csv', 'xlsx', 'xls']
        )

    if transactions_file is None:
        st.info('Спочатку завантажте CSV або XLSX з транзакціями клієнтів.')
        st.stop()

    df, _, notes = load_and_prepare_transactions(transactions_file)

    products_for_word2vec = build_products_for_word2vec(transactions_file)
    products_for_word2vec = assign_auto_categories(products_for_word2vec)

    word2vec_model = train_word2vec(products_for_word2vec)
    product_clusters = cluster_products(products_for_word2vec, word2vec_model)
    cluster_summary = semantic_cluster_summary(product_clusters)
    product_category_profile = customer_category_profile(products_for_word2vec)
    semantic_profile = build_customer_semantic_profile(product_clusters)

    required_after_mapping = ['customer_id', 'transaction_date']
    missing = [col for col in required_after_mapping if col not in df.columns]

    if missing:
        st.error(
            'Не вдалося автоматично привести файл до потрібного формату. '
            f'Не вистачає колонок після розпізнавання: {missing}'
        )
        st.write('Колонки у файлі:', list(df.columns))
        st.stop()

    featured = add_base_features(df)

    horizon_days = 90
    train_featured = build_forward_churn_dataset(featured, horizon_days=horizon_days)

    if len(train_featured) == 0:
        st.error('Не вдалося побудувати train set для forward-looking churn. Замало даних після відсікання horizon window.')
        st.stop()

    if train_featured['churn'].nunique() < 2:
        st.error('У forward-looking target вийшов лише один клас churn. Спробуйте інший horizon_days.')
        st.stop()

    X_train, y, groups, _ = build_training_matrices(train_featured)

    churn_service = ChurnModelService(use_xgboost=False)
    churn_artifacts = churn_service.train(X_train, y, groups)

    featured['is_currently_active'] = featured['days_since_last_purchase'] <= horizon_days

    X_all, _, _, _ = build_training_matrices(featured.assign(churn=0))
    predicted = attach_predictions(featured, X_all, churn_artifacts)

    rfm = build_rfm_table(featured)
    predicted = predicted.merge(
        rfm[['customer_id', 'rfm_segment', 'RFM_score']],
        on='customer_id',
        how='left'
    )

    customer_clusters = build_customer_clusters(predicted)
    if len(customer_clusters) > 0:
        predicted = predicted.merge(customer_clusters, on='customer_id', how='left')

    category_profile_main = build_category_profile_from_main_csv(predicted)
    if len(category_profile_main) > 0:
        predicted = predicted.merge(
            category_profile_main[['customer_id', 'top_categories', 'dominant_category']],
            on='customer_id',
            how='left'
        )
    else:
        predicted['top_categories'] = ''
        predicted['dominant_category'] = ''

    if len(product_category_profile) > 0:
        profile_products = product_category_profile.rename(columns={
            'top_categories': 'top_categories_products',
            'dominant_category': 'dominant_category_products',
        })
        predicted = predicted.merge(
            profile_products[['customer_id', 'top_categories_products', 'dominant_category_products']],
            on='customer_id',
            how='left'
        )
    else:
        predicted['top_categories_products'] = ''
        predicted['dominant_category_products'] = ''

    if len(semantic_profile) > 0:
        predicted = predicted.merge(
            semantic_profile[['customer_id', 'top_categories_semantic', 'dominant_category_semantic']],
            on='customer_id',
            how='left'
        )
    else:
        predicted['top_categories_semantic'] = ''
        predicted['dominant_category_semantic'] = ''

    cooling = compute_cooling_triggers(predicted)
    if len(cooling) > 0:
        predicted = predicted.merge(cooling, on='customer_id', how='left')
    else:
        predicted['cooling_flag'] = False
        predicted['cooling_score'] = 0
        predicted['basket_drop_flag'] = False
        predicted['items_drop_flag'] = False
        predicted['frequency_drop_flag'] = False

    for col in ['cooling_flag', 'basket_drop_flag', 'items_drop_flag', 'frequency_drop_flag']:
        if col in predicted.columns:
            predicted[col] = predicted[col].fillna(False)

    if 'cooling_score' in predicted.columns:
        predicted['cooling_score'] = pd.to_numeric(predicted['cooling_score'], errors='coerce').fillna(0).astype(int)

    trigger_df = build_individual_triggers(predicted)

    latest_customer_state = predicted.groupby('customer_id').tail(1).copy()
    latest_customer_state = latest_customer_state.merge(
        trigger_df,
        on='customer_id',
        how='left',
        suffixes=('', '_trigger')
    )

    latest_customer_state['resolved_category'] = latest_customer_state.apply(choose_best_category, axis=1)
    latest_customer_state['top_categories_display'] = latest_customer_state.apply(choose_best_top_categories, axis=1)
    latest_customer_state['dominant_category_display'] = latest_customer_state['resolved_category']

    if 'trigger_reason' in latest_customer_state.columns:
        latest_customer_state['trigger_reason'] = latest_customer_state['trigger_reason'].apply(
            lambda x: clean_text_value(x, '')
        )
    else:
        latest_customer_state['trigger_reason'] = ''

    latest_customer_state['recommended_action'] = latest_customer_state.apply(
        build_auto_recommended_action,
        axis=1
    )
    latest_customer_state['recommended_channel'] = latest_customer_state.apply(
        build_auto_recommended_channel,
        axis=1
    )

    latest_customer_state['dominant_category'] = latest_customer_state['dominant_category_display']
    latest_customer_state['top_categories'] = latest_customer_state['top_categories_display']

    rescue_queue = build_rescue_queue(latest_customer_state)
    if 'recommended_action' not in rescue_queue.columns:
        rescue_queue = rescue_queue.merge(
            latest_customer_state[['customer_id', 'recommended_action', 'recommended_channel']],
            on='customer_id',
            how='left'
        )

    active_customers = latest_customer_state[latest_customer_state['is_currently_active']].copy()
    top_risk_active = (
        active_customers[['customer_id', 'churn_probability_percent']]
        .sort_values(by='churn_probability_percent', ascending=False)
        .head(10)
    )

    tabs = st.tabs([
        '1. Overview',
        '2. Churn model',
        '3. RFM',
        '4. Categories',
        '5. Customer clusters',
        '6. Rescue queue',
        '7. What-if',
        '8. Campaign builder',
    ])

    with tabs[0]:
        st.markdown('### Огляд')
        c1, c2, c3 = st.columns(3)

        metric_row(
            c1, 'Клієнтів', predicted['customer_id'].nunique(),
            c2, 'Рядків у датасеті', len(predicted),
            c3, 'ROC-AUC', f"{churn_artifacts.roc_auc:.4f}",
        )

        if notes:
            st.markdown('### Нотатки автопідготовки')
            for note in notes:
                st.write(f'- {note}')

        st.markdown('### Розподіл клієнтів за ризиком')
        risk_dist = latest_customer_state['risk_class'].value_counts().reset_index()
        risk_dist.columns = ['risk_class', 'customers']
        st.bar_chart(risk_dist.set_index('risk_class'))

        st.markdown('### ТОП-10 активних клієнтів з найвищим ризиком')
        st.dataframe(top_risk_active, use_container_width=True)

        st.markdown('### Перші рядки даних після автопідготовки')
        st.dataframe(predicted.head(20), use_container_width=True)

    with tabs[1]:
        st.markdown('### Churn model')

        plot_roc_curve(
            churn_artifacts.fpr,
            churn_artifacts.tpr,
            churn_artifacts.roc_auc
        )

        st.markdown('### Feature importance')
        importance = feature_importance_table(churn_artifacts).head(20)
        st.dataframe(importance, use_container_width=True)

        st.markdown('### ТОП-10 з найвищим ризиком')
        st.dataframe(top_risk_active, use_container_width=True)

        download_dataframe_button(top_risk_active, 'top_risk_active.csv', 'Завантажити TOP-10')

    with tabs[2]:
        st.markdown('### RFM-аналіз')
        st.dataframe(rfm.head(30), use_container_width=True)

        seg_counts = rfm['rfm_segment'].value_counts().reset_index()
        seg_counts.columns = ['segment', 'customers']
        st.bar_chart(seg_counts.set_index('segment'))

        download_dataframe_button(rfm, 'rfm_table.csv', 'Завантажити RFM таблицю')

    with tabs[3]:
        st.markdown('### Категорії та товарна орієнтація')
        st.write('Якщо в датасеті є колонки виду category_spend_*, вони будуть використані автоматично.')

        category_profile_main = build_category_profile_from_main_csv(predicted)
        category_clusters = build_category_clusters_from_main_csv(predicted)

        st.markdown('### Узгоджений профіль категорій клієнтів')
        display_cols = [
            'customer_id',
            'dominant_category_display',
            'top_categories_display',
            'customer_cluster_name',
        ]
        display_cols = [col for col in display_cols if col in latest_customer_state.columns]
        st.dataframe(latest_customer_state[display_cols].head(50), use_container_width=True)

        dominant_counts_resolved = latest_customer_state['dominant_category_display'].value_counts()
        st.markdown('### Найчастіша домінуюча категорія')
        st.bar_chart(dominant_counts_resolved)

        if len(category_profile_main) > 0:
            st.markdown('### Профіль категорій з category_spend_*')
            st.dataframe(
                category_profile_main[['customer_id', 'top_categories', 'dominant_category']].head(50),
                use_container_width=True
            )

            if len(category_clusters) > 0:
                st.markdown('### Кластеризація клієнтів по категоріях')
                st.dataframe(category_clusters.head(50), use_container_width=True)

                cluster_counts = category_clusters['category_cluster'].value_counts().sort_index()
                st.bar_chart(cluster_counts)
        else:
            if len(product_category_profile) > 0:
                display_profile = product_category_profile.copy()
                display_profile['top_categories'] = display_profile['top_categories'].apply(
                    lambda x: clean_text_value(x, CATEGORY_FALLBACK_LABEL)
                )
                display_profile['dominant_category'] = display_profile['dominant_category'].apply(
                    lambda x: clean_text_value(x, CATEGORY_FALLBACK_LABEL)
                )

                st.markdown('### Профіль категорій клієнтів з автокатегоризації товарів')
                st.dataframe(
                    display_profile[['customer_id', 'top_categories', 'dominant_category', 'purchased_categories_count']].head(50),
                    use_container_width=True
                )

            st.info('Колонки category_spend_* не знайдені. Тому нижче показано fallback на Word2Vec і автоматичні категорії з назв товарів.')

            if word2vec_model is None or len(product_clusters) == 0:
                st.warning(
                    'Не вдалося побудувати Word2Vec-кластери. '
                    'Перевірте, що встановлено gensim і що у файлі достатньо назв товарів.'
                )
            else:
                st.markdown('### Семантичні кластери товарів (Word2Vec)')
                summary_cols = [
                    col for col in ['semantic_cluster_name', 'products_count', 'dominant_category', 'example_products']
                    if col in cluster_summary.columns
                ]
                st.dataframe(cluster_summary[summary_cols], use_container_width=True)

                if {'semantic_cluster_name', 'products_count'}.issubset(cluster_summary.columns):
                    cluster_counts = (
                        cluster_summary[['semantic_cluster_name', 'products_count']]
                        .set_index('semantic_cluster_name')
                    )
                    st.bar_chart(cluster_counts)

                st.markdown('### Графік розбиття семантичних кластерів')
                plot_df = product_clusters[['pca_x', 'pca_y', 'semantic_cluster_name']].dropna().copy()

                if len(plot_df) > 4000:
                    plot_df = plot_df.sample(4000, random_state=42)

                fig, ax = plt.subplots(figsize=(10, 6))
                for cluster_name, group in plot_df.groupby('semantic_cluster_name'):
                    ax.scatter(group['pca_x'], group['pca_y'], label=cluster_name, alpha=0.6)

                ax.set_xlabel('PCA 1')
                ax.set_ylabel('PCA 2')
                ax.set_title('Розбиття товарів на семантичні кластери')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)

                st.markdown('### Приклади товарів по кластерах')
                show_cols = [col for col in ['product_id', 'product_name', 'category', 'semantic_cluster_name'] if col in product_clusters.columns]
                st.dataframe(product_clusters[show_cols].head(100), use_container_width=True)

                st.markdown('### Розподіл автокатегорій')
                if 'category' in product_clusters.columns:
                    category_counts = product_clusters['category'].value_counts()
                    st.bar_chart(category_counts)

    with tabs[4]:
        st.markdown('### Кластеризація клієнтів')
        if len(customer_clusters) == 0:
            st.warning('Замало поведінкових колонок для K-Means.')
        else:
            show_cols = ['customer_id', 'customer_cluster_name']
            show_cols = [col for col in show_cols if col in customer_clusters.columns]

            st.dataframe(customer_clusters[show_cols].head(50), use_container_width=True)

            cluster_counts = customer_clusters['customer_cluster_name'].value_counts()
            st.bar_chart(cluster_counts)

    with tabs[5]:
        st.markdown('### Черга на порятунок')
        min_risk = st.slider('Мінімальна ймовірність відтоку (%)', 0, 100, 40)

        filtered_queue = rescue_queue[rescue_queue['churn_probability_percent'] >= min_risk].copy()

        if 'cooling_flag' in filtered_queue.columns:
            only_cooling = st.checkbox('Показати лише тих, у кого є cooling trigger')
            if only_cooling:
                filtered_queue = filtered_queue[filtered_queue['cooling_flag'] == True]

        display_cols = [
            'customer_id',
            'churn_probability_percent',
            'risk_class',
            'rfm_segment',
            'customer_cluster_name',
            'dominant_category',
            'top_categories',
            'cooling_flag',
            'cooling_score',
            'recommended_action',
            'recommended_channel',
        ]
        display_cols = [col for col in display_cols if col in filtered_queue.columns]

        st.dataframe(filtered_queue[display_cols].head(100), use_container_width=True)
        download_dataframe_button(filtered_queue[display_cols], 'rescue_queue.csv', 'Завантажити rescue queue')

    with tabs[6]:
        st.markdown('### What-if симулятор')

        latest = latest_customer_state.copy()
        latest = latest[latest['is_currently_active']].copy()
        latest = latest.sort_values('churn_probability_percent', ascending=False)

        if len(latest) == 0:
            st.info('Немає активних клієнтів для симулятора.')
        else:
            customer_ids = latest['customer_id'].astype(str).tolist()
            selected_customer_id = st.selectbox('Оберіть клієнта', customer_ids)

            selected_row = latest[latest['customer_id'].astype(str) == selected_customer_id].iloc[0]

            freq_change = st.slider('Покращення частоти замовлень', 0.0, 1.0, 0.2, 0.05)
            ticket_change = st.slider('Покращення середнього чека', 0.0, 1.0, 0.1, 0.05)
            promo_change = st.slider('Покращення реакції на акції', 0.0, 1.0, 0.1, 0.05)

            scenario_df = simulate_what_if(
                selected_row,
                X_all.loc[selected_row.name],
                churn_artifacts,
                freq_change,
                ticket_change,
                promo_change
            )

            st.dataframe(scenario_df, use_container_width=True)
            st.line_chart(scenario_df.set_index('scenario'))

    with tabs[7]:
        st.markdown('### Конструктор кампаній')

        latest = latest_customer_state.copy()

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            risk_options = ['All'] + sorted_options(latest['risk_class'])
            selected_risk = st.selectbox('Фільтр по ризику', risk_options)

            category_options = ['All'] + sorted_options(latest['dominant_category_display'])
            selected_category_filter = st.selectbox('Фільтр по категорії', category_options)

        with filter_col2:
            segment_options = ['All'] + sorted_options(latest['rfm_segment'])
            selected_segment = st.selectbox('Фільтр по RFM-сегменту', segment_options)

            cluster_options = ['All'] + sorted_options(latest['customer_cluster_name'])
            selected_cluster = st.selectbox('Фільтр по кластеру клієнтів', cluster_options)

        flag_col1, flag_col2 = st.columns(2)
        with flag_col1:
            only_active = st.checkbox('Лише активні клієнти', value=True)
        with flag_col2:
            only_cooling = st.checkbox('Лише з cooling trigger', value=False)

        audience = latest.copy()

        if only_active and 'is_currently_active' in audience.columns:
            audience = audience[audience['is_currently_active'] == True]

        if selected_risk != 'All':
            audience = audience[audience['risk_class'] == selected_risk]

        if selected_segment != 'All':
            audience = audience[audience['rfm_segment'] == selected_segment]

        if selected_category_filter != 'All':
            audience = audience[audience['dominant_category_display'] == selected_category_filter]

        if selected_cluster != 'All':
            audience = audience[audience['customer_cluster_name'] == selected_cluster]

        if only_cooling and 'cooling_flag' in audience.columns:
            audience = audience[audience['cooling_flag'] == True]

        audience = audience.sort_values('churn_probability_percent', ascending=False).copy()

        st.markdown('### Налаштування кампанії')
        setup_col1, setup_col2 = st.columns(2)

        with setup_col1:
            campaign_name = st.text_input('Назва кампанії', value='Retention campaign')
            offer_type = st.selectbox(
                'Тип оферу',
                list(OFFER_TYPE_LABELS.keys()),
                format_func=lambda x: OFFER_TYPE_LABELS[x]
            )
            offer_category_mode = st.selectbox(
                'Категорія оферу',
                ['Auto (from customer profile)', 'Manual'],
                format_func=lambda x: 'Авто з профілю клієнта' if x == 'Auto (from customer profile)' else 'Вручну'
            )
            manual_offer_category = st.text_input('Ручна категорія оферу', value='')

        with setup_col2:
            channel_mode = st.selectbox(
                'Канал кампанії',
                list(CHANNEL_LABELS.keys()),
                format_func=lambda x: CHANNEL_LABELS[x]
            )
            campaign_description = st.text_area(
                'Опис кампанії',
                value='Кампанія для утримання клієнтів із підвищеним ризиком відтоку.'
            )
            audience_limit = st.slider('Скільки рядків показувати в таблицях', 10, 500, 100, 10)

        st.markdown('### Припущення для оцінки ефекту')
        effect_col1, effect_col2, effect_col3 = st.columns(3)
        with effect_col1:
            freq_change = st.slider('Покращення частоти', 0.0, 1.0, 0.15, 0.05, key='campaign_freq_change')
        with effect_col2:
            ticket_change = st.slider('Покращення середнього чека', 0.0, 1.0, 0.10, 0.05, key='campaign_ticket_change')
        with effect_col3:
            promo_change = st.slider('Покращення реакції на акції', 0.0, 1.0, 0.10, 0.05, key='campaign_promo_change')

        st.write(f'Розмір аудиторії: {len(audience)}')

        if len(audience) == 0:
            st.warning('За поточними фільтрами аудиторія порожня.')
        else:
            audience = audience.copy()
            audience_X = X_all.loc[audience.index].copy()

            summary_df, effect_details = estimate_campaign_effect(
                audience,
                audience_X,
                churn_artifacts,
                freq_change,
                ticket_change,
                promo_change
            )

            audience['campaign_category'] = audience['resolved_category']
            if offer_category_mode == 'Manual' and manual_offer_category.strip():
                audience['campaign_category'] = clean_text_value(manual_offer_category.strip(), CATEGORY_FALLBACK_LABEL)

            audience['recommended_action'] = audience.apply(build_auto_recommended_action, axis=1)
            audience['recommended_channel'] = audience.apply(build_auto_recommended_channel, axis=1)

            if offer_type == 'Auto (recommended)':
                audience['final_action'] = audience['recommended_action']
            else:
                audience['final_action'] = audience['campaign_category'].apply(
                    lambda x: build_manual_action_text(offer_type, x)
                )

            if channel_mode == 'Auto (recommended)':
                audience['final_channel'] = audience['recommended_channel']
            else:
                audience['final_channel'] = channel_mode

            audience['final_action'] = audience['final_action'].apply(
                lambda x: clean_text_value(x, f'Комунікація по категорії {CATEGORY_FALLBACK_LABEL}')
            )
            audience['final_channel'] = audience['final_channel'].apply(
                lambda x: clean_text_value(x, CHANNEL_FALLBACK_LABEL)
            )

            campaign_table = audience.merge(
                effect_details,
                on='customer_id',
                how='left'
            )

            campaign_table['campaign_name'] = campaign_name.strip() or 'Retention campaign'
            campaign_table['campaign_description'] = campaign_description.strip()
            campaign_table['offer_type'] = OFFER_TYPE_LABELS.get(offer_type, offer_type)
            campaign_table['dominant_category'] = campaign_table['dominant_category_display']
            campaign_table['top_categories'] = campaign_table['top_categories_display']

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric('Аудиторія', int(summary_df.iloc[0]['audience_size']))
            with metric_col2:
                st.metric('Сер. ризик до, %', summary_df.iloc[0]['avg_risk_before_pct'])
            with metric_col3:
                st.metric('Сер. зниження, p.p.', summary_df.iloc[0]['avg_reduction_pp'])

            metric_col4, metric_col5, metric_col6 = st.columns(3)
            with metric_col4:
                st.metric('Сер. ризик після, %', summary_df.iloc[0]['avg_risk_after_pct'])
            with metric_col5:
                st.metric('Клієнтів з покращенням', int(summary_df.iloc[0]['customers_improved']))
            with metric_col6:
                st.metric(
                    'High risk: до -> після',
                    f"{int(summary_df.iloc[0]['high_risk_before'])} -> {int(summary_df.iloc[0]['high_risk_after'])}"
                )

            st.markdown('### Розподіл аудиторії за customer clusters')
            cluster_dist = campaign_table['customer_cluster_name'].value_counts().reset_index()
            cluster_dist.columns = ['customer_cluster_name', 'customers']
            st.dataframe(cluster_dist, use_container_width=True)

            st.markdown('### Підсумок кампанії')
            st.dataframe(summary_df, use_container_width=True)

            st.markdown('### Деталі оцінки ефекту')
            st.dataframe(effect_details.head(audience_limit), use_container_width=True)

            st.markdown('### Фінальна таблиця кампанії')
            show_cols = [
                'campaign_name',
                'customer_id',
                'risk_class',
                'rfm_segment',
                'customer_cluster_name',
                'dominant_category',
                'campaign_category',
                'top_categories',
                'cooling_flag',
                'trigger_reason',
                'recommended_action',
                'recommended_channel',
                'final_action',
                'final_channel',
                'churn_probability_percent',
                'probability_before_pct',
                'probability_after_pct',
                'reduction_pp',
            ]
            show_cols = [col for col in show_cols if col in campaign_table.columns]

            st.dataframe(campaign_table[show_cols].head(audience_limit), use_container_width=True)

            audience_export_cols = [
                'customer_id',
                'risk_class',
                'rfm_segment',
                'customer_cluster_name',
                'dominant_category_display',
                'resolved_category',
                'top_categories_display',
                'cooling_flag',
                'trigger_reason',
                'recommended_action',
                'recommended_channel',
                'campaign_category',
                'final_action',
                'final_channel',
                'churn_probability_percent',
            ]
            audience_export_cols = [col for col in audience_export_cols if col in audience.columns]

            download_dataframe_button(
                campaign_table[show_cols],
                'campaign_builder_output.csv',
                'Завантажити фінальну кампанію'
            )
            download_dataframe_button(
                effect_details,
                'campaign_effect_details.csv',
                'Завантажити оцінку ефекту'
            )
            download_dataframe_button(
                audience[audience_export_cols],
                'campaign_audience.csv',
                'Завантажити аудиторію кампанії'
            )


if __name__ == '__main__':
    main()