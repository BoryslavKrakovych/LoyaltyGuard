from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

import numpy as np
import pandas as pd
import streamlit as st


COLUMN_ALIASES = {
    'customer_id': [
        'customer_id', 'customer id', 'customerid',
        'client_id', 'client id',
        'userid', 'user_id',
        'cust_id', 'custid',
        'customer'
    ],
    'transaction_date': [
        'transaction_date', 'transaction date', 'transactiondate',
        'invoice_date', 'invoice date', 'invoicedate',
        'order_date', 'order date', 'orderdate',
        'purchase_date', 'purchase date', 'purchasedate',
        'date', 'datetime'
    ],
    'transaction_id': [
        'transaction_id', 'transaction id', 'transactionid',
        'invoice', 'invoice_no', 'invoice no', 'invoiceno',
        'order_id', 'order id', 'orderid',
        'bill_no', 'bill no',
        'receipt_id', 'receipt id'
    ],
    'product_id': [
        'product_id', 'product id', 'productid',
        'stockcode', 'stock_code',
        'sku',
        'item_id', 'item id'
    ],
    'product_name': [
        'product_name', 'product name', 'productname',
        'description',
        'item_name', 'item name',
        'name', 'title'
    ],
    'quantity': [
        'quantity', 'qty', 'items', 'units', 'count'
    ],
    'price': [
        'price',
        'unit_price', 'unit price', 'unitprice',
        'item_price', 'item price',
        'amount'
    ],
    'country': [
        'country', 'region'
    ],
    'churn': [
        'churn', 'is_churn', 'target', 'label'
    ],
}


def _normalize_col_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(name).strip().lower())


def _clean_id_series(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    cleaned = cleaned.where(cleaned.notna(), np.nan)
    cleaned = cleaned.astype('string')
    cleaned = cleaned.str.replace(r'\.0$', '', regex=True).str.strip()
    cleaned = cleaned.replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, '<NA>': pd.NA})
    return cleaned


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def infer_column_mapping(columns: list[str]) -> dict[str, str]:
    normalized_to_original = {
        _normalize_col_name(col): col for col in columns
    }

    mapping: dict[str, str] = {}

    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_key = _normalize_col_name(alias)
            if alias_key in normalized_to_original:
                mapping[canonical_name] = normalized_to_original[alias_key]
                break

    return mapping


@st.cache_data(show_spinner=False)
def load_csv_file(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def load_csv_path(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_table_file(uploaded_file) -> pd.DataFrame:
    file_name = getattr(uploaded_file, 'name', '').lower()

    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        excel_file = pd.ExcelFile(uploaded_file)
        frames = []

        for sheet_name in excel_file.sheet_names:
            part = pd.read_excel(excel_file, sheet_name=sheet_name)
            part['source_sheet'] = sheet_name
            frames.append(part)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    return pd.read_csv(uploaded_file)


def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ['transaction_date', 'last_purchase_date', 'invoice_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


def align_columns_for_inference(X_reference: pd.DataFrame, X_new: pd.DataFrame) -> pd.DataFrame:
    reference_columns = list(X_reference.columns)
    aligned = X_new.copy()

    for column in reference_columns:
        if column not in aligned.columns:
            aligned[column] = 0

    aligned = aligned.reindex(columns=reference_columns, fill_value=0)

    object_columns = aligned.select_dtypes(include=['object', 'string', 'category']).columns
    if len(object_columns) > 0:
        aligned[object_columns] = aligned[object_columns].fillna('')

    numeric_columns = aligned.columns.difference(object_columns)
    if len(numeric_columns) > 0:
        aligned[numeric_columns] = aligned[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    return aligned


def ensure_columns(df: pd.DataFrame, required_columns: list[str]) -> tuple[bool, list[str]]:
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


def _normalize_existing_churn(series: pd.Series) -> pd.Series:
    if series.dtype.kind in 'biufc':
        return pd.to_numeric(series, errors='coerce').fillna(0).astype(int).clip(0, 1)

    normalized = series.astype(str).str.strip().str.lower()

    true_values = {'1', 'true', 'yes', 'y', 'churn', 'churned', 'lost'}
    false_values = {'0', 'false', 'no', 'n', 'active', 'retained', 'loyal'}

    result = pd.Series(np.nan, index=series.index)

    result[normalized.isin(true_values)] = 1
    result[normalized.isin(false_values)] = 0

    numeric = pd.to_numeric(normalized, errors='coerce')
    result = result.fillna(numeric)

    return result.fillna(0).astype(int).clip(0, 1)


def _build_churn_from_recency(df: pd.DataFrame) -> pd.DataFrame:
    if 'customer_id' not in df.columns or 'transaction_date' not in df.columns:
        df = df.copy()
        df['churn'] = 0
        return df

    work = df.copy()
    work['transaction_date'] = pd.to_datetime(work['transaction_date'], errors='coerce')
    work = work.dropna(subset=['customer_id', 'transaction_date'])

    if len(work) == 0:
        df = df.copy()
        df['churn'] = 0
        return df

    latest = work.groupby('customer_id')['transaction_date'].max().reset_index()
    snapshot_date = latest['transaction_date'].max() + pd.Timedelta(days=1)

    latest['recency_days'] = (snapshot_date - latest['transaction_date']).dt.days
    threshold = max(30, int(latest['recency_days'].quantile(0.75)))
    latest['churn'] = (latest['recency_days'] > threshold).astype(int)

    if latest['churn'].nunique() < 2:
        threshold = max(15, int(latest['recency_days'].median()))
        latest['churn'] = (latest['recency_days'] > threshold).astype(int)

    if latest['churn'].nunique() < 2 and len(latest) >= 2:
        latest = latest.sort_values('recency_days').reset_index(drop=True)
        latest['churn'] = 0
        split_idx = max(1, int(len(latest) * 0.7))
        latest.loc[split_idx:, 'churn'] = 1

    result = df.merge(
        latest[['customer_id', 'churn']],
        on='customer_id',
        how='left'
    )
    result['churn'] = result['churn'].fillna(0).astype(int)
    return result


def prepare_transactions_dataframe(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    df = raw_df.copy()
    detected_mapping = infer_column_mapping(df.columns.tolist())
    notes: list[str] = []

    rename_map = {}
    for canonical_name, source_name in detected_mapping.items():
        if source_name in df.columns and canonical_name not in df.columns:
            rename_map[source_name] = canonical_name

    df = df.rename(columns=rename_map)

    for id_col in ['customer_id', 'transaction_id', 'product_id']:
        if id_col in df.columns:
            df[id_col] = _clean_id_series(df[id_col])

    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')

    if 'customer_id' in df.columns:
        before = len(df)
        df = df[df['customer_id'].notna()].copy()
        removed = before - len(df)
        if removed > 0:
            notes.append(f'видалено {removed} рядків без customer_id')

    if 'transaction_date' in df.columns:
        before = len(df)
        df = df[df['transaction_date'].notna()].copy()
        removed = before - len(df)
        if removed > 0:
            notes.append(f'видалено {removed} рядків без transaction_date')

    if 'quantity' in df.columns:
        df['quantity'] = _to_numeric(df['quantity'])

    if 'price' in df.columns:
        df['price'] = _to_numeric(df['price'])

    line_item_like = (
        'transaction_id' in df.columns and
        ('quantity' in df.columns or 'product_id' in df.columns or 'product_name' in df.columns)
    )

    if line_item_like:
        group_cols = ['customer_id', 'transaction_id']
        if 'transaction_date' in df.columns:
            group_cols.append('transaction_date')

        agg_map: dict[str, str] = {}

        if 'quantity' in df.columns:
            agg_map['quantity'] = 'sum'

        if 'price' in df.columns and 'quantity' in df.columns:
            df['line_total'] = df['quantity'].fillna(0) * df['price'].fillna(0)
            agg_map['line_total'] = 'sum'
            notes.append('total_sales побудовано як Quantity × Price')

        if 'product_id' in df.columns:
            agg_map['product_id'] = 'nunique'

        if 'country' in df.columns:
            agg_map['country'] = 'first'

        if 'source_sheet' in df.columns:
            agg_map['source_sheet'] = 'first'

        tx = df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()

        if 'quantity' in tx.columns:
            tx = tx.rename(columns={'quantity': 'basket_items'})

        if 'line_total' in tx.columns:
            tx = tx.rename(columns={'line_total': 'total_sales'})

        if 'product_id' in tx.columns:
            tx = tx.rename(columns={'product_id': 'distinct_products'})
    else:
        tx = df.copy()

    if 'basket_items' not in tx.columns:
        if 'quantity' in tx.columns:
            tx['basket_items'] = _to_numeric(tx['quantity']).fillna(0).abs()
        else:
            tx['basket_items'] = 1.0

    if 'total_sales' not in tx.columns:
        if 'price' in tx.columns and 'quantity' in tx.columns:
            tx['total_sales'] = _to_numeric(tx['price']).fillna(0) * _to_numeric(tx['quantity']).fillna(0)
            notes.append('total_sales побудовано як quantity × price')
        elif 'price' in tx.columns:
            tx['total_sales'] = _to_numeric(tx['price']).fillna(0)
            notes.append('total_sales взято з price')
        else:
            tx['total_sales'] = 0.0

    tx['basket_items'] = pd.to_numeric(tx['basket_items'], errors='coerce').fillna(0).abs()
    tx['total_sales'] = pd.to_numeric(tx['total_sales'], errors='coerce').fillna(0)

    if 'transaction_date' in tx.columns:
        tx['transaction_date'] = pd.to_datetime(tx['transaction_date'], errors='coerce')

    if 'customer_id' in tx.columns and 'transaction_date' in tx.columns:
        sort_cols = ['customer_id', 'transaction_date']
        if 'transaction_id' in tx.columns:
            sort_cols.append('transaction_id')

        tx = tx.sort_values(sort_cols).reset_index(drop=True)

        if 'transaction_id' not in tx.columns:
            tx['transaction_id'] = (
                tx.groupby('customer_id').cumcount() + 1
            ).astype(str)

        tx['online_purchases'] = tx.groupby('customer_id').cumcount() + 1
    else:
        if 'transaction_id' not in tx.columns:
            tx['transaction_id'] = np.arange(1, len(tx) + 1).astype(str)

        if 'online_purchases' not in tx.columns:
            tx['online_purchases'] = 1

    if 'in_store_purchases' not in tx.columns:
        tx['in_store_purchases'] = 0

    if 'avg_purchase_value' not in tx.columns:
        tx['avg_purchase_value'] = tx['total_sales']

    if 'avg_discount_used' not in tx.columns:
        tx['avg_discount_used'] = 0.0

    if 'promo_used' not in tx.columns:
        tx['promo_used'] = 0.0

    if 'promo_response_rate' not in tx.columns:
        tx['promo_response_rate'] = 0.0

    if 'total_transactions' not in tx.columns:
        tx['total_transactions'] = 1

    if 'churn' in tx.columns:
        tx['churn'] = _normalize_existing_churn(tx['churn'])
    else:
        notes.append('churn не знайдено у файлі; target буде побудовано пізніше з future window')

    tx = tx.replace([np.inf, -np.inf], np.nan)

    return tx, detected_mapping, notes


@st.cache_data(show_spinner=False)
def load_and_prepare_transactions(uploaded_file) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    raw_df = load_table_file(uploaded_file)
    prepared_df, mapping, notes = prepare_transactions_dataframe(raw_df)
    prepared_df = standardize_dates(prepared_df)
    return prepared_df, mapping, notes


def merge_customer_product_stats(transactions: pd.DataFrame, products: Optional[pd.DataFrame]) -> pd.DataFrame:
    if products is None or products.empty or 'customer_id' not in products.columns:
        return transactions.copy()

    products = products.copy()
    products['text_for_embedding'] = (
        products.get('product_name', pd.Series('', index=products.index)).fillna('').astype(str)
        + ' '
        + products.get('product_description', pd.Series('', index=products.index)).fillna('').astype(str)
    ).str.strip()

    agg = products.groupby('customer_id').agg(
        distinct_products=('product_name', 'nunique') if 'product_name' in products.columns else ('customer_id', 'size'),
        distinct_categories=('category', 'nunique') if 'category' in products.columns else ('customer_id', 'size'),
        product_rows=('customer_id', 'size'),
    ).reset_index()

    merged = transactions.merge(agg, on='customer_id', how='left')
    return merged


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator.div(denominator).replace([np.inf, -np.inf], np.nan)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if 'customer_id' not in work.columns:
        raise ValueError('customer_id is required for feature engineering.')

    if 'transaction_date' in work.columns:
        work['transaction_date'] = pd.to_datetime(work['transaction_date'], errors='coerce')
        work = work.sort_values(['customer_id', 'transaction_date', 'transaction_id'] if 'transaction_id' in work.columns else ['customer_id', 'transaction_date']).copy()
    else:
        work = work.sort_values(['customer_id']).copy()

    if 'total_sales' not in work.columns:
        work['total_sales'] = 0.0
    work['total_sales'] = pd.to_numeric(work['total_sales'], errors='coerce').fillna(0.0)

    if 'basket_items' not in work.columns:
        work['basket_items'] = 1.0
    work['basket_items'] = pd.to_numeric(work['basket_items'], errors='coerce').fillna(0.0)

    if 'avg_discount_used' not in work.columns:
        work['avg_discount_used'] = 0.0
    work['avg_discount_used'] = pd.to_numeric(work['avg_discount_used'], errors='coerce').fillna(0.0)

    if 'promo_used' not in work.columns:
        work['promo_used'] = 0.0
    work['promo_used'] = pd.to_numeric(work['promo_used'], errors='coerce').fillna(0.0)

    if 'promo_response_rate' not in work.columns:
        work['promo_response_rate'] = 0.0
    work['promo_response_rate'] = pd.to_numeric(work['promo_response_rate'], errors='coerce').fillna(0.0).clip(0.0, 1.0)

    if 'online_purchases' not in work.columns:
        work['online_purchases'] = work.groupby('customer_id').cumcount() + 1
    work['online_purchases'] = pd.to_numeric(work['online_purchases'], errors='coerce').fillna(0.0)

    if 'in_store_purchases' not in work.columns:
        work['in_store_purchases'] = 0.0
    work['in_store_purchases'] = pd.to_numeric(work['in_store_purchases'], errors='coerce').fillna(0.0)

    work['customer_order_count'] = work.groupby('customer_id').cumcount() + 1
    work['total_transactions'] = work.get('total_transactions', work['customer_order_count'])
    work['total_transactions'] = pd.to_numeric(work['total_transactions'], errors='coerce').fillna(work['customer_order_count'])

    if 'transaction_date' in work.columns:
        prev_purchase = work.groupby('customer_id')['transaction_date'].shift(1)
        work['days_since_prev_purchase'] = (
            work['transaction_date'] - prev_purchase
        ).dt.days.fillna(0).clip(lower=0)
        work['customer_mean_gap'] = (
            work.groupby('customer_id')['days_since_prev_purchase']
            .transform(lambda s: s.replace(0, np.nan).expanding().mean())
            .fillna(0)
        )
        last_purchase = work.groupby('customer_id')['transaction_date'].transform('max')
        snapshot_date = work['transaction_date'].max() + pd.Timedelta(days=1)
        work['days_since_last_purchase'] = (snapshot_date - last_purchase).dt.days.clip(lower=0)
        work['customer_tenure_days'] = (
            work['transaction_date'] - work.groupby('customer_id')['transaction_date'].transform('min')
        ).dt.days.fillna(0).clip(lower=0)
    else:
        work['days_since_prev_purchase'] = 0.0
        work['customer_mean_gap'] = 0.0
        work['days_since_last_purchase'] = 0.0
        work['customer_tenure_days'] = 0.0

    work['avg_ticket_size'] = (
        work.groupby('customer_id')['total_sales']
        .transform(lambda s: s.expanding().mean())
        .fillna(0.0)
    )
    work['avg_purchase_value'] = pd.to_numeric(
        work.get('avg_purchase_value', work['avg_ticket_size']),
        errors='coerce'
    ).fillna(work['avg_ticket_size'])

    work['avg_item_value'] = _safe_ratio(work['total_sales'], work['basket_items']).fillna(0.0)
    work['discount_impact'] = (work['avg_discount_used'] * work['promo_response_rate']).fillna(0.0)

    if 'distinct_products' in work.columns:
        work['distinct_products'] = pd.to_numeric(work['distinct_products'], errors='coerce').fillna(0.0)
    elif 'product_id' in work.columns:
        work['distinct_products'] = (
            work.groupby('customer_id')['product_id']
            .cumcount() + 1
        ).astype(float)
    else:
        work['distinct_products'] = 0.0

    category_spend_cols = [col for col in work.columns if col.startswith('category_spend_')]
    if category_spend_cols:
        for col in category_spend_cols:
            work[col] = pd.to_numeric(work[col], errors='coerce').fillna(0.0)
        work['distinct_categories'] = (work[category_spend_cols] > 0).sum(axis=1).astype(float)
    elif 'category' in work.columns:
        work['distinct_categories'] = (
            work.groupby('customer_id')['category']
            .cumcount() + 1
        ).astype(float)
    else:
        work['distinct_categories'] = 0.0

    work['purchase_frequency'] = _safe_ratio(work['customer_order_count'], work['customer_tenure_days'] + 1).fillna(0.0)
    work['items_per_order'] = _safe_ratio(work['basket_items'], work['total_transactions']).fillna(0.0)

    numeric_columns = work.select_dtypes(include=[np.number, 'bool']).columns
    if len(numeric_columns) > 0:
        work[numeric_columns] = work[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if 'churn' in work.columns:
        work['churn'] = _normalize_existing_churn(work['churn'])

    return work


def build_forward_churn_dataset(
    df: pd.DataFrame,
    horizon_days: int = 90,
) -> pd.DataFrame:
    if 'customer_id' not in df.columns:
        raise ValueError('customer_id is required.')
    if 'transaction_date' not in df.columns:
        raise ValueError('transaction_date is required.')

    work = df.copy()
    work['transaction_date'] = pd.to_datetime(work['transaction_date'], errors='coerce')
    work = work.dropna(subset=['customer_id', 'transaction_date']).copy()

    sort_cols = ['customer_id', 'transaction_date']
    if 'transaction_id' in work.columns:
        sort_cols.append('transaction_id')

    work = work.sort_values(sort_cols).copy()

    work['next_purchase_date'] = work.groupby('customer_id')['transaction_date'].shift(-1)
    work['days_to_next_purchase'] = (
        work['next_purchase_date'] - work['transaction_date']
    ).dt.days

    data_end_date = work['transaction_date'].max()

    work['has_full_horizon'] = (
        work['transaction_date'] + pd.to_timedelta(horizon_days, unit='D')
    ) <= data_end_date

    work = work[work['has_full_horizon']].copy()

    work['churn'] = (
        work['next_purchase_date'].isna()
        | (work['days_to_next_purchase'] > horizon_days)
    ).astype(int)

    return work


def build_training_matrices(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    if 'customer_id' not in df.columns:
        raise ValueError('customer_id is required for model training.')
    if 'churn' not in df.columns:
        raise ValueError('churn is required for model training.')

    work = df.copy()

    excluded_columns = {
        'customer_id',
        'transaction_id',
        'transaction_date',
        'last_purchase_date',
        'invoice_date',
        'source_sheet',
        'churn',
        'churn_probability',
        'churn_probability_percent',
        'risk_class',
        'rfm_segment',
        'RFM_score',
        'recommended_action',
        'recommended_channel',
        'top_categories',
        'dominant_category',
        'days_since_last_purchase',
        'days_to_next_purchase',
        'has_full_horizon',
        'is_currently_active',
    }

    bool_columns = work.select_dtypes(include=['bool']).columns.tolist()
    if bool_columns:
        work[bool_columns] = work[bool_columns].astype(int)

    numeric_columns = [
        col for col in work.select_dtypes(include=[np.number]).columns
        if col not in excluded_columns
    ]

    if not numeric_columns:
        raise ValueError('No numeric features are available for model training after preprocessing.')

    X = work[numeric_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y = _normalize_existing_churn(work['churn'])
    groups = work['customer_id'].astype(str)

    return X, y, groups, numeric_columns
