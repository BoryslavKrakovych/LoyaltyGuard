from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


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
    'quantity': ['quantity', 'qty', 'items', 'units', 'count'],
    'price': ['price', 'unit_price', 'unit price', 'unitprice', 'item_price', 'item price', 'amount'],
    'country': ['country', 'region'],
    'churn': ['churn', 'is_churn', 'target', 'label'],
    'category': ['category', 'product_category', 'segment', 'group'],
}


RFM_SEGMENTS = [
    'VIP / Champions',
    'New / Promising',
    'Need Attention',
    'At Risk',
    'Lost',
]


CLUSTER_FEATURES = [
    'Recency',
    'Frequency',
    'Monetary',
    'avg_ticket_size',
    'items_per_order',
]


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


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator.div(denominator).replace([np.inf, -np.inf], np.nan)


def _expanding_mean_before(series: pd.Series) -> pd.Series:
    shifted = series.shift(1)
    return shifted.expanding().mean()


def infer_column_mapping(columns: list[str]) -> dict[str, str]:
    normalized_to_original = {_normalize_col_name(col): col for col in columns}
    mapping: dict[str, str] = {}

    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_key = _normalize_col_name(alias)
            if alias_key in normalized_to_original:
                mapping[canonical_name] = normalized_to_original[alias_key]
                break

    return mapping


def load_table_file(file_path: str | Path) -> pd.DataFrame:
    path_obj = Path(file_path)
    file_name = path_obj.name.lower()

    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        excel_file = pd.ExcelFile(file_path)
        frames = []
        for sheet_name in excel_file.sheet_names:
            part = pd.read_excel(excel_file, sheet_name=sheet_name)
            part['source_sheet'] = sheet_name
            frames.append(part)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    return pd.read_csv(file_path)


def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['transaction_date', 'last_purchase_date', 'invoice_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def align_columns_for_inference(reference_columns: list[str], X_new: pd.DataFrame) -> pd.DataFrame:
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
        if 'product_name' in df.columns:
            agg_map['product_name'] = lambda s: ' | '.join(sorted({str(x).strip() for x in s.dropna() if str(x).strip()}))
        if 'category' in df.columns:
            agg_map['category'] = lambda s: ' | '.join(sorted({str(x).strip() for x in s.dropna() if str(x).strip()}))
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
        if 'product_name' in tx.columns:
            tx['product_text'] = tx['product_name']
    else:
        tx = df.copy()

    if 'product_name' in tx.columns and 'product_text' not in tx.columns:
        tx['product_text'] = tx['product_name']
    if 'category' in tx.columns and 'dominant_category' not in tx.columns:
        tx['dominant_category'] = tx['category']

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
            tx['transaction_id'] = (tx.groupby('customer_id').cumcount() + 1).astype(str)
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


def load_and_prepare_transactions(file_path: str | Path) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    raw_df = load_table_file(file_path)
    prepared_df, mapping, notes = prepare_transactions_dataframe(raw_df)
    prepared_df = standardize_dates(prepared_df)
    return prepared_df, mapping, notes


def merge_customer_product_stats(transactions: pd.DataFrame, products: Optional[pd.DataFrame]) -> pd.DataFrame:
    if products is None or products.empty:
        return transactions.copy()

    products = products.copy()
    if 'customer_id' not in products.columns:
        mapping = infer_column_mapping(products.columns.tolist())
        if 'customer_id' in mapping:
            products = products.rename(columns={mapping['customer_id']: 'customer_id'})

    if 'customer_id' not in products.columns:
        return transactions.copy()

    if 'product_name' not in products.columns:
        product_mapping = infer_column_mapping(products.columns.tolist())
        if 'product_name' in product_mapping:
            products = products.rename(columns={product_mapping['product_name']: 'product_name'})

    agg = products.groupby('customer_id').agg(
        distinct_products=('product_name', 'nunique') if 'product_name' in products.columns else ('customer_id', 'size'),
        distinct_categories=('category', 'nunique') if 'category' in products.columns else ('customer_id', 'size'),
        product_rows=('customer_id', 'size'),
    ).reset_index()

    merged = transactions.merge(agg, on='customer_id', how='left')
    return merged


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if 'customer_id' not in work.columns:
        raise ValueError('customer_id is required for feature engineering.')

    if 'transaction_date' in work.columns:
        work['transaction_date'] = pd.to_datetime(work['transaction_date'], errors='coerce')
        sort_cols = ['customer_id', 'transaction_date']
        if 'transaction_id' in work.columns:
            sort_cols.append('transaction_id')
        work = work.sort_values(sort_cols).copy()
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

    work['customer_order_count'] = work.groupby('customer_id').cumcount() + 1
    work['total_transactions'] = pd.to_numeric(work.get('total_transactions', work['customer_order_count']), errors='coerce').fillna(work['customer_order_count'])

    if 'transaction_date' in work.columns:
        prev_purchase = work.groupby('customer_id')['transaction_date'].shift(1)
        work['days_since_prev_purchase'] = (work['transaction_date'] - prev_purchase).dt.days.fillna(0).clip(lower=0)
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
    work['customer_avg_order_value_before'] = (
        work.groupby('customer_id')['total_sales']
        .transform(_expanding_mean_before)
        .fillna(work['avg_ticket_size'])
    )
    work['customer_avg_basket_before'] = (
        work.groupby('customer_id')['basket_items']
        .transform(_expanding_mean_before)
        .fillna(work['basket_items'])
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
        work['distinct_products'] = (work.groupby('customer_id')['product_id'].cumcount() + 1).astype(float)
    else:
        work['distinct_products'] = 0.0

    if 'category' in work.columns:
        work['distinct_categories'] = (work.groupby('customer_id')['category'].cumcount() + 1).astype(float)
    else:
        category_spend_cols = [col for col in work.columns if col.startswith('category_spend_')]
        if category_spend_cols:
            for col in category_spend_cols:
                work[col] = pd.to_numeric(work[col], errors='coerce').fillna(0.0)
            work['distinct_categories'] = (work[category_spend_cols] > 0).sum(axis=1).astype(float)
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


def _score_series(series: pd.Series, reverse: bool = False) -> pd.Series:
    filled = pd.to_numeric(series, errors='coerce').fillna(series.median() if hasattr(series, 'median') else 0)
    ranked = filled.rank(method='first', ascending=not reverse)
    bins = pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    scored = pd.Series(bins.astype(int), index=series.index)
    if reverse:
        return 6 - scored
    return scored


def build_rfm_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'customer_id' not in df.columns:
        return pd.DataFrame(columns=['customer_id', 'Recency', 'Frequency', 'Monetary', 'R_score', 'F_score', 'M_score', 'RFM_score', 'rfm_segment'])

    work = df.copy()
    work['transaction_date'] = pd.to_datetime(work['transaction_date'], errors='coerce')
    work = work.dropna(subset=['customer_id', 'transaction_date']).copy()
    if work.empty:
        return pd.DataFrame(columns=['customer_id', 'Recency', 'Frequency', 'Monetary', 'R_score', 'F_score', 'M_score', 'RFM_score', 'rfm_segment'])

    snapshot_date = work['transaction_date'].max() + pd.Timedelta(days=1)

    rfm = work.groupby('customer_id').agg(
        last_purchase_date=('transaction_date', 'max'),
        Frequency=('transaction_id', 'nunique') if 'transaction_id' in work.columns else ('customer_id', 'size'),
        Monetary=('total_sales', 'sum'),
    ).reset_index()
    rfm['Recency'] = (snapshot_date - rfm['last_purchase_date']).dt.days.clip(lower=0)

    rfm['R_score'] = _score_series(rfm['Recency'], reverse=True)
    rfm['F_score'] = _score_series(rfm['Frequency'])
    rfm['M_score'] = _score_series(rfm['Monetary'])
    rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
    rfm['rfm_segment'] = rfm.apply(assign_rfm_segment, axis=1)

    return rfm[['customer_id', 'Recency', 'Frequency', 'Monetary', 'R_score', 'F_score', 'M_score', 'RFM_score', 'rfm_segment']]


def assign_rfm_segment(row: pd.Series) -> str:
    r = int(row.get('R_score', 3))
    f = int(row.get('F_score', 3))
    m = int(row.get('M_score', 3))

    if r >= 4 and f >= 4 and m >= 3:
        return 'VIP / Champions'
    if r >= 4 and (f <= 3 or m <= 3):
        return 'New / Promising'
    if r <= 2 and f <= 2 and m <= 2:
        return 'Lost'
    if r <= 2 and (f >= 3 or m >= 3):
        return 'At Risk'
    return 'Need Attention'


def build_customer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if 'customer_id' not in df.columns:
        raise ValueError('customer_id is required.')

    latest = df.groupby('customer_id').tail(1).copy()
    rfm = build_rfm_summary(df)
    customer_df = latest.merge(rfm, on='customer_id', how='left')

    if 'product_text' not in customer_df.columns and 'product_name' in customer_df.columns:
        customer_df['product_text'] = customer_df['product_name']
    if 'dominant_category' not in customer_df.columns and 'category' in customer_df.columns:
        customer_df['dominant_category'] = customer_df['category']

    return customer_df


def build_forward_churn_dataset(df: pd.DataFrame, horizon_days: int = 90) -> pd.DataFrame:
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
    work['days_to_next_purchase'] = (work['next_purchase_date'] - work['transaction_date']).dt.days

    data_end_date = work['transaction_date'].max()
    work['has_full_horizon'] = (work['transaction_date'] + pd.to_timedelta(horizon_days, unit='D')) <= data_end_date
    work = work[work['has_full_horizon']].copy()

    work['churn'] = (
        work['next_purchase_date'].isna() | (work['days_to_next_purchase'] > horizon_days)
    ).astype(int)
    return work


def build_training_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    if 'customer_id' not in df.columns:
        raise ValueError('customer_id is required for model training.')
    if 'churn' not in df.columns:
        raise ValueError('churn is required for model training.')

    work = df.copy()

    excluded_columns = {
        'customer_id', 'transaction_id', 'transaction_date', 'last_purchase_date', 'invoice_date', 'source_sheet',
        'churn', 'next_purchase_date', 'days_to_next_purchase', 'has_full_horizon',
        'RFM_score', 'rfm_segment', 'risk_class', 'recommended_action', 'cluster_name'
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

    X = work[numeric_columns].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = _normalize_existing_churn(work['churn'])
    groups = work['customer_id'].astype(str)
    return X, y, groups, numeric_columns


def save_pipeline_datasets(
    df_featured: pd.DataFrame,
    customer_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    featured_path = out_dir / 'featured_data.csv'
    customer_path = out_dir / 'customer_level_data.csv'
    x_path = out_dir / 'X_train.csv'
    y_path = out_dir / 'y_train.csv'

    df_featured.to_csv(featured_path, index=False)
    customer_df.to_csv(customer_path, index=False)
    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)

    return {
        'featured_path': featured_path,
        'customer_path': customer_path,
        'x_path': x_path,
        'y_path': y_path,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Підготовка датасету LoyaltyGuard для churn / RFM / кластеризації')
    parser.add_argument('transactions', type=str, help='Шлях до файлу з транзакціями (CSV або Excel)')
    parser.add_argument('--products', type=str, default=None, help='Шлях до файлу з продуктами (опціонально)')
    parser.add_argument('--horizon', type=int, default=90, help='Горизонт прогнозування відтоку в днях')
    parser.add_argument('--output_dir', type=str, default='.', help='Директорія для збереження результатів')
    args = parser.parse_args()

    print(f'Завантаження та обробка транзакцій з: {args.transactions}...')
    df, mapping, notes = load_and_prepare_transactions(args.transactions)
    print(f'Мапінг колонок: {mapping}')
    for note in notes:
        print(f'Примітка: {note}')

    if args.products:
        print(f'Завантаження продуктів з: {args.products}...')
        products_df = load_table_file(args.products)
        df = merge_customer_product_stats(df, products_df)

    print('Генерація базових ознак...')
    df = add_base_features(df)

    if 'churn' not in df.columns or df['churn'].nunique() < 2:
        print(f'Формування цільової змінної churn (горизонт {args.horizon} днів)...')
        df = build_forward_churn_dataset(df, horizon_days=args.horizon)

    print('Побудова RFM...')
    customer_df = build_customer_dataset(df)
    print('Формування тренувальних матриць...')
    X, y, groups, numeric_cols = build_training_matrices(df)

    paths = save_pipeline_datasets(df, customer_df, X, y, args.output_dir)
    print('\nГотово! Збережені файли:')
    for name, path in paths.items():
        print(f'- {name}: {path}')
    print(f'- numeric_features: {len(numeric_cols)}')
