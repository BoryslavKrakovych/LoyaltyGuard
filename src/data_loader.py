from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import MAPPING_TEMPLATES_DIR


COLUMN_ALIASES = {
    'customer_id': [
        'customer_id', 'customer id', 'customerid',
        'client_id', 'client id',
        'userid', 'user_id',
        'cust_id', 'custid',
        'customer', 'client', 'user', 'buyer', 'покупець', 'клієнт', 'клиент'
    ],
    'transaction_date': [
        'transaction_date', 'transaction date', 'transactiondate',
        'invoice_date', 'invoice date', 'invoicedate',
        'order_date', 'order date', 'orderdate',
        'purchase_date', 'purchase date', 'purchasedate',
        'date', 'datetime', 'timestamp', 'created_at', 'created at', 'дата', 'дата транзакції', 'дата покупки', 'дата продажу'
    ],
    'transaction_id': [
        'transaction_id', 'transaction id', 'transactionid',
        'invoice', 'invoice_no', 'invoice no', 'invoiceno', 'receipt', 'receipt_no', 'check_id', 'чек', 'номер чеку',
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
        'description', 'desc',
        'item_name', 'item name',
        'name', 'title', 'товар', 'назва товару', 'product description'
    ],
    'quantity': [
        'quantity', 'qty', 'items', 'units', 'count'
    ],
    'price': [
        'price',
        'unit_price', 'unit price', 'unitprice',
        'item_price', 'item price',
        'amount', 'sum', 'total', 'value', 'revenue', 'sales', 'сума', 'вартість'
    ],
    'country': [
        'country', 'region'
    ],
    'churn': [
        'churn', 'is_churn', 'target', 'label'
    ],
}


CANONICAL_TRANSACTION_COLUMNS = [
    'customer_id',
    'transaction_date',
    'transaction_id',
    'product_id',
    'product_name',
    'quantity',
    'price',
    'country',
    'churn',
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



def _safe_template_name(template_name: str) -> str:
    template_name = str(template_name).strip()
    if not template_name:
        raise ValueError('Вкажіть назву шаблону маппінгу.')
    return re.sub(r'[^a-zA-Z0-9А-Яа-я_\-]+', '_', template_name)


def list_mapping_templates() -> list[str]:
    templates = []
    for path in MAPPING_TEMPLATES_DIR.glob('*.json'):
        templates.append(path.stem)
    return sorted(templates)


def save_mapping_template(template_name: str, mapping: dict[str, str]) -> Path:
    safe_name = _safe_template_name(template_name)
    path = MAPPING_TEMPLATES_DIR / f'{safe_name}.json'

    cleaned_mapping = {
        key: value
        for key, value in mapping.items()
        if key and value
    }

    path.write_text(
        json.dumps(cleaned_mapping, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    return path


def load_mapping_template(template_name: str) -> dict[str, str]:
    template_name = str(template_name).strip()
    if not template_name:
        return {}

    safe_name = _safe_template_name(template_name)
    path = MAPPING_TEMPLATES_DIR / f'{safe_name}.json'

    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding='utf-8'))


def load_table_from_database(connection_string: str, sql_query: str) -> pd.DataFrame:
    if not str(connection_string).strip():
        raise ValueError('Не вказано connection string для бази даних.')

    if not str(sql_query).strip():
        raise ValueError('Не вказано SQL-запит.')

    try:
        from sqlalchemy import create_engine
    except Exception as error:
        raise ImportError(
            'Для підключення до БД встановіть SQLAlchemy: pip install SQLAlchemy'
        ) from error

    engine = create_engine(connection_string)

    with engine.connect() as connection:
        return pd.read_sql(sql_query, connection)


def list_database_tables(connection_string: str) -> list[str]:
    """Return list of table names available in the database."""
    if not str(connection_string).strip():
        return []
    try:
        from sqlalchemy import create_engine, inspect as sa_inspect
    except Exception:
        return []

    try:
        engine = create_engine(connection_string)
        inspector = sa_inspect(engine)
        tables = inspector.get_table_names()
        views = inspector.get_view_names()
        return sorted(tables + views)
    except Exception:
        return []


def preview_column_info(df: pd.DataFrame, n_examples: int = 3) -> list[dict]:
    """
    For each column return: name, dtype, null%, and up to n_examples sample values.
    Used to render the mapping UI with context.
    """
    rows = []
    total = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        null_pct = round(series.isna().sum() / total * 100, 1)
        examples = (
            series.dropna()
            .astype(str)
            .str.strip()
            .replace('', pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        rows.append({
            'column': col,
            'dtype': str(series.dtype),
            'null_pct': null_pct,
            'examples': examples[:n_examples],
        })
    return rows


REQUIRED_CANONICAL_FIELDS = ('customer_id', 'transaction_date')


def template_matches_schema(template: dict[str, str], columns: list[str]) -> bool:
    """
    Return True if the saved template is a SAFE auto-apply for the current dataset:
      1. Template is non-empty.
      2. Every mapped source column actually exists in the current dataset
         (else mapping points to a non-existent column).
      3. All REQUIRED_CANONICAL_FIELDS are mapped (customer_id, transaction_date).
         Without these, downstream feature engineering crashes.
    Якщо хоч одна вимога не виконана — UI маппінгу показується, не автозастосовується.
    """
    if not template:
        return False

    mapped_sources = [v for v in template.values() if v]
    if not mapped_sources:
        return False
    if not all(src in columns for src in mapped_sources):
        return False

    # Усі обов'язкові канонічні поля мають бути присутні І вказувати на існуючу колонку.
    for required in REQUIRED_CANONICAL_FIELDS:
        src = template.get(required)
        if not src or src not in columns:
            return False

    return True


def load_csv_path(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_table_path(path: str | Path) -> pd.DataFrame:
    file_path = Path(path).expanduser()

    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f'Файл не знайдено: {file_path}')

    suffix = file_path.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(file_path)

    if suffix in {'.xlsx', '.xls'}:
        excel_file = pd.ExcelFile(file_path)
        frames = []

        for sheet_name in excel_file.sheet_names:
            part = pd.read_excel(excel_file, sheet_name=sheet_name)
            part['source_sheet'] = sheet_name
            frames.append(part)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    raise ValueError('Підтримуються лише файли CSV, XLSX або XLS.')


def load_table_bytes(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(io.BytesIO(file_bytes))

    if suffix in {'.xlsx', '.xls'}:
        excel_buffer = io.BytesIO(file_bytes)
        excel_file = pd.ExcelFile(excel_buffer)
        frames = []

        for sheet_name in excel_file.sheet_names:
            part = pd.read_excel(excel_file, sheet_name=sheet_name)
            part['source_sheet'] = sheet_name
            frames.append(part)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    raise ValueError('Підтримуються лише файли CSV, XLSX або XLS.')


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


def prepare_transactions_dataframe(
    raw_df: pd.DataFrame,
    manual_mapping: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    df = raw_df.copy()
    detected_mapping = infer_column_mapping(df.columns.tolist())
    notes: list[str] = []

    if manual_mapping:
        for canonical_name, source_name in manual_mapping.items():
            if source_name:
                detected_mapping[canonical_name] = source_name

    rename_map = {}
    for canonical_name, source_name in detected_mapping.items():
        if source_name in df.columns and source_name != canonical_name:
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


def load_and_prepare_transactions_from_bytes(
    file_bytes: bytes,
    file_name: str,
    manual_mapping: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    raw_df = load_table_bytes(file_bytes, file_name)
    prepared_df, mapping, notes = prepare_transactions_dataframe(raw_df, manual_mapping=manual_mapping)
    prepared_df = standardize_dates(prepared_df)
    return prepared_df, mapping, notes


def load_and_prepare_transactions_from_dataframe(
    raw_df: pd.DataFrame,
    manual_mapping: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    prepared_df, mapping, notes = prepare_transactions_dataframe(raw_df, manual_mapping=manual_mapping)
    prepared_df = standardize_dates(prepared_df)
    return prepared_df, mapping, notes


def load_and_prepare_transactions(
    uploaded_file,
    manual_mapping: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    if isinstance(uploaded_file, (str, Path)):
        raw_df = load_table_path(uploaded_file)
    else:
        raise ValueError('Для цієї функції потрібно передати шлях до файлу. Для upload використовуйте load_and_prepare_transactions_from_bytes().')

    prepared_df, mapping, notes = prepare_transactions_dataframe(raw_df, manual_mapping=manual_mapping)
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
