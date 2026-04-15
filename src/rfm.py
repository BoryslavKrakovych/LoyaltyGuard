from __future__ import annotations

import pandas as pd


def build_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    if 'transaction_date' not in df.columns or 'customer_id' not in df.columns:
        raise ValueError('transaction_date and customer_id are required for RFM analysis.')

    work_df = df.copy()
    work_df['transaction_date'] = pd.to_datetime(work_df['transaction_date'], errors='coerce')
    snapshot_date = work_df['transaction_date'].max() + pd.Timedelta(days=1)

    sales_col = 'total_sales' if 'total_sales' in work_df.columns else None

    rfm = work_df.groupby('customer_id').agg(
        recency=('transaction_date', lambda x: (snapshot_date - x.max()).days),
        frequency=('transaction_id', 'nunique') if 'transaction_id' in work_df.columns else ('transaction_date', 'count'),
        monetary=(sales_col, 'sum') if sales_col else ('transaction_date', 'count'),
    ).reset_index()

    rfm['R_score'] = pd.qcut(rfm['recency'].rank(method='first', ascending=False), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
    rfm['rfm_segment'] = rfm.apply(_segment_from_scores, axis=1)
    return rfm


def _segment_from_scores(row: pd.Series) -> str:
    r, f, m = row['R_score'], row['F_score'], row['M_score']

    if r >= 4 and f >= 4 and m >= 4:
        return 'VIP / Champions'
    if r <= 2 and (f >= 4 or m >= 4):
        return 'At Risk'
    if r >= 4 and f <= 2:
        return 'New / Promising'
    if r == 3 and f in [2, 3] and m in [2, 3, 4]:
        return 'Need Attention'
    return 'Lost'
