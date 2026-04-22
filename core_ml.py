from pathlib import Path
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


CURRENT_DIR = Path(__file__).parent
MODELS_DIR = CURRENT_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True, parents=True)
RANDOM_STATE = 42


@dataclass
class ChurnArtifacts:
    model: object
    X_train_columns: list[str]
    train_index: np.ndarray
    test_index: np.ndarray
    roc_auc: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    algorithm_name: str


class ChurnModelService:
    def __init__(self, use_xgboost: bool = True, random_state: int = RANDOM_STATE):
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        self.random_state = random_state

    def _build_model(self):
        if self.use_xgboost:
            return XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                eval_metric='logloss',
            ), 'XGBoost'

        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.random_state,
        ), 'GradientBoosting'

    def train(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, test_size: float = 0.15) -> ChurnArtifacts:
        if y.nunique() < 2:
            raise ValueError('Цільова змінна churn повинна містити щонайменше два класи (0 та 1).')

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model, algorithm_name = self._build_model()
        model.fit(X_train, y_train)

        if y_test.nunique() >= 2:
            proba_test = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, proba_test)
            roc_auc = auc(fpr, tpr)
        else:
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            thresholds = np.array([np.inf, 0.0])
            roc_auc = float('nan')

        return ChurnArtifacts(
            model=model,
            X_train_columns=list(X_train.columns),
            train_index=train_idx,
            test_index=test_idx,
            roc_auc=float(roc_auc),
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            algorithm_name=algorithm_name,
        )


def build_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    work_df['transaction_date'] = pd.to_datetime(
        work_df.get('transaction_date', pd.Timestamp.today()),
        errors='coerce',
    )
    snapshot_date = work_df['transaction_date'].max() + pd.Timedelta(days=1)

    sales_col = 'total_sales' if 'total_sales' in work_df.columns else None

    rfm = work_df.groupby('customer_id').agg(
        recency=('transaction_date', lambda x: (snapshot_date - x.max()).days),
        frequency=('transaction_id', 'nunique') if 'transaction_id' in work_df.columns else ('transaction_date', 'count'),
        monetary=(sales_col, 'sum') if sales_col else ('transaction_date', 'count'),
    ).reset_index()

    rfm['R_score'] = pd.qcut(
        rfm['recency'].rank(method='first', ascending=False),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    rfm['F_score'] = pd.qcut(
        rfm['frequency'].rank(method='first'),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    rfm['M_score'] = pd.qcut(
        rfm['monetary'].rank(method='first'),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    rfm['RFM_score'] = (
        rfm['R_score'].astype(str) +
        rfm['F_score'].astype(str) +
        rfm['M_score'].astype(str)
    )

    def _segment_from_scores(row: pd.Series) -> str:
        r = row['R_score']
        f = row['F_score']
        m = row['M_score']

        if r >= 4 and f >= 4 and m >= 4:
            return 'VIP / Champions'
        if r <= 2 and (f >= 4 or m >= 4):
            return 'At Risk'
        if r >= 4 and f <= 2:
            return 'New / Promising'
        if r == 3 and f in [2, 3] and m in [2, 3, 4]:
            return 'Need Attention'
        return 'Lost'

    rfm['rfm_segment'] = rfm.apply(_segment_from_scores, axis=1)
    return rfm


def _safe_value(row: pd.Series, column: str) -> float:
    return float(pd.to_numeric(pd.Series([row.get(column, 0.0)]), errors='coerce').fillna(0.0).iloc[0])


def _make_unique_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        used_names.add(name)
        return name

    counter = 2
    new_name = f'{name} {counter}'
    while new_name in used_names:
        counter += 1
        new_name = f'{name} {counter}'

    used_names.add(new_name)
    return new_name


def _choose_cluster_name(row: pd.Series, rank: int, medians: dict[str, float]) -> str:
    sales = _safe_value(row, 'total_sales')
    avg_ticket = _safe_value(row, 'avg_ticket_size')
    basket = _safe_value(row, 'basket_items')
    online = _safe_value(row, 'online_purchases')
    in_store = _safe_value(row, 'in_store_purchases')
    promo = _safe_value(row, 'promo_response_rate')
    distinct_products = _safe_value(row, 'distinct_products')
    distinct_categories = _safe_value(row, 'distinct_categories')

    diversity = distinct_products + distinct_categories

    if rank == 0:
        return 'VIP клієнти'

    if sales >= medians.get('total_sales', 0) and avg_ticket >= medians.get('avg_ticket_size', 0):
        return 'Клієнти з високим середнім чеком'

    if online >= medians.get('online_purchases', 0) and basket >= medians.get('basket_items', 0):
        return 'Активні покупці'

    if promo >= medians.get('promo_response_rate', 0):
        return 'Промо-активні клієнти'

    if diversity >= medians.get('diversity', 0):
        return 'Різноманітні покупці'

    if in_store > online and in_store >= medians.get('in_store_purchases', 0):
        return 'Офлайн-орієнтовані клієнти'

    if basket >= medians.get('basket_items', 0):
        return 'Покупці з великим кошиком'

    if avg_ticket >= medians.get('avg_ticket_size', 0):
        return 'Покупці з високим чеком'

    if sales <= medians.get('total_sales', 0) and online <= medians.get('online_purchases', 0):
        return 'Малоактивні клієнти'

    return 'Помірні покупці'


def build_customer_clusters(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    work = df.copy()

    candidate_cols = [
        'total_sales',
        'avg_ticket_size',
        'basket_items',
        'online_purchases',
        'in_store_purchases',
        'promo_response_rate',
        'distinct_products',
        'distinct_categories',
    ]
    feature_cols = [col for col in candidate_cols if col in work.columns]

    if 'customer_id' not in work.columns or len(feature_cols) < 2:
        return pd.DataFrame(columns=['customer_id', 'customer_cluster', 'customer_cluster_name'])

    if 'transaction_date' in work.columns:
        latest = (
            work.sort_values(['customer_id', 'transaction_date'])
            .groupby('customer_id')
            .tail(1)
            .copy()
        )
    else:
        latest = work.drop_duplicates('customer_id').copy()

    model_data = latest[['customer_id'] + feature_cols].copy()
    model_data[feature_cols] = model_data[feature_cols].fillna(model_data[feature_cols].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_data[feature_cols])

    n_clusters = max(2, min(n_clusters, len(model_data)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    model_data['customer_cluster'] = kmeans.fit_predict(X_scaled)

    cluster_stats = model_data.groupby('customer_cluster')[feature_cols].mean().reset_index()

    score_cols = [col for col in ['total_sales', 'avg_ticket_size', 'basket_items', 'online_purchases'] if col in cluster_stats.columns]
    if score_cols:
        cluster_stats['business_score'] = cluster_stats[score_cols].sum(axis=1)
    else:
        cluster_stats['business_score'] = 0.0

    cluster_stats['diversity'] = (
        cluster_stats['distinct_products'] if 'distinct_products' in cluster_stats.columns else 0
    ) + (
        cluster_stats['distinct_categories'] if 'distinct_categories' in cluster_stats.columns else 0
    )

    cluster_stats = cluster_stats.sort_values('business_score', ascending=False).reset_index(drop=True)

    median_features = {}
    for col in [
        'total_sales',
        'avg_ticket_size',
        'basket_items',
        'online_purchases',
        'in_store_purchases',
        'promo_response_rate',
        'diversity',
    ]:
        if col in cluster_stats.columns:
            median_features[col] = float(cluster_stats[col].median())
        else:
            median_features[col] = 0.0

    used_names: set[str] = set()
    rank_to_name: dict[int, str] = {}

    for rank, row in cluster_stats.iterrows():
        raw_name = _choose_cluster_name(row, rank, median_features)
        final_name = _make_unique_name(raw_name, used_names)
        rank_to_name[int(row['customer_cluster'])] = final_name

    model_data['customer_cluster_name'] = model_data['customer_cluster'].map(rank_to_name)

    return model_data[['customer_id', 'customer_cluster', 'customer_cluster_name']]


if __name__ == '__main__':
    print(f'Робоча папка: {CURRENT_DIR}')
    print('Завантаження підготовлених матриць...')

    try:
        X = pd.read_csv(CURRENT_DIR / 'X_train.csv')
        y = pd.read_csv(CURRENT_DIR / 'y_train.csv').squeeze()
        featured_data = pd.read_csv(CURRENT_DIR / 'featured_data.csv')
    except FileNotFoundError as error:
        print(
            f'Помилка завантаження: {error}\n'
            'Переконайтеся, що файли X_train.csv, y_train.csv та featured_data.csv лежать поруч із цим скриптом.'
        )
        raise SystemExit(1)

    groups = featured_data['customer_id'].astype(str)

    print('Тренування моделі Churn...')
    churn_service = ChurnModelService(use_xgboost=True)
    artifacts = churn_service.train(X, y, groups)
    print(f'Модель Churn натреновано! Алгоритм: {artifacts.algorithm_name} | ROC-AUC: {artifacts.roc_auc:.4f}')

    print('Формування RFM-сегментації...')
    rfm = build_rfm_table(featured_data)

    print('Кластеризація клієнтів...')
    clusters = build_customer_clusters(featured_data, n_clusters=5)

    print(f'Збереження моделей у директорію {MODELS_DIR}...')
    joblib.dump(artifacts, MODELS_DIR / 'churn_artifacts.joblib')
    joblib.dump(rfm, MODELS_DIR / 'rfm_table.joblib')
    joblib.dump(clusters, MODELS_DIR / 'customer_clusters.joblib')

    print('Готово! Всі моделі успішно збережено.')