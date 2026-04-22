from pathlib import Path
import warnings

import joblib
import pandas as pd

from src.camp_push import generate_push_campaign
from core_ml import ChurnModelService, build_customer_clusters, build_rfm_table
from src.rfm_cool import perform_churn_and_cooling_analysis
from src.word_vec import build_user_profiles

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
FEATURED_FILE = BASE_DIR / 'featured_data.csv'
X_TRAIN_FILE = BASE_DIR / 'X_train.csv'
Y_TRAIN_FILE = BASE_DIR / 'y_train.csv'
MODELS_DIR = BASE_DIR / 'models'
CHURN_MODEL_FILE = MODELS_DIR / 'churn_artifacts.joblib'
RFM_TABLE_FILE = MODELS_DIR / 'rfm_table.joblib'
CUSTOMER_CLUSTERS_FILE = MODELS_DIR / 'customer_clusters.joblib'
OUTPUT_FILE = BASE_DIR / 'push_campaign_ready.csv'
QUEUE_FILE = BASE_DIR / 'rescue_queue.csv'
JSON_EXPORT_FILE = BASE_DIR / 'campaign_export.json'
FEEDBACK_FILE = BASE_DIR / 'feedback_log.csv'


def ensure_models() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    if CHURN_MODEL_FILE.exists() and RFM_TABLE_FILE.exists() and CUSTOMER_CLUSTERS_FILE.exists():
        return

    if not FEATURED_FILE.exists() or not X_TRAIN_FILE.exists() or not Y_TRAIN_FILE.exists():
        print("❌ Помилка: немає featured_data.csv, X_train.csv або y_train.csv.")
        return

    print('   -> Артефакти core_ml не знайдені. Автоматично тренуємо через core_ml.py...')
    X = pd.read_csv(X_TRAIN_FILE)
    y = pd.read_csv(Y_TRAIN_FILE).squeeze()
    featured = pd.read_csv(FEATURED_FILE)
    groups = featured['customer_id'].astype(str)

    churn_service = ChurnModelService(use_xgboost=True)
    churn_artifacts = churn_service.train(X, y, groups, test_size=0.15)
    rfm_df = build_rfm_table(featured)
    clusters_df = build_customer_clusters(featured)

    joblib.dump(churn_artifacts, CHURN_MODEL_FILE)
    joblib.dump(rfm_df, RFM_TABLE_FILE)
    joblib.dump(clusters_df, CUSTOMER_CLUSTERS_FILE)


def run_daily_loyalty_campaign() -> None:
    print('=== LOYALTYGUARD: ЛОКАЛЬНИЙ АНАЛІТИЧНИЙ МОДУЛЬ ===')

    if not FEATURED_FILE.exists():
        print("❌ Помилка: 'featured_data.csv' не знайдено. Спочатку підготуйте дані через app.py або xcelerator.py.")
        return

    df = pd.read_csv(FEATURED_FILE)
    if 'customer_id' not in df.columns:
        print("❌ Помилка: у 'featured_data.csv' немає колонки 'customer_id'.")
        return

    ensure_models()

    analysis_results = perform_churn_and_cooling_analysis(
        df,
        model_path=str(CHURN_MODEL_FILE),
        rfm_path=str(RFM_TABLE_FILE),
        clusters_path=str(CUSTOMER_CLUSTERS_FILE),
    )
    user_profiles = build_user_profiles(df)
    final_campaign = generate_push_campaign(
        analysis_results,
        user_profiles,
        feedback_path=str(FEEDBACK_FILE),
    )

    analysis_results.to_csv(QUEUE_FILE, index=False)
    final_campaign.to_csv(OUTPUT_FILE, index=False)
    final_campaign.to_json(JSON_EXPORT_FILE, orient='records', force_ascii=False, indent=2)

    print(f'\n✅ ГОТОВО! У rescue queue: {len(analysis_results)} клієнтів.')
    print(f"CSV кампанії: {OUTPUT_FILE.name}")
    print(f"JSON експорт: {JSON_EXPORT_FILE.name}")

    if len(final_campaign) > 0:
        print("\nПрев'ю перших 5 повідомлень:")
        print(final_campaign[['customer_id', 'recommended_action', 'push_text']].head(5).to_string(index=False))
    else:
        print('\nНемає клієнтів, які потрапили у цільову аудиторію.')


if __name__ == '__main__':
    run_daily_loyalty_campaign()
