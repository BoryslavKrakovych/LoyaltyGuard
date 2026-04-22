from pathlib import Path
import warnings

import pandas as pd

from src.rfm_cool import perform_churn_and_cooling_analysis
from src.word_vec import build_user_profiles
from src.camp_push import generate_push_campaign
from src.modeler import train_churn_model, train_kmeans_model, train_word2vec_model

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
FEATURED_FILE = BASE_DIR / 'featured_data.csv'
CUSTOMER_FILE = BASE_DIR / 'customer_level_data.csv'
X_TRAIN_FILE = BASE_DIR / 'X_train.csv'
Y_TRAIN_FILE = BASE_DIR / 'y_train.csv'
MODELS_DIR = BASE_DIR / 'models'
CHURN_MODEL_FILE = MODELS_DIR / 'churn_model.joblib'
WORD2VEC_MODEL_FILE = MODELS_DIR / 'word2vec.model'
KMEANS_MODEL_FILE = MODELS_DIR / 'kmeans_model.joblib'
OUTPUT_FILE = BASE_DIR / 'push_campaign_ready.csv'
QUEUE_FILE = BASE_DIR / 'rescue_queue.csv'
JSON_EXPORT_FILE = BASE_DIR / 'campaign_export.json'
FEEDBACK_FILE = BASE_DIR / 'feedback_log.csv'


def ensure_models() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    if not CHURN_MODEL_FILE.exists() and X_TRAIN_FILE.exists() and Y_TRAIN_FILE.exists():
        print('   -> XGBoost модель не знайдена. Автоматично тренуємо...')
        train_churn_model(X_TRAIN_FILE, Y_TRAIN_FILE, CHURN_MODEL_FILE)

    if not WORD2VEC_MODEL_FILE.exists() and FEATURED_FILE.exists():
        print('   -> Word2Vec модель не знайдена. Автоматично тренуємо...')
        train_word2vec_model(FEATURED_FILE, WORD2VEC_MODEL_FILE)

    if not KMEANS_MODEL_FILE.exists() and CUSTOMER_FILE.exists():
        print('   -> K-means модель не знайдена. Автоматично тренуємо...')
        train_kmeans_model(CUSTOMER_FILE, KMEANS_MODEL_FILE)


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
        kmeans_model_path=str(KMEANS_MODEL_FILE),
    )
    user_profiles = build_user_profiles(df, model_path=str(WORD2VEC_MODEL_FILE))
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
