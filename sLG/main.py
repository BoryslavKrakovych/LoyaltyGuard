import pandas as pd
import warnings
from step1_analysis import perform_churn_and_cooling_analysis
from step2_categories import build_user_profiles
from step3_campaigns import generate_push_campaign

warnings.filterwarnings('ignore')

def run_daily_loyalty_campaign():
    print("=== LOYALTYGUARD: ГЕНЕРАЦІЯ ЩОДЕННИХ КАМПАНІЙ ===")
    
    # 1. Завантажуємо вчорашні/свіжі дані (результат роботи xcelerator)
    try:
        df = pd.read_csv("featured_data.csv")
    except FileNotFoundError:
        print("❌ Помилка: 'featured_data.csv' не знайдено. Запустіть xcelerator.py для підготовки даних.")
        return

    # 2. Проганяємо через наші 3 кроки (які використовують готові моделі)
    analysis_results = perform_churn_and_cooling_analysis(df)
    user_profiles = build_user_profiles(df)
    final_campaign = generate_push_campaign(analysis_results, user_profiles)
    
    # 3. Зберігаємо результат для розсилки
    output_file = "push_campaign_ready.csv"
    final_campaign.to_csv(output_file, index=False)
    
    print(f"\n✅ ГОТОВО! Знайдено {len(final_campaign)} клієнтів для розсилки.")
    print(f"Файл збережено як: '{output_file}'")
    
    if len(final_campaign) > 0:
        print("\nПрев'ю перших 3 повідомлень:")
        print(final_campaign[['customer_id', 'push_text']].head(3).to_string(index=False))

if __name__ == "__main__":
    run_daily_loyalty_campaign()