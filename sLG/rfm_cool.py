import pandas as pd
import joblib

def perform_churn_and_cooling_analysis(df_featured, model_path="models/churn_model.joblib"):
    print("   -> Аналіз відтоку та активності...")
    # Беремо останній зріз по кожному клієнту
    latest = df_featured.groupby('customer_id').tail(1).copy()
    
    # 1. ЗАСТОСУВАННЯ МОДЕЛІ ВІДТОКУ (Inference)
    try:
        artifacts = joblib.load(model_path)
        model = artifacts['model']
        features = artifacts['features']
        
        # Заповнюємо відсутні колонки нулями (щоб порядок фіч збігався з тренуванням)
        for col in features:
            if col not in latest.columns:
                latest[col] = 0
                
        X_new = latest[features].fillna(0)
        latest['churn_probability'] = model.predict_proba(X_new)[:, 1]
    except FileNotFoundError:
        print("   [!] Модель відтоку не знайдена. Розрахунок ймовірності пропущено.")
        latest['churn_probability'] = 0.0

    # 2. ДЕТЕКЦІЯ ОХОЛОДЖЕННЯ (Cooling)
    # Якщо клієнт не купує довше, ніж його середня пауза * 1.5
    latest['cooling_flag'] = latest['days_since_prev_purchase'] > (latest['customer_mean_gap'] * 1.5)
    
    # 3. ВІДБІР ЦІЛЬОВОЇ АУДИТОРІЇ
    # Беремо тих, у кого високий ризик відтоку (>60%) АБО спрацював тригер охолодження
    latest['is_target'] = (latest['churn_probability'] > 0.60) | latest['cooling_flag']
    
    return latest[['customer_id', 'is_target', 'churn_probability', 'cooling_flag']]