import pandas as pd

def generate_push_campaign(analysis_df, profiles_df):
    print("   -> Генерація текстів для пуш-повідомлень...")
    final = analysis_df.merge(profiles_df, on='customer_id')
    
    # Відфільтровуємо лише тих, кому потрібна увага
    targets = final[final['is_target'] == True].copy()
    
    def get_message(row):
        category = str(row['favorite_category']).title()
        churn_prob = row['churn_probability']
        
        # Якщо високий ризик відтоку (ML модель)
        if churn_prob > 0.60:
            return f"Ми сумуємо! Повертайтесь, для вас діє спеціальна знижка на '{category}'."
        # Якщо просто став рідше купувати (Cooling Trigger)
        elif row['cooling_flag']:
            return f"Зробіть свої покупки ще вигіднішими! Перевірте новинки у категорії '{category}'."
        
        return f"Персональна пропозиція: '{category}' чекає на вас."

    targets['push_text'] = targets.apply(get_message, axis=1)
    
    # Округлюємо ймовірність для красивого звіту
    targets['churn_probability'] = (targets['churn_probability'] * 100).round(1).astype(str) + '%'
    
    return targets[['customer_id', 'favorite_category', 'churn_probability', 'cooling_flag', 'push_text']]