import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from dataclasses import dataclass # <-- Додали імпорт

# --- ДОДАЄМО ОПИС КЛАСУ ДЛЯ РОЗПАКУВАННЯ МОДЕЛІ ---
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

# Шляхи до файлів
CURRENT_DIR = Path(__file__).parent
MODELS_DIR = CURRENT_DIR / 'models'

class LoyaltyGuardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LoyaltyGuard: Генератор кампаній")
        self.root.geometry("1050x550")
        
        # Робимо дизайн трохи сучаснішим
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Основний контейнер
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        tk.Label(main_frame, text="🎯 LoyaltyGuard", font=("Arial", 18, "bold")).pack(anchor="w")
        tk.Label(
            main_frame, 
            text="Система аналізує історію покупок та генерує персональні пропозиції для клієнтів, що перестають купувати.", 
            font=("Arial", 11), fg="#555555"
        ).pack(anchor="w", pady=(0, 15))

        # Налаштування таблиці
        columns = ("id", "segment", "category", "risk", "message", "channel")
        self.tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=15)

        self.tree.heading("id", text="ID Клієнта")
        self.tree.heading("segment", text="Сегмент (RFM)")
        self.tree.heading("category", text="Улюблена Категорія")
        self.tree.heading("risk", text="Ризик (%)")
        self.tree.heading("message", text="Рекомендований Меседж")
        self.tree.heading("channel", text="Канал")

        self.tree.column("id", width=120)
        self.tree.column("segment", width=120)
        self.tree.column("category", width=150)
        self.tree.column("risk", width=80, anchor="center")
        self.tree.column("message", width=380)
        self.tree.column("channel", width=100, anchor="center")

        # Додаємо скролбар
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Панель кнопок знизу
        btn_frame = tk.Frame(main_frame, pady=10)
        btn_frame.pack(fill=tk.X)

        self.btn_export = tk.Button(
            btn_frame, text="📥 Зберегти базу для Push-розсилки (CSV)", 
            font=("Arial", 11, "bold"), bg="#4CAF50", fg="white", 
            padx=15, pady=5, command=self.export_csv, state=tk.DISABLED
        )
        self.btn_export.pack(side=tk.RIGHT)

        # Статус-бар знизу вікна
        self.status_var = tk.StringVar()
        self.status_var.set("Завантаження моделей та аналіз даних. Зачекайте...")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=10)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Змінна для зберігання фінального датафрейму
        self.final_df = None

        # Запускаємо обробку даних через 100мс після відкриття вікна
        self.root.after(100, self.process_data)

    def process_data(self):
        try:
            # 1. ЗАВАНТАЖЕННЯ ДАНИХ
            featured_data = pd.read_csv(CURRENT_DIR / 'featured_data.csv')
            churn_model = joblib.load(MODELS_DIR / 'churn_artifacts.joblib')
            rfm = joblib.load(MODELS_DIR / 'rfm_table.joblib')
            
            # 2. АНАЛІЗ ТА ПРЕДІКТ
            latest_state = featured_data.groupby('customer_id').tail(1).copy()
            latest_state = latest_state.merge(rfm[['customer_id', 'rfm_segment']], on='customer_id', how='left')
            
            cat_cols = [c for c in latest_state.columns if c.startswith('category_spend_')]
            if cat_cols:
                latest_state['favorite_category'] = latest_state[cat_cols].idxmax(axis=1).str.replace('category_spend_', '').str.capitalize()
            else:
                latest_state['favorite_category'] = "Загальний асортимент"

            X_inference = latest_state[churn_model.X_train_columns].fillna(0)
            probs = churn_model.model.predict_proba(X_inference)[:, 1]
            latest_state['churn_risk_pct'] = np.round(probs * 100, 1)

            latest_state['is_cooling'] = (latest_state['days_since_prev_purchase'] > latest_state['customer_mean_gap'] * 1.5) | (latest_state['churn_risk_pct'] > 40)
            rescue_queue = latest_state[latest_state['is_cooling']].copy()

            # 3. ГЕНЕРАЦІЯ КАМПАНІЇ
            def generate_campaign(row):
                cat = row['favorite_category']
                risk = row['churn_risk_pct']
                segment = str(row['rfm_segment'])
                
                if risk > 70 and "At Risk" in segment:
                    return f"🔥 Повертайтесь! Знижка -20% на {cat}", "Push + SMS"
                elif risk > 50:
                    return f"✨ Ми зібрали для вас новинки: {cat}", "Push"
                else:
                    return f"👋 Здається, вам сподобається: {cat}", "Push"

            campaigns = rescue_queue.apply(generate_campaign, axis=1)
            rescue_queue['message_template'] = [c[0] for c in campaigns]
            rescue_queue['channel'] = [c[1] for c in campaigns]
            
            rescue_queue = rescue_queue.sort_values('churn_risk_pct', ascending=False)

            self.final_df = rescue_queue[[
                'customer_id', 'rfm_segment', 'favorite_category', 
                'churn_risk_pct', 'message_template', 'channel'
            ]]

            # 4. ВІДОБРАЖЕННЯ В ТАБЛИЦІ
            for _, row in self.final_df.iterrows():
                self.tree.insert("", "end", values=(
                    row['customer_id'], row['rfm_segment'], row['favorite_category'], 
                    row['churn_risk_pct'], row['message_template'], row['channel']
                ))

            self.status_var.set(f"✅ Готово! Знайдено {len(self.final_df)} клієнтів для розсилки.")
            self.btn_export.config(state=tk.NORMAL)

        except FileNotFoundError as e:
            messagebox.showerror("Помилка файлу", f"Не знайдено файл: {e.filename}\nПереконайтесь, що дані підготовлені.")
            self.status_var.set("Помилка завантаження файлів.")
        except Exception as e:
            messagebox.showerror("Помилка", f"Сталася помилка під час обробки:\n{str(e)}")
            self.status_var.set("Помилка аналізу.")

    def export_csv(self):
        if self.final_df is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile="push_campaign_queue.csv",
                title="Зберегти базу",
                filetypes=[("CSV файли", "*.csv"), ("Всі файли", "*.*")]
            )
            if file_path:
                self.final_df.rename(columns={
                    'customer_id': 'ID Клієнта',
                    'rfm_segment': 'Сегмент (RFM)',
                    'favorite_category': 'Улюблена Категорія',
                    'churn_risk_pct': 'Ризик відтоку (%)',
                    'message_template': 'Меседж',
                    'channel': 'Канал'
                }).to_csv(file_path, index=False, encoding='utf-8-sig')
                
                messagebox.showinfo("Успіх", f"Базу успішно збережено за адресою:\n{file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LoyaltyGuardApp(root)
    root.mainloop()