import pandas as pd
import joblib
import re
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from gensim.models import Word2Vec

def train_churn_model(x_path, y_path, output_path):
    print("1. Тренування моделі відтоку (Churn)...")
    try:
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
    except FileNotFoundError:
        print(f"❌ Помилка: Не знайдено файли {x_path} або {y_path}. Запустіть xcelerator!")
        return

    # Ініціалізуємо градієнтний бустинг (або XGBoost, якщо він встановлений)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
    model.fit(X, y.values.ravel())
    
    # Зберігаємо не тільки модель, а й список колонок, 
    # щоб при інференсі X_new мав такий самий порядок фіч
    artifacts = {
        'model': model,
        'features': list(X.columns)
    }
    joblib.dump(artifacts, output_path)
    print(f"   ✅ Модель відтоку натренована та збережена у: {output_path}")

def train_word2vec_model(data_path, output_path):
    print("2. Тренування семантичної моделі товарів (Word2Vec)...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Помилка: Не знайдено файл {data_path}.")
        return

    # Витягуємо та токенізуємо унікальні назви товарів
    def tokenize(text):
        return [w for w in re.findall(r'[a-zA-Z0-9]+', str(text).lower()) if len(w) > 3]
    
    products = df['product_name'].dropna().unique()
    sentences = [tokenize(p) for p in products]
    sentences = [s for s in sentences if s] # Відкидаємо порожні
    
    if len(sentences) < 5:
        print("   ⚠️ Замало даних для Word2Vec. Пропускаємо.")
        return

    # Тренуємо Gensim Word2Vec
    model = Word2Vec(sentences, vector_size=20, window=3, min_count=1, workers=1)
    model.save(str(output_path))
    print(f"   ✅ Word2Vec натренована та збережена у: {output_path}")

if __name__ == "__main__":
    print("=== ЗАПУСК LOYALTY MODELER ===")
    
    # Створюємо папку для моделей, якщо її немає
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Шляхи до файлів, які згенерував xcelerator
    X_TRAIN_FILE = "X_train.csv"
    Y_TRAIN_FILE = "y_train.csv"
    FEATURED_FILE = "featured_data.csv" # Використовуємо для витягування назв товарів
    
    # Тренуємо та зберігаємо
    train_churn_model(X_TRAIN_FILE, Y_TRAIN_FILE, models_dir / "churn_model.joblib")
    train_word2vec_model(FEATURED_FILE, models_dir / "word2vec.model")
    
    print("\nВсі моделі успішно оновлені! Тепер можна запускати щоденний main.py.")