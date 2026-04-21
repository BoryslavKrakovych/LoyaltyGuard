import pandas as pd
import re
from collections import Counter
from gensim.models import Word2Vec

def build_user_profiles(df, model_path="models/word2vec.model"):
    print("   -> Профілювання інтересів клієнтів...")
    
    try:
        model = Word2Vec.load(model_path)
        vocab = set(model.wv.index_to_key) # Слова, які знає модель
    except FileNotFoundError:
        print("   [!] Word2Vec модель не знайдена. Використовується простий підрахунок.")
        vocab = None

    def tokenize(text):
        return [w for w in re.findall(r'[a-zA-Z0-9]+', str(text).lower()) if len(w) > 3]

    def get_favorite_category(items):
        words = []
        for item in items:
            tokens = tokenize(item)
            # Якщо модель є, беремо тільки ті слова, які вона "розуміє"
            if vocab:
                tokens = [t for t in tokens if t in vocab]
            words.extend(tokens)
            
        count = Counter(words)
        return count.most_common(1)[0][0] if count else "улюблені товари"

    # Групуємо всі покупки юзера і шукаємо головне слово
    profiles = df.groupby('customer_id')['product_name'].apply(list).reset_index()
    profiles['favorite_category'] = profiles['product_name'].apply(get_favorite_category)
    
    return profiles[['customer_id', 'favorite_category']]