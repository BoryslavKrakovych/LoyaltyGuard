from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.churn_model import ChurnModelService
from src.clustering import build_customer_clusters
from src.data_loader import load_csv_path, merge_customer_product_stats, standardize_dates
from src.feature_engineering import add_base_features, build_forward_churn_dataset, build_training_matrices
from src.product_analytics import cluster_products, customer_category_profile, train_word2vec
from src.rfm import build_rfm_table
from src.config import MODELS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transactions', required=True)
    parser.add_argument('--products', required=False, default=None)
    args = parser.parse_args()

    transactions = standardize_dates(load_csv_path(args.transactions))
    products = load_csv_path(args.products) if args.products else None

    merged = merge_customer_product_stats(transactions, products)
    featured = add_base_features(merged)
    train_featured = build_forward_churn_dataset(featured, horizon_days=90)

    X, y, groups, _ = build_training_matrices(train_featured)
    churn_service = ChurnModelService(use_xgboost=True)
    artifacts = churn_service.train(X, y, groups)

    rfm = build_rfm_table(featured)
    clusters = build_customer_clusters(featured)
    category_profile = customer_category_profile(products)
    word2vec = train_word2vec(products) if products is not None else None
    product_clusters = cluster_products(products, word2vec) if products is not None else None

    joblib.dump(artifacts, MODELS_DIR / 'churn_artifacts.joblib')
    joblib.dump(rfm, MODELS_DIR / 'rfm_table.joblib')
    joblib.dump(clusters, MODELS_DIR / 'customer_clusters.joblib')
    joblib.dump(category_profile, MODELS_DIR / 'category_profile.joblib')
    if word2vec is not None:
        word2vec.save(str(MODELS_DIR / 'word2vec.model'))
    if product_clusters is not None and len(product_clusters) > 0:
        product_clusters.to_csv(MODELS_DIR / 'product_clusters.csv', index=False)

    print('Artifacts saved into models/')
    print(f'Churn ROC-AUC: {artifacts.roc_auc:.4f}')


if __name__ == '__main__':
    main()
