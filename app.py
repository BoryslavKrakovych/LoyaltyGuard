from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.camp_push import append_feedback, filter_campaign_audience, generate_push_campaign
from src.modeler import train_churn_model, train_kmeans_model, train_word2vec_model
from src.rfm_cool import perform_churn_and_cooling_analysis, predict_single_customer_what_if
from src.word_vec import build_user_profiles
from src.xcelerator import (
    add_base_features,
    build_customer_dataset,
    build_forward_churn_dataset,
    build_training_matrices,
    load_and_prepare_transactions,
    load_table_file,
    merge_customer_product_stats,
    save_pipeline_datasets,
)

BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / 'streamlit_data'
MODELS_DIR = WORK_DIR / 'models'
FEEDBACK_FILE = WORK_DIR / 'feedback_log.csv'
WORK_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title='LoyaltyGuard', layout='wide')
st.title('LoyaltyGuard')
st.caption('Локальний аналітичний модуль для прогнозу відтоку та retention-кампаній у рітейлі')


def save_uploaded_file(uploaded_file, suffix: str) -> Path:
    safe_name = uploaded_file.name.replace(' ', '_')
    file_path = WORK_DIR / f'{suffix}_{safe_name}'
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


def build_pipeline(transactions_path: Path, products_path: Path | None, horizon_days: int, n_clusters: int):
    df, mapping, notes = load_and_prepare_transactions(transactions_path)

    if products_path is not None:
        products_df = load_table_file(products_path)
        df = merge_customer_product_stats(df, products_df)

    df = add_base_features(df)

    if 'churn' not in df.columns or df['churn'].nunique() < 2:
        df = build_forward_churn_dataset(df, horizon_days=horizon_days)
        notes.append(f'churn побудовано автоматично на горизонті {horizon_days} днів')

    customer_df = build_customer_dataset(df)
    X, y, _, _ = build_training_matrices(df)

    saved = save_pipeline_datasets(df, customer_df, X, y, WORK_DIR)

    churn_model_path = MODELS_DIR / 'churn_model.joblib'
    word2vec_model_path = MODELS_DIR / 'word2vec.model'
    kmeans_model_path = MODELS_DIR / 'kmeans_model.joblib'
    rescue_queue_path = WORK_DIR / 'rescue_queue.csv'
    campaign_path = WORK_DIR / 'push_campaign_ready.csv'
    campaign_json_path = WORK_DIR / 'push_campaign_ready.json'

    train_churn_model(saved['x_path'], saved['y_path'], churn_model_path)
    train_word2vec_model(saved['featured_path'], word2vec_model_path)
    train_kmeans_model(saved['customer_path'], kmeans_model_path, n_clusters=n_clusters)

    analysis_df = perform_churn_and_cooling_analysis(
        df,
        model_path=str(churn_model_path),
        kmeans_model_path=str(kmeans_model_path),
    )
    profiles_df = build_user_profiles(df, model_path=str(word2vec_model_path))
    campaign_df = generate_push_campaign(
        analysis_df,
        profiles_df,
        feedback_path=str(FEEDBACK_FILE),
    )

    analysis_df.to_csv(rescue_queue_path, index=False)
    campaign_df.to_csv(campaign_path, index=False)
    campaign_df.to_json(campaign_json_path, orient='records', force_ascii=False, indent=2)

    return {
        'featured_df': df,
        'customer_df': customer_df,
        'analysis_df': analysis_df,
        'campaign_df': campaign_df,
        'mapping': mapping,
        'notes': notes,
        'paths': {
            **saved,
            'churn_model_path': churn_model_path,
            'word2vec_model_path': word2vec_model_path,
            'kmeans_model_path': kmeans_model_path,
            'rescue_queue_path': rescue_queue_path,
            'campaign_path': campaign_path,
            'campaign_json_path': campaign_json_path,
        },
    }


with st.sidebar:
    st.header('Параметри')
    horizon_days = st.number_input('Горизонт прогнозу відтоку, днів', min_value=30, max_value=365, value=90, step=10)
    n_clusters = st.slider('Кількість кластерів K-means', min_value=1, max_value=10, value=5, step=1)

transactions_file = st.file_uploader('Файл транзакцій (CSV, XLSX, XLS)', type=['csv', 'xlsx', 'xls'])
products_file = st.file_uploader('Файл продуктів (опціонально)', type=['csv', 'xlsx', 'xls'])
run_button = st.button('Запустити аналіз', type='primary', use_container_width=True)

if 'pipeline_result' not in st.session_state:
    st.session_state['pipeline_result'] = None

if run_button:
    if transactions_file is None:
        st.error('Завантаж файл транзакцій.')
    else:
        transactions_path = save_uploaded_file(transactions_file, 'transactions')
        products_path = save_uploaded_file(products_file, 'products') if products_file is not None else None

        try:
            with st.spinner('Обробка даних, навчання моделей і побудова кампанії...'):
                st.session_state['pipeline_result'] = build_pipeline(
                    transactions_path=transactions_path,
                    products_path=products_path,
                    horizon_days=int(horizon_days),
                    n_clusters=int(n_clusters),
                )
            st.success('Готово.')
        except Exception as error:
            st.exception(error)

result = st.session_state.get('pipeline_result')

if result is None:
    st.info('Завантаж файл транзакцій і натисни "Запустити аналіз".')
else:
    featured_df = result['featured_df']
    customer_df = result['customer_df']
    analysis_df = result['analysis_df']
    campaign_df = result['campaign_df']
    mapping = result['mapping']
    notes = result['notes']
    paths = result['paths']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Клієнтів', int(featured_df['customer_id'].nunique()) if 'customer_id' in featured_df.columns else 0)
    col2.metric('Рядків транзакцій', len(featured_df))
    col3.metric('У rescue queue', int(len(analysis_df[analysis_df['is_target']])) if 'is_target' in analysis_df.columns else len(analysis_df))
    col4.metric('У кампанії', len(campaign_df))

    tab_rfm, tab_churn, tab_clusters, tab_rescue, tab_what_if, tab_campaign, tab_feedback = st.tabs([
     'RFM', 'Churn', 'Кластери', 'Черга на порятунок', 'What-if', 'Конструктор кампаній', 'Фідбек'
    ])

    # with tab_data:
    #     # st.subheader('Мапінг колонок')
    #     # st.json(mapping if mapping else {})
    #     #
    #     # st.subheader('Примітки підготовки')
    #     # if notes:
    #     #     for note in notes:
    #     #         st.write('-', note)
    #     # else:
    #     #     st.write('Додаткових приміток немає.')
    #
    #     st.subheader('Підготовлені транзакції')
    #     st.dataframe(featured_df.head(100), use_container_width=True)
    #
    #     st.download_button(
    #         'Завантажити featured_data.csv',
    #         data=paths['featured_path'].read_bytes(),
    #         file_name='featured_data.csv',
    #         mime='text/csv',
    #         use_container_width=True,
    #     )

    with tab_rfm:
        st.subheader('RFM-профіль клієнтів')
        rfm_cols = ['customer_id', 'Recency', 'Frequency', 'Monetary', 'R_score', 'F_score', 'M_score', 'RFM_score', 'rfm_segment']
        available_cols = [col for col in rfm_cols if col in customer_df.columns]
        st.dataframe(customer_df[available_cols].head(100), use_container_width=True)
        if 'rfm_segment' in customer_df.columns:
            st.subheader('Кількість клієнтів за сегментами')
            st.dataframe(customer_df['rfm_segment'].value_counts(dropna=False).rename_axis('rfm_segment').reset_index(name='customers'), use_container_width=True)

    with tab_churn:
        st.subheader('Прогноз відтоку')
        churn_cols = [
            'customer_id', 'churn_probability', 'risk_class', 'cooling_flag',
            'frequency_drop_flag', 'basket_drop_flag', 'amount_drop_flag',
            'recommended_action'
        ]
        available_cols = [col for col in churn_cols if col in analysis_df.columns]
        show_df = analysis_df[available_cols].copy()
        if 'churn_probability' in show_df.columns:
            show_df['churn_probability'] = (show_df['churn_probability'] * 100).round(1).astype(str) + '%'
        st.dataframe(show_df.head(100), use_container_width=True)

    with tab_clusters:
        st.subheader('K-means сегментація клієнтів')
        cluster_cols = ['customer_id', 'cluster', 'cluster_name', 'Monetary', 'Recency', 'Frequency']
        available_cols = [col for col in cluster_cols if col in analysis_df.columns]
        st.dataframe(analysis_df[available_cols].head(100), use_container_width=True)
        if 'cluster_name' in analysis_df.columns:
            st.subheader('Розподіл за кластерами')
            st.dataframe(analysis_df['cluster_name'].value_counts(dropna=False).rename_axis('cluster_name').reset_index(name='customers'), use_container_width=True)

    with tab_rescue:
        st.subheader('Черга на порятунок')
        queue = analysis_df[analysis_df['is_target']].copy() if 'is_target' in analysis_df.columns else analysis_df.copy()
        st.dataframe(queue, use_container_width=True)

        col_csv, col_json = st.columns(2)
        col_csv.download_button(
            'Завантажити rescue_queue.csv',
            data=paths['rescue_queue_path'].read_bytes(),
            file_name='rescue_queue.csv',
            mime='text/csv',
            use_container_width=True,
        )
        col_json.download_button(
            'Завантажити push_campaign_ready.json',
            data=paths['campaign_json_path'].read_bytes(),
            file_name='push_campaign_ready.json',
            mime='application/json',
            use_container_width=True,
        )

    with tab_what_if:
        st.subheader('What-if симулятор')
        if customer_df.empty:
            st.info('Немає даних для симуляції.')
        else:
            customer_options = customer_df['customer_id'].astype(str).tolist()
            selected_customer = st.selectbox('Оберіть клієнта', customer_options)

            ticket_change = st.slider(
                'Зміна середнього чека, %',
                min_value=-50,
                max_value=50,
                value=10,
                step=5,
            )

            base_row = customer_df[customer_df['customer_id'].astype(str) == str(selected_customer)].head(1)
            current_row = analysis_df[analysis_df['customer_id'].astype(str) == str(selected_customer)].head(1)

            if not base_row.empty:
                simulation = predict_single_customer_what_if(
                    customer_row=base_row.iloc[0],
                    model_path=str(paths['churn_model_path']),
                    frequency_change_pct=0.0,
                    ticket_change_pct=float(ticket_change),
                    promo_change_pp=0.0,
                )

                if not current_row.empty and 'churn_probability' in current_row.columns:
                    st.write('Поточна ймовірність відтоку:', f"{current_row.iloc[0]['churn_probability'] * 100:.1f}%")
                else:
                    st.write('Поточна ймовірність відтоку:', 'н/д')

                st.write('Ймовірність після зміни:', f"{simulation['churn_probability'] * 100:.1f}%")
                st.write('Новий клас ризику:', simulation['risk_class'])

    with tab_campaign:
        st.subheader('Конструктор кампаній')
        risk_options = ['Усі'] + sorted(campaign_df['risk_class'].dropna().unique().tolist()) if 'risk_class' in campaign_df.columns else ['Усі']
        rfm_options = ['Усі'] + sorted(campaign_df['rfm_segment'].dropna().unique().tolist()) if 'rfm_segment' in campaign_df.columns else ['Усі']
        category_options = ['Усі'] + sorted(campaign_df['favorite_category'].dropna().unique().tolist()) if 'favorite_category' in campaign_df.columns else ['Усі']
        cluster_options = ['Усі'] + sorted(campaign_df['cluster_name'].dropna().unique().tolist()) if 'cluster_name' in campaign_df.columns else ['Усі']

        selected_risk = st.selectbox('Risk class', risk_options)
        selected_rfm = st.selectbox('RFM segment', rfm_options)
        selected_category = st.selectbox('Категорія', category_options)
        selected_cluster = st.selectbox('Кластер', cluster_options)

        filtered_campaign = filter_campaign_audience(
            campaign_df,
            risk_class=selected_risk,
            rfm_segment=selected_rfm,
            category=selected_category,
            cluster_name=selected_cluster,
        )
        st.dataframe(filtered_campaign, use_container_width=True)

        csv_bytes = filtered_campaign.to_csv(index=False).encode('utf-8')
        st.download_button(
            'Завантажити відфільтровану кампанію CSV',
            data=csv_bytes,
            file_name='filtered_campaign.csv',
            mime='text/csv',
            use_container_width=True,
        )

    with tab_feedback:
        st.subheader('Індивідуальні тригери та фідбек')
        feedback_customer = st.text_input('customer_id')
        feedback_category = st.text_input('category')
        feedback_type = st.selectbox('feedback_type', ['not_interested', 'already_bought'])
        feedback_date = st.date_input('date')

        if st.button('Зберегти фідбек'):
            if feedback_customer and feedback_category:
                append_feedback(
                    feedback_path=FEEDBACK_FILE,
                    customer_id=str(feedback_customer),
                    category=str(feedback_category),
                    feedback_type=str(feedback_type),
                    date_value=str(feedback_date),
                )
                st.success('Фідбек збережено.')
            else:
                st.error('Вкажи customer_id і category.')

        if FEEDBACK_FILE.exists():
            feedback_df = pd.read_csv(FEEDBACK_FILE)
            st.dataframe(feedback_df, use_container_width=True)
