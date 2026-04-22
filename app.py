from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.camp_push import append_feedback, filter_campaign_audience, generate_push_campaign
from core_ml import ChurnModelService, build_customer_clusters, build_rfm_table
from src.rfm_cool import (
    build_customer_dataset,
    perform_churn_and_cooling_analysis,
    predict_single_customer_what_if,
)
from src.word_vec import build_user_profiles
from xcelerator import (
    add_base_features,
    build_forward_churn_dataset,
    build_training_matrices,
    load_and_prepare_transactions,
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


def save_pipeline_tables(
    df: pd.DataFrame,
    customer_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Path]:
    featured_path = WORK_DIR / 'featured_data.csv'
    customer_path = WORK_DIR / 'customer_level_data.csv'
    x_path = WORK_DIR / 'X_train.csv'
    y_path = WORK_DIR / 'y_train.csv'

    df.to_csv(featured_path, index=False)
    customer_df.to_csv(customer_path, index=False)
    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)

    return {
        'featured_path': featured_path,
        'customer_path': customer_path,
        'x_path': x_path,
        'y_path': y_path,
    }


def build_pipeline(
    transactions_path: Path,
    horizon_days: int,
    n_clusters: int,
) -> dict:
    df, mapping, notes = load_and_prepare_transactions(transactions_path)

    df = add_base_features(df)

    if 'churn' not in df.columns or df['churn'].nunique() < 2:
        df = build_forward_churn_dataset(df, horizon_days=horizon_days)
        notes.append(f'churn побудовано автоматично на горизонті {horizon_days} днів')

    X, y, groups, _ = build_training_matrices(df)

    churn_service = ChurnModelService(use_xgboost=True)
    churn_artifacts = churn_service.train(X, y, groups, test_size=0.15)

    rfm_df = build_rfm_table(df)
    clusters_df = build_customer_clusters(df, n_clusters=n_clusters)
    customer_df = build_customer_dataset(df, rfm_df=rfm_df, clusters_df=clusters_df)

    saved = save_pipeline_tables(df, customer_df, X, y)

    churn_model_path = MODELS_DIR / 'churn_artifacts.joblib'
    rfm_table_path = MODELS_DIR / 'rfm_table.joblib'
    customer_clusters_path = MODELS_DIR / 'customer_clusters.joblib'
    word2vec_model_path = MODELS_DIR / 'word2vec.model'
    rescue_queue_path = WORK_DIR / 'rescue_queue.csv'
    campaign_path = WORK_DIR / 'push_campaign_ready.csv'
    campaign_json_path = WORK_DIR / 'push_campaign_ready.json'

    joblib.dump(churn_artifacts, churn_model_path)
    joblib.dump(rfm_df, rfm_table_path)
    joblib.dump(clusters_df, customer_clusters_path)

    analysis_df = perform_churn_and_cooling_analysis(
        df,
        model_path=str(churn_model_path),
        rfm_path=str(rfm_table_path),
        clusters_path=str(customer_clusters_path),
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
            'rfm_table_path': rfm_table_path,
            'customer_clusters_path': customer_clusters_path,
            'word2vec_model_path': word2vec_model_path,
            'rescue_queue_path': rescue_queue_path,
            'campaign_path': campaign_path,
            'campaign_json_path': campaign_json_path,
        },
    }


def _clean_category_value(value: object) -> str:
    if pd.isna(value):
        return ''
    text = str(value).strip()
    if text.lower() in {'', '<na>', 'nan', 'none'}:
        return ''
    return text


def _build_preview_message(risk_pct: float, rfm_segment: str, favorite_category: str) -> tuple[str, str]:
    if risk_pct > 70 and 'At Risk' in rfm_segment:
        return f'🔥 Повертайтесь! Знижка -20% на {favorite_category}', 'Push + SMS'
    if risk_pct > 50:
        return f'✨ Ми зібрали для вас новинки: {favorite_category}', 'Push'
    return f'👋 Здається, вам сподобається: {favorite_category}', 'Push'


@st.fragment
def render_campaign_page(campaign_df: pd.DataFrame) -> None:
    st.subheader('Конструктор кампаній')

    if campaign_df.empty:
        st.warning('Немає клієнтів з визначеними категоріями для кампанії.')
        return

    risk_options = ['Усі'] + sorted(campaign_df['risk_class'].dropna().unique().tolist()) if 'risk_class' in campaign_df.columns else ['Усі']
    rfm_options = ['Усі'] + sorted(campaign_df['rfm_segment'].dropna().unique().tolist()) if 'rfm_segment' in campaign_df.columns else ['Усі']
    category_options = ['Усі'] + sorted(campaign_df['favorite_category'].dropna().unique().tolist()) if 'favorite_category' in campaign_df.columns else ['Усі']
    cluster_options = ['Усі'] + sorted(campaign_df['cluster_name'].dropna().unique().tolist()) if 'cluster_name' in campaign_df.columns else ['Усі']

    filt1, filt2, filt3, filt4 = st.columns(4)
    selected_risk = filt1.selectbox('Risk class', risk_options, key='campaign_risk')
    selected_rfm = filt2.selectbox('RFM segment', rfm_options, key='campaign_rfm')
    selected_category = filt3.selectbox('Категорія', category_options, key='campaign_category')
    selected_cluster = filt4.selectbox('Кластер', cluster_options, key='campaign_cluster')

    filtered_campaign = filter_campaign_audience(
        campaign_df,
        risk_class=selected_risk,
        rfm_segment=selected_rfm,
        category=selected_category,
        cluster_name=selected_cluster,
    )

    pretty_campaign = filtered_campaign.rename(columns={
        'customer_id': 'ID Клієнта',
        'rfm_segment': 'Сегмент (RFM)',
        'favorite_category': 'Улюблена Категорія',
        'churn_probability_pct': 'Ризик (%)',
        'push_text': 'Рекомендований Меседж',
        'channel': 'Канал',
    })

    campaign_cols = [
        'ID Клієнта',
        'Сегмент (RFM)',
        'Улюблена Категорія',
        'Ризик (%)',
        'Рекомендований Меседж',
        'Канал',
    ]
    available_cols = [col for col in campaign_cols if col in pretty_campaign.columns]

    st.dataframe(pretty_campaign[available_cols], use_container_width=True)

    csv_bytes = pretty_campaign[available_cols].to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        'Зберегти базу для Push-розсилки (CSV)',
        data=csv_bytes,
        file_name='push_campaign_queue.csv',
        mime='text/csv',
        use_container_width=True,
    )


@st.fragment
def render_what_if_page(
    customer_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    campaign_df: pd.DataFrame,
    paths: dict,
) -> None:
    st.subheader('What-if симулятор')

    if customer_df.empty:
        st.info('Немає даних для симуляції.')
        return

    customer_options = customer_df['customer_id'].astype(str).tolist()

    selected_customer = st.selectbox(
        'Оберіть клієнта',
        customer_options,
        key='what_if_customer',
    )

    ticket_change = st.slider(
        'Зміна середнього чека, %',
        min_value=-50,
        max_value=50,
        value=10,
        step=5,
        key='what_if_ticket_change',
    )

    base_row = customer_df[customer_df['customer_id'].astype(str) == str(selected_customer)].head(1)
    current_row = analysis_df[analysis_df['customer_id'].astype(str) == str(selected_customer)].head(1)

    if base_row.empty:
        st.warning('Клієнта не знайдено для симуляції.')
        return

    simulation = predict_single_customer_what_if(
        customer_row=base_row.iloc[0],
        model_path=str(paths['churn_model_path']),
        frequency_change_pct=0.0,
        ticket_change_pct=float(ticket_change),
        promo_change_pp=0.0,
    )

    if not current_row.empty and 'churn_probability' in current_row.columns:
        current_probability = float(current_row.iloc[0]['churn_probability']) * 100
        st.write('Поточна ймовірність відтоку:', f'{current_probability:.1f}%')
    else:
        st.write('Поточна ймовірність відтоку:', 'н/д')

    st.write('Ймовірність після зміни:', f"{simulation['churn_probability'] * 100:.1f}%")
    st.write('Новий клас ризику:', simulation['risk_class'])

    favorite_category = _clean_category_value(base_row.iloc[0].get('dominant_category', ''))

    if not favorite_category and 'favorite_category' in campaign_df.columns:
        found = campaign_df[campaign_df['customer_id'].astype(str) == str(selected_customer)]
        if not found.empty:
            favorite_category = _clean_category_value(found.iloc[0].get('favorite_category', ''))

    if not favorite_category:
        favorite_category = 'категорія не визначена'

    rfm_segment = str(base_row.iloc[0].get('rfm_segment', ''))
    new_risk_pct = simulation['churn_probability'] * 100

    preview_text, preview_channel = _build_preview_message(
        risk_pct=new_risk_pct,
        rfm_segment=rfm_segment,
        favorite_category=favorite_category,
    )

    st.markdown('**Preview push-повідомлення після зміни:**')
    st.write(preview_text)
    st.write('Канал:', preview_channel)


@st.fragment
def render_feedback_page() -> None:
    st.subheader('Індивідуальні тригери та фідбек')

    feedback_customer = st.text_input('customer_id', key='feedback_customer')
    feedback_category = st.text_input('category', key='feedback_category')
    feedback_type = st.selectbox(
        'feedback_type',
        ['not_interested', 'already_bought'],
        key='feedback_type',
    )
    feedback_date = st.date_input('date', key='feedback_date')

    if st.button('Зберегти фідбек', key='save_feedback_button'):
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


with st.sidebar:
    st.header('Параметри')
    horizon_days = st.number_input(
        'Горизонт прогнозу відтоку, днів',
        min_value=30,
        max_value=365,
        value=90,
        step=10,
    )
    n_clusters = st.slider(
        'Кількість кластерів K-means',
        min_value=2,
        max_value=10,
        value=5,
        step=1,
    )

transactions_file = st.file_uploader(
    'Файл транзакцій (CSV, XLSX, XLS)',
    type=['csv', 'xlsx', 'xls'],
)

run_button = st.button('Запустити аналіз', type='primary', use_container_width=True)

if 'pipeline_result' not in st.session_state:
    st.session_state['pipeline_result'] = None

if 'active_page' not in st.session_state:
    st.session_state['active_page'] = 'Конструктор кампаній'

if run_button:
    if transactions_file is None:
        st.error('Завантаж файл транзакцій.')
    else:
        transactions_path = save_uploaded_file(transactions_file, 'transactions')

        try:
            with st.spinner('Обробка даних, навчання моделей та побудова кампанії...'):
                st.session_state['pipeline_result'] = build_pipeline(
                    transactions_path=transactions_path,
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
    paths = result['paths']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Клієнтів', int(featured_df['customer_id'].nunique()) if 'customer_id' in featured_df.columns else 0)
    col2.metric('Рядків транзакцій', len(featured_df))
    col3.metric(
        'У rescue queue',
        int(len(analysis_df[analysis_df['is_target']])) if 'is_target' in analysis_df.columns else len(analysis_df),
    )
    col4.metric('У кампанії', len(campaign_df))

    selected_page = st.radio(
        'Розділ',
        ['Конструктор кампаній', 'What-if', 'Фідбек'],
        horizontal=True,
        key='active_page',
    )

    if selected_page == 'Конструктор кампаній':
        render_campaign_page(campaign_df)
    elif selected_page == 'What-if':
        render_what_if_page(
            customer_df=customer_df,
            analysis_df=analysis_df,
            campaign_df=campaign_df,
            paths=paths,
        )
    else:
        render_feedback_page()