import numpy as np
import streamlit as st
import pandas as pd
from kafka import KafkaProducer
import json
import time
import os
import uuid
from contextlib import closing
import psycopg2

# Конфигурация Kafka
KAFKA_CONFIG = {
    "bootstrap_servers": os.getenv("KAFKA_BROKERS", "kafka:9092"),
    "topic": os.getenv("KAFKA_TOPIC", "transactions")
}

# Конфигурация Postgres
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "fraud"),
    "user": os.getenv("POSTGRES_USER", "fraud"),
    "password": os.getenv("POSTGRES_PASSWORD", "fraud")
}

def load_file(uploaded_file):
    """Загрузка CSV файла в DataFrame"""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {str(e)}")
        return None


def get_last_frauds(limit=10):
    """Последние транзакции с флагом фрода."""
    query = """
        SELECT transaction_id, score, fraud_flag, created_at
        FROM scores
        WHERE fraud_flag = 1
        ORDER BY created_at DESC, id DESC
        LIMIT %s
    """
    with closing(psycopg2.connect(**POSTGRES_CONFIG)) as conn:
        return pd.read_sql_query(query, conn, params=(limit,))


def get_last_scores(limit=100):
    """Скоры последних транзакций для гистограммы."""
    query = """
        SELECT score
        FROM scores
        ORDER BY created_at DESC, id DESC
        LIMIT %s
    """
    with closing(psycopg2.connect(**POSTGRES_CONFIG)) as conn:
        return pd.read_sql_query(query, conn, params=(limit,))


def get_stats():
    """Сводка по витрине: сколько всего записей, сколько фродов и скорость поступления."""
    query = """
        SELECT
            count(*)                                                   AS total,
            sum(CASE WHEN fraud_flag = 1 THEN 1 ELSE 0 END)             AS frauds,
            min(created_at)                                            AS first_at,
            max(created_at)                                            AS last_at,
            sum(CASE WHEN created_at > now() - interval '1 minute'
                     THEN 1 ELSE 0 END)                                AS last_minute
        FROM scores
    """
    with closing(psycopg2.connect(**POSTGRES_CONFIG)) as conn:
        return pd.read_sql_query(query, conn).iloc[0]


def send_to_kafka(df, topic, bootstrap_servers):
    """Отправка данных в Kafka с уникальным ID транзакции"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            security_protocol="PLAINTEXT"
        )
        
        # Генерация уникальных ID для всех транзакций
        df['transaction_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        
        progress_bar = st.progress(0)
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            # Отправляем данные вместе с ID
            producer.send(
                topic, 
                value={
                    "transaction_id": row['transaction_id'],
                    "data": row.drop('transaction_id').to_dict()
                }
            )
            progress_bar.progress((idx + 1) / total_rows)
            time.sleep(0.01)
            
        producer.flush()
     
        return True
    except Exception as e:
        st.error(f"Ошибка отправки данных: {str(e)}")
        return False

# Инициализация состояния
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

if "results_visible" not in st.session_state:
    st.session_state.results_visible = False

# Интерфейс
st.title("📤 Отправка данных в Kafka")

# Блок загрузки файлов
uploaded_file = st.file_uploader(
    "Загрузите CSV файл с транзакциями",
    type=["csv"]
)

if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
    # Добавляем файл в состояние
    st.session_state.uploaded_files[uploaded_file.name] = {
        "status": "Загружен",
        "df": load_file(uploaded_file)
    }
    st.success(f"Файл {uploaded_file.name} успешно загружен!")

# Список загруженных файлов
if st.session_state.uploaded_files:
    st.subheader("🗂 Список загруженных файлов")
    
    for file_name, file_data in st.session_state.uploaded_files.items():
        cols = st.columns([4, 2, 2])
        
        with cols[0]:
            st.markdown(f"**Файл:** `{file_name}`")
            st.markdown(f"**Статус:** `{file_data['status']}`")
        
        with cols[2]:
            if st.button(f"Отправить {file_name}", key=f"send_{file_name}"):
                if file_data["df"] is not None:
                    with st.spinner("Отправка..."):
                        success = send_to_kafka(
                            file_data["df"],
                            KAFKA_CONFIG["topic"],
                            KAFKA_CONFIG["bootstrap_servers"]
                        )
                        if success:
                            st.session_state.uploaded_files[file_name]["status"] = "Отправлен"
                            st.rerun()
                else:
                    st.error("Файл не содержит данных")

# Раздел с результатами скоринга
st.divider()
st.subheader("📊 Результаты скоринга")

controls = st.columns([1, 1, 2])
with controls[0]:
    show_results = st.button("Посмотреть результаты")
with controls[1]:
    auto_refresh = st.toggle(
        "Автообновление",
        key="auto_refresh",
        help="Раз в 3 секунды подтягивать свежие результаты из базы",
    )

if show_results:
    # Включаем отображение раздела
    st.session_state.results_visible = True

if st.session_state.get("results_visible"):

    @st.fragment(run_every=3 if auto_refresh else None)
    def render_results():
        """Отрисовывает сводку и результаты; при автообновлении перечитывает базу."""
        try:
            stats = get_stats()
            frauds = get_last_frauds(10)
            scores = get_last_scores(100)
        except Exception as e:
            st.error(f"Не удалось получить данные из Postgres: {str(e)}")
            return

        # Сводные метрики по витрине
        metrics = st.columns(4)
        metrics[0].metric("Всего транзакций", f"{int(stats['total']):,}")
        metrics[1].metric("Флагов фрода", f"{int(stats['frauds']):,}")
        if stats["total"]:
            metrics[2].metric(
                "Доля фрода", f"{100 * stats['frauds'] / stats['total']:.2f}%"
            )
        else:
            metrics[2].metric("Доля фрода", "—")
        metrics[3].metric("За последнюю минуту", f"{int(stats['last_minute']):,}")

        # 1. Последние транзакции с флагом фрода
        st.markdown("**Последние 10 транзакций с флагом фрода**")

        if frauds.empty:
            st.info("Транзакции с флагом фрода не найдены.")
        else:
            display_df = frauds.copy()
            display_df["created_at"] = pd.to_datetime(
                display_df["created_at"]
            ).dt.strftime("%Y-%m-%d %H:%M:%S")
            display_df["score"] = display_df["score"].round(6)
            display_df.columns = ["ID транзакции", "Скор", "Флаг фрода", "Время"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # 2. Распределение скоров последних транзакций
        st.markdown("**Распределение скоров последних 100 транзакций**")

        if scores.empty:
            st.info("В базе пока нет транзакций.")
        else:
            st.caption(f"Транзакций в выборке: {len(scores)}")
            hist, edges = np.histogram(scores["score"], bins=20, range=(0, 1))
            chart_data = pd.DataFrame(
                {"Скор": [f"{edges[i]:.2f}–{edges[i + 1]:.2f}" for i in range(len(hist))],
                 "Транзакций": hist}
            ).set_index("Скор")
            st.bar_chart(chart_data, color="#4C78A8")

    render_results()