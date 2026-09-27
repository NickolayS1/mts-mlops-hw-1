import numpy as np
import streamlit as st
import pandas as pd
from kafka import KafkaProducer
import json
import time
import os
import uuid
import threading
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


def get_recent_transactions(limit=15):
    """Последние обработанные транзакции — чтобы видеть сам поток."""
    query = """
        SELECT created_at, transaction_id, score, fraud_flag
        FROM scores
        ORDER BY id DESC
        LIMIT %s
    """
    with closing(psycopg2.connect(**POSTGRES_CONFIG)) as conn:
        return pd.read_sql_query(query, conn, params=(limit,))


def get_stats():
    """Сводка по витрине: всего записей, фродов и как давно пришла последняя.

    sum() по пустой таблице возвращает NULL, поэтому оборачиваем его в
    coalesce — на пустой витрине нужен ноль, а не None.
    """
    query = """
        SELECT
            count(*)                                                          AS total,
            coalesce(sum(CASE WHEN fraud_flag = 1 THEN 1 ELSE 0 END), 0)       AS frauds,
            extract(epoch FROM (now() - max(created_at)))                     AS seconds_since_last
        FROM scores
    """
    with closing(psycopg2.connect(**POSTGRES_CONFIG)) as conn:
        return pd.read_sql_query(query, conn).iloc[0]


def fetch_results():
    """Один заход в базу за всем, что нужно разделу результатов."""
    return {
        "stats": get_stats(),
        "frauds": get_last_frauds(10),
        "scores": get_last_scores(100),
        "recent": get_recent_transactions(15),
    }


def clear_scores():
    """Очищает витрину, чтобы начать наблюдение с чистого листа."""
    with closing(psycopg2.connect(**POSTGRES_CONFIG)) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE scores RESTART IDENTITY")
        conn.commit()


def humanize_age(seconds):
    """«3 сек назад» вместо голого числа."""
    if seconds is None or pd.isna(seconds):
        return "—"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f} сек назад"
    if seconds < 3600:
        return f"{seconds / 60:.0f} мин назад"
    return f"{seconds / 3600:.1f} ч назад"


# Состояние фоновых отправок. Общее для всех перерисовок скрипта,
# поэтому хранится на уровне модуля, а не в session_state.
_send_lock = threading.Lock()
_send_state = {}


def get_send_state(file_name):
    """Снимок прогресса отправки файла (или None, если отправка не запускалась)."""
    with _send_lock:
        state = _send_state.get(file_name)
        return dict(state) if state else None


def is_sending(file_name):
    state = get_send_state(file_name)
    return bool(state) and not state["done"]


def any_sending():
    """Идёт ли сейчас хоть одна отправка — по этому флагу обновляется раздел результатов."""
    with _send_lock:
        return any(not s["done"] for s in _send_state.values())


def _send_worker(file_name, df, topic, bootstrap_servers, delay):
    """Отправляет строки в Kafka в отдельном потоке.

    Скрипт Streamlit выполняется сверху вниз в одном потоке: если слать
    прямо в нём, страница замирает на всё время отправки и графики не
    обновляются. Поэтому отправка вынесена в фоновый поток, а UI лишь
    читает её прогресс.
    """
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            security_protocol="PLAINTEXT"
        )
        total = len(df)
        for idx, (_, row) in enumerate(df.iterrows()):
            producer.send(
                topic,
                value={
                    "transaction_id": str(uuid.uuid4()),
                    "data": row.to_dict(),
                }
            )
            with _send_lock:
                _send_state[file_name]["sent"] = idx + 1
            if delay:
                time.sleep(delay)
        producer.flush()
        with _send_lock:
            _send_state[file_name]["done"] = True
    except Exception as e:
        with _send_lock:
            state = _send_state.setdefault(file_name, {"sent": 0, "total": len(df)})
            state["error"] = str(e)
            state["done"] = True


def start_sending(file_name, df, topic, bootstrap_servers, delay):
    """Запускает фоновую отправку, если она ещё не идёт."""
    if is_sending(file_name):
        return
    with _send_lock:
        _send_state[file_name] = {
            "sent": 0, "total": len(df), "done": False, "error": None
        }
    # Новая отправка — файл снова «не завершён»
    st.session_state.settled_sends.discard(file_name)
    thread = threading.Thread(
        target=_send_worker,
        args=(file_name, df, topic, bootstrap_servers, delay),
        daemon=True,
    )
    thread.start()


def settle_finished_sends():
    """Останавливает автообновление, когда отправка закончилась.

    run_every фрагмента вычисляется в момент его создания, поэтому после
    завершения отправки фрагмент продолжал бы перерисовываться каждую
    секунду. Здесь мы один раз делаем полный прогон скрипта, чтобы
    run_every пересчитался и стал None.

    Вызывается из тела фрагментов: при их перерисовке скрипт целиком не
    выполняется, поэтому проверку нужно делать именно там.
    """
    with _send_lock:
        done = {name for name, s in _send_state.items() if s["done"]}
    unsettled = done - st.session_state.settled_sends
    if unsettled:
        st.session_state.settled_sends |= unsettled
        st.rerun(scope="app")

# Инициализация состояния
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

if "results_visible" not in st.session_state:
    st.session_state.results_visible = False

if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

# Имена файлов, для которых автообновление уже остановлено
if "settled_sends" not in st.session_state:
    st.session_state.settled_sends = set()

# Останавливаем автообновление, если фоновая отправка завершилась
settle_finished_sends()

# Интерфейс
st.title("Скоринг транзакций")
st.caption(
    "Загруженные транзакции уходят в Kafka, оцениваются моделью и попадают в Postgres"
)

# Блок загрузки файлов
uploaded_file = st.file_uploader(
    "CSV файл с транзакциями",
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
    st.subheader("Загруженные файлы")

    # Скорость отправки: без паузы пачка уходит быстрее, чем её можно увидеть
    st.slider(
        "Пауза между сообщениями, мс",
        min_value=0,
        max_value=200,
        value=25,
        step=5,
        key="send_delay_ms",
        help="Замедляет отправку, чтобы поток было видно в разделе результатов",
    )

    @st.fragment(run_every=1 if any_sending() else None)
    def render_files():
        """Список файлов и прогресс отправки. Пока идёт отправка, обновляется
        раз в секунду — иначе полоса прогресса замирает на первом кадре."""
        for file_name, file_data in st.session_state.uploaded_files.items():
            cols = st.columns([4, 2, 2])

            with cols[0]:
                st.markdown(f"**Файл:** `{file_name}`")
                st.markdown(f"**Статус:** `{file_data['status']}`")

            with cols[2]:
                if st.button(f"Отправить {file_name}", key=f"send_{file_name}"):
                    if file_data["df"] is not None:
                        # Отправка идёт в фоне, чтобы интерфейс не замирал
                        start_sending(
                            file_name,
                            file_data["df"],
                            KAFKA_CONFIG["topic"],
                            KAFKA_CONFIG["bootstrap_servers"],
                            delay=st.session_state.send_delay_ms / 1000.0,
                        )
                        # Перерисовываем весь скрипт: run_every фрагментов
                        # вычисляется только при полном прогоне
                        st.rerun(scope="app")
                    else:
                        st.error("Файл не содержит данных")

            state = get_send_state(file_name)
            if state:
                if state.get("error"):
                    st.error(f"Ошибка отправки: {state['error']}")
                else:
                    sent, total = state["sent"], state["total"]
                    st.progress(sent / total if total else 0.0)
                    if state["done"]:
                        st.caption(f"Отправлено {sent} из {total}")
                    else:
                        st.caption(f"Отправка... {sent} из {total}")

        # Отправка закончилась — гасим автообновление фрагментов
        settle_finished_sends()

    render_files()


# Раздел с результатами скоринга
st.divider()
st.subheader("Результаты скоринга")

# Управление объявлено до фрагмента: иначе значение переключателя
# на момент создания фрагмента ещё неизвестно
controls = st.columns([1, 1, 1, 2])
with controls[0]:
    show_results = st.button("Посмотреть результаты")
with controls[1]:
    auto_refresh = st.toggle(
        "Автообновление",
        key="auto_refresh",
        help="Обновлять раздел каждые 2 секунды",
    )
with controls[2]:
    reset_clicked = st.button("Очистить историю")

if show_results:
    st.session_state.results_visible = True

# Очистка подтверждается вторым нажатием, чтобы не стереть данные случайно
if reset_clicked:
    st.session_state.confirm_reset = True

if st.session_state.get("confirm_reset"):
    st.warning("Удалить все записи из витрины? Действие необратимо.")
    confirm_cols = st.columns([1, 1, 4])
    with confirm_cols[0]:
        if st.button("Да, удалить", type="primary"):
            try:
                clear_scores()
                st.session_state.confirm_reset = False
                st.success("История очищена.")
                st.rerun()
            except Exception as e:
                st.error(f"Не удалось очистить историю: {str(e)}")
    with confirm_cols[1]:
        if st.button("Отмена"):
            st.session_state.confirm_reset = False
            st.rerun()


def draw_results():
    """Рисует сводку, ленту и графики по данным из Postgres."""
    try:
        data = fetch_results()
    except Exception as e:
        st.error(f"Не удалось получить данные из Postgres: {str(e)}")
        return

    stats, frauds, scores, recent = (
        data["stats"], data["frauds"], data["scores"], data["recent"]
    )

    # Сводные метрики по витрине
    metrics = st.columns(4)
    metrics[0].metric("Всего транзакций", f"{int(stats['total']):,}")
    metrics[1].metric("Фродовых транзакций", f"{int(stats['frauds']):,}")
    if stats["total"]:
        metrics[2].metric("Доля фрода", f"{100 * stats['frauds'] / stats['total']:.2f}%")
    else:
        metrics[2].metric("Доля фрода", "—")
    metrics[3].metric("Последняя запись", humanize_age(stats["seconds_since_last"]))

    # 1. Лента последних транзакций — по ней видно сам поток
    st.markdown("**Последние обработанные транзакции**")

    if recent.empty:
        st.info("В базе пока нет транзакций.")
    else:
        feed = recent.copy()
        feed["created_at"] = pd.to_datetime(feed["created_at"]).dt.strftime("%H:%M:%S")
        feed["score"] = feed["score"].round(4)
        feed["fraud_flag"] = feed["fraud_flag"].map({1: "фрод", 0: "норма"})
        feed["transaction_id"] = feed["transaction_id"].str.slice(0, 8) + "…"
        feed.columns = ["Время", "ID транзакции", "Вероятность", "Результат"]
        st.dataframe(feed, use_container_width=True, hide_index=True)

    # 2. Последние транзакции с флагом фрода
    st.markdown("**Последние 10 фродовых транзакций**")

    if frauds.empty:
        st.info("Фродовых транзакций не найдено.")
    else:
        display_df = frauds.copy()
        display_df["created_at"] = pd.to_datetime(
            display_df["created_at"]
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
        display_df["score"] = display_df["score"].round(6)
        display_df.columns = ["ID транзакции", "Вероятность", "Фрод", "Время"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 3. Распределение вероятностей последних транзакций
    st.markdown("**Распределение вероятностей последних 100 транзакций**")

    if scores.empty:
        st.info("В базе пока нет транзакций.")
    else:
        st.caption(f"Транзакций в выборке: {len(scores)}")
        hist, edges = np.histogram(scores["score"], bins=20, range=(0, 1))
        chart_data = pd.DataFrame(
            {"Вероятность": [f"{edges[i]:.2f}–{edges[i + 1]:.2f}" for i in range(len(hist))],
             "Транзакций": hist}
        ).set_index("Вероятность")
        st.bar_chart(chart_data, color="#3B6E8F")


@st.fragment(run_every=2 if (auto_refresh or any_sending()) else None)
def results_fragment():
    """Фрагмент раздела результатов.

    Обновляется каждые 2 секунды, если включено автообновление или идёт
    фоновая отправка — так график растёт прямо во время загрузки.
    """
    draw_results()


if st.session_state.get("results_visible"):
    results_fragment()