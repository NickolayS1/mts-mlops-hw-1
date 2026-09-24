-- Витрина для результатов скоринга.
-- Создаётся автоматически при первой инициализации контейнера Postgres.

CREATE TABLE IF NOT EXISTS scores (
    id             BIGSERIAL PRIMARY KEY,
    transaction_id TEXT        NOT NULL,
    score          DOUBLE PRECISION NOT NULL,
    fraud_flag     INTEGER     NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Индекс для выборки последних транзакций и поиска фродов
CREATE INDEX IF NOT EXISTS scores_created_at_idx ON scores (created_at DESC);
CREATE INDEX IF NOT EXISTS scores_fraud_flag_idx ON scores (fraud_flag);