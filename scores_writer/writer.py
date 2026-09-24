"""Сервис записи результатов скоринга в Postgres.

Читает топик scores и складывает сообщения (transaction_id, score, fraud_flag)
в витрину scores.
"""

import json
import logging
import os
import sys
import time

import psycopg2
from confluent_kafka import Consumer, KafkaError

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
SCORING_TOPIC = os.getenv("KAFKA_SCORING_TOPIC", "scores")
CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "scores-writer")

PG_HOST = os.getenv("POSTGRES_HOST", "postgres")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB = os.getenv("POSTGRES_DB", "fraud")
PG_USER = os.getenv("POSTGRES_USER", "fraud")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fraud")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv("LOG_PATH", "/app/logs/service.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

INSERT_SQL = (
    "INSERT INTO scores (transaction_id, score, fraud_flag) VALUES (%s, %s, %s)"
)


def connect_db(retries=30, delay=2):
    """Подключаемся к Postgres, дожидаясь его готовности."""
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(
                host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                user=PG_USER, password=PG_PASSWORD
            )
            conn.autocommit = True
            logger.info('Connected to Postgres at %s:%s', PG_HOST, PG_PORT)
            return conn
        except psycopg2.OperationalError as e:
            logger.warning('Postgres is not ready (%s/%s): %s', attempt, retries, e)
            time.sleep(delay)
    raise RuntimeError('Could not connect to Postgres')


class ScoresWriter:
    def __init__(self):
        self.conn = connect_db()
        self.consumer = Consumer({
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': CONSUMER_GROUP,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        })
        self.consumer.subscribe([SCORING_TOPIC])

    def save(self, record):
        """Сохраняем одну запись скоринга в витрину."""
        with self.conn.cursor() as cur:
            cur.execute(INSERT_SQL, (
                record['transaction_id'],
                float(record['score']),
                int(record['fraud_flag'])
            ))

    def run(self):
        logger.info('Writing scores from topic %s to Postgres...', SCORING_TOPIC)
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error('Kafka error: %s', msg.error())
                continue
            try:
                record = json.loads(msg.value().decode('utf-8'))
                self.save(record)
                # Смещение фиксируем только после успешной записи
                self.consumer.commit(asynchronous=False)
                logger.info('Saved %s', record['transaction_id'])
            except Exception as e:
                logger.error('Error processing message: %s', e)


if __name__ == "__main__":
    logger.info('Starting scores writer service...')
    writer = None
    try:
        writer = ScoresWriter()
        writer.run()
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
    except Exception as e:
        logger.error('Service failed: %s', e)
        sys.exit(1)
    finally:
        if writer is not None:
            writer.consumer.close()
            writer.conn.close()