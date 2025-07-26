from pytest import fixture
from pathlib import Path
import json


@fixture(scope="function")
def forecasts_lstm_base(client):
    json_path = Path("/Users/oleg/projects/interactive-modeling-and-forecasting-service/"
                     "app/tests/data/month/forecasts_lstm_base.json")
    # Читаем файл и загружаем содержимое в словарь
    with json_path.open("r", encoding="utf-8") as fp:
        received_data: dict = json.load(fp)

    return received_data