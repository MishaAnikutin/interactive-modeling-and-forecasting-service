from pytest import fixture
from pathlib import Path
import json


@fixture(scope="function")
def forecasts_lstm_base(client):
    json_path = Path("/Users/oleg/projects/interactive-modeling-and-forecasting-service/"
                     "app/tests/data/month/forecasts_lstm_base.json")
    with json_path.open("r", encoding="utf-8") as fp:
        received_data: dict = json.load(fp)

    return received_data

@fixture(scope="function")
def forecasts_lstm_exog(client):
    json_path = Path("/Users/oleg/projects/interactive-modeling-and-forecasting-service/"
                     "app/tests/data/month/forecasts_lstm_exog.json")
    with json_path.open("r", encoding="utf-8") as fp:
        received_data: dict = json.load(fp)

    return received_data