import pytest
from datetime import datetime, timedelta

from src.core.domain import DataFrequency
from src.infrastructure.adapters.timeseries import FrequencyDeterminer


# ─────────────────────────── Фикстуры ────────────────────────────
@pytest.fixture
def yearly_data():
    base = datetime(2010, 1, 1)
    return [base.replace(year=base.year + i) for i in range(6)]


@pytest.fixture
def quarterly_data():
    quarters = [1, 4, 7, 10]
    result = []
    for y in range(2):
        for q in quarters:
            result.append(datetime(2020 + y, q, 1))
    return result


@pytest.fixture
def monthly_data():
    return [datetime(2021, m, 1) for m in range(1, 13)]


@pytest.fixture
def daily_data():
    start = datetime(2024, 3, 1)
    return [start + timedelta(days=i) for i in range(30)]


@pytest.fixture
def hourly_data():
    start = datetime(2024, 3, 1)
    return [start + timedelta(hours=i) for i in range(48)]


@pytest.fixture
def minute_data():
    start = datetime(2024, 3, 1, 12, 0)
    return [start + timedelta(minutes=i) for i in range(120)]


# ─────────────────────────── Тесты ───────────────────────────────
@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("yearly_data", DataFrequency.year),
        ("quarterly_data", DataFrequency.quart),
        ("monthly_data", DataFrequency.month),
        ("daily_data", DataFrequency.day),
        ("hourly_data", DataFrequency.hour),
        ("minute_data", DataFrequency.minute),
    ],
)
def test_determine_various_frequencies(request, fixture_name, expected):
    timestamps = request.getfixturevalue(fixture_name)
    # Перемешиваем порядок, чтобы проверить независимость от сортировки
    shuffled = list(reversed(timestamps))
    assert FrequencyDeterminer.determine(shuffled) is expected


def test_single_observation_returns_month():
    ts = [datetime(2024, 1, 15)]
    assert FrequencyDeterminer.determine(ts) is DataFrequency.month


def test_empty_sequence_defaults_to_day():
    assert FrequencyDeterminer.determine([]) is DataFrequency.day