import pytest
from datetime import datetime, timedelta

from fastapi import HTTPException

from src.core.domain import DataFrequency
from tests.conftest import freq_determiner

def test_empty_timestamps(freq_determiner):
    with pytest.raises(HTTPException) as exc:
        freq_determiner.determine([])
    assert exc.value.status_code == 400
    assert "Ряд должен быть не пустой" in exc.value.detail

def test_single_timestamp(freq_determiner):
    date = datetime(2023, 1, 1)
    result = freq_determiner.determine([date])
    assert result == DataFrequency.day

def test_daily_frequency(freq_determiner):
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    result = freq_determiner.determine(dates)
    assert result == DataFrequency.day

def test_monthly_frequency(freq_determiner):
    # Последние дни месяцев
    dates = [
        datetime(2023, 1, 31),
        datetime(2023, 2, 28),
        datetime(2023, 3, 31),
        datetime(2023, 4, 30),
        datetime(2023, 5, 31),
        datetime(2023, 6, 30),
        datetime(2023, 7, 31),
        datetime(2023, 8, 31),
        datetime(2023, 9, 30),
        datetime(2023, 10, 31),
    ]
    result = freq_determiner.determine(dates)
    assert result == DataFrequency.month

def test_quarterly_frequency(freq_determiner):
    # Последние дни кварталов
    dates = [
        datetime(2023, 3, 31),
        datetime(2023, 6, 30),
        datetime(2023, 9, 30),
        datetime(2023, 12, 31),
        datetime(2024, 3, 31),
        datetime(2024, 6, 30),
        datetime(2024, 9, 30),
        datetime(2024, 12, 31),
        datetime(2025, 3, 31),
        datetime(2025, 6, 30),
    ]
    result = freq_determiner.determine(dates)
    assert result == DataFrequency.quart

def test_yearly_frequency(freq_determiner):
    # Последние дни годов
    dates = [
        datetime(2020, 12, 31),
        datetime(2021, 12, 31),
        datetime(2022, 12, 31),
        datetime(2023, 12, 31),
        datetime(2024, 12, 31),
        datetime(2025, 12, 31),
        datetime(2026, 12, 31),
        datetime(2027, 12, 31),
        datetime(2028, 12, 31),
        datetime(2029, 12, 31),
    ]
    result = freq_determiner.determine(dates)
    assert result == DataFrequency.year

def test_unsupported_frequency(freq_determiner):
    dates = [datetime(2023, 1, 1) + timedelta(days=i*5) for i in range(10)]
    with pytest.raises(HTTPException) as exc:
        freq_determiner.determine(dates)
    assert exc.value.status_code == 400
    assert "неподдерживаемую частотность" in exc.value.detail

def test_timestamps_with_time(freq_determiner):
    dates = [datetime(2023, 1, 1, 12, 30) + timedelta(days=i) for i in range(10)]
    with pytest.raises(HTTPException) as exc:
        freq_determiner.determine(dates)
    assert exc.value.status_code == 400
    assert "Даты должны быть без времени" in exc.value.detail

def test_inconsistent_frequency(freq_determiner):
    dates = [
        datetime(2023, 1, 31),
        datetime(2023, 2, 28),
        datetime(2023, 4, 30),  # Пропущен март
        datetime(2023, 5, 31),
    ]
    with pytest.raises(HTTPException) as exc:
        freq_determiner.determine(dates)
    assert exc.value.status_code == 400
    assert "не постоянной частотности" in exc.value.detail

def test_monthly_with_non_last_day(freq_determiner):
    dates = [
        datetime(2023, 1, 1),
        datetime(2023, 2, 1),
        datetime(2023, 3, 1),
        datetime(2023, 4, 1),
    ]
    with pytest.raises(HTTPException) as exc:
        freq_determiner.determine(dates)
    assert exc.value.status_code == 400
    assert "не является последним днем месяца" in exc.value.detail

def test_quarterly_with_non_last_day(freq_determiner):
    dates = [
        datetime(2023, 3, 1),
        datetime(2023, 6, 1),
        datetime(2023, 9, 1),
        datetime(2023, 12, 1),
    ]
    with pytest.raises(HTTPException) as exc:
        freq_determiner.determine(dates)
    assert "не является последним днем месяца" in exc.value.detail

def test_yearly_with_non_last_day(freq_determiner):
    dates = [
        datetime(2023, 12, 1),
        datetime(2024, 12, 1),
        datetime(2025, 12, 1),
    ]
    with pytest.raises(HTTPException) as exc:
        freq_determiner.determine(dates)
    assert "не является последним днем месяца" in exc.value.detail