from typing import Optional

import pandas as pd
from datetime import datetime
from src.core.domain.validation import ValidationStrategyI, ValidationIssue, ValidationType
from .visitor import ValidationVisitor


# README: этот код я скопировал из тетрадки и пока не рефакторил. Хотя стоит

def Fcount(diffs: pd.Series) -> dict[str, int]:
    """Функция подсчета для определения частотности"""
    res = {"D": 0, "W": 0, "ME": 0, "QE": 0, "YE": 0, "#": 0}

    for i in diffs.index:
        match i:
            case pd.Timedelta(days=1):
                res["D"] += diffs[i]
            case pd.Timedelta(days=7):
                res["W"] += diffs[i]
            case (
            pd.Timedelta(days=28)
            | pd.Timedelta(days=29)
            | pd.Timedelta(days=30)
            | pd.Timedelta(days=31)
            ):
                res["ME"] += diffs[i]
            case pd.Timedelta(days=90) | pd.Timedelta(days=91) | pd.Timedelta(days=92):
                res["QE"] += diffs[i]
            case pd.Timedelta(days=365) | pd.Timedelta(days=366):
                res["YE"] += diffs[i]
            case _:
                res["#"] += diffs[i]

    return res


def detect_frequency(ts: pd.Series) -> str:
    """Функция определения частотности временного ряда"""
    if len(ts) == 1:
        return "SINGLE_OBSERVATION"

    # Преобразуем индекс в datetime и сортируем
    dates = pd.to_datetime(ts.index, dayfirst=True, format="%d.%m.%Y")
    sorted_dates = sorted(dates)

    # Вычисляем разницы между датами
    diffs_series = pd.Series(sorted_dates).diff().value_counts()
    diffs = Fcount(diffs_series)
    max_diff = max(diffs.values())

    # Константы для определения смешанной частотности
    const_d = 1
    const_w = 1
    const_m = 1
    const_q = 1
    const_y = 1

    if diffs["D"] == max_diff:
        if diffs["W"] > const_w or diffs["ME"] > const_m or diffs["QE"] > const_q or diffs["YE"] > const_y:
            return "MIXED_FREQUENCY"
        return "D"
    elif diffs["W"] == max_diff:
        if diffs["D"] > const_d or diffs["ME"] > const_m or diffs["QE"] > const_q or diffs["YE"] > const_y:
            return "MIXED_FREQUENCY"
        return "W"
    elif diffs["ME"] == max_diff:
        if diffs["D"] > const_d or diffs["W"] > const_w or diffs["QE"] > const_q or diffs["YE"] > const_y:
            return "MIXED_FREQUENCY"
        return "ME"
    elif diffs["QE"] == max_diff:
        if diffs["D"] > const_d or diffs["W"] > const_w or diffs["ME"] > const_m or diffs["YE"] > const_y:
            return "MIXED_FREQUENCY"
        return "QE"
    elif diffs["YE"] == max_diff:
        if diffs["D"] > const_d or diffs["W"] > const_w or diffs["ME"] > const_m or diffs["QE"] > const_q:
            return "MIXED_FREQUENCY"
        return "YE"

    return "UNDETERMINED"


@ValidationVisitor.register(ValidationType.EXCESSIVE_ZEROS)
class ExcessiveZerosValidator(ValidationStrategyI):
    def check(self, ts: pd.Series) -> Optional[ValidationIssue]:
        """Проверка на избыточное количество нулей в данных"""
        try:
            # Определяем частотность
            freq = detect_frequency(ts)

            # Подсчитываем нули
            zeros_count = (ts == 0).sum()
            total_count = len(ts)
            zeros_ratio = zeros_count / total_count

            # Вычисляем максимальную последовательность нулей
            zero_sequences = (ts == 0).astype(int)
            if zero_sequences.sum() > 0:
                consecutive_zeros = zero_sequences.groupby((zero_sequences != 0).cumsum()).sum().max()
            else:
                consecutive_zeros = 0

            if freq == "D" and (zeros_ratio > 0.4 or consecutive_zeros > 14):
                is_severity = True
                message = "В ряде присутствует много нулей. Дневные данные."
            elif freq == "W" and (zeros_ratio > 0.1 or consecutive_zeros > 4):
                is_severity = True
                message = "В ряде присутствует много нулей. Недельные данные."
            elif freq in ["ME", "QE"] and (zeros_ratio > 0.1 or consecutive_zeros > 2):
                is_severity = True
                message = "В ряде присутствует много нулей. Месячные/квартальные данные."
            elif freq == "YE" and (zeros_ratio > 0.3 or consecutive_zeros > 2):
                is_severity = True
                message = "В ряде присутствует много нулей. Годовые данные."
            else:
                # Если нет проблем с нулями, возвращаем None
                return None

            return ValidationIssue(
                type=ValidationType.EXCESSIVE_ZEROS,
                is_severity=is_severity,
                message=message
            )

        except Exception as e:
            return ValidationIssue(
                type=ValidationType.EXCESSIVE_ZEROS,
                is_severity=True,
                message=f"Ошибка при проверке нулей: {str(e)}"
            )


@ValidationVisitor.register(ValidationType.SINGLE_OBSERVATION_FREQUENCY)
class SingleObservationFrequencyValidator(ValidationStrategyI):
    def check(self, ts: pd.Series) -> Optional[ValidationIssue]:
        """Проверка на ряд с единственным наблюдением"""
        if len(ts) == 1:
            return ValidationIssue(
                type=ValidationType.SINGLE_OBSERVATION_FREQUENCY,
                is_severity=True,
                message="Невозможно определить частотность. Ряд состоит из одного наблюдения"
            )
        # Если ряд содержит больше одного наблюдения, возвращаем None
        return None


@ValidationVisitor.register(ValidationType.UNDETERMINED_FREQUENCY)
class UndeterminedFrequencyValidator(ValidationStrategyI):
    def check(self, ts: pd.Series) -> Optional[ValidationIssue]:
        """Проверка на неопределенную частотность"""
        if len(ts) <= 1:
            return ValidationIssue(
                type=ValidationType.UNDETERMINED_FREQUENCY,
                is_severity=True,
                message="Недостаточно данных для определения частотности"
            )

        freq = detect_frequency(ts)
        is_severity = freq == "UNDETERMINED"

        if not is_severity:
            # Если частотность определена нормально, возвращаем None
            return None

        return ValidationIssue(
            type=ValidationType.UNDETERMINED_FREQUENCY,
            is_severity=is_severity,
            message="Невозможно определить частотность временного ряда"
        )


@ValidationVisitor.register(ValidationType.SPARSE_DATA)
class SparseDataValidator(ValidationStrategyI):
    def check(self, ts: pd.Series) -> Optional[ValidationIssue]:
        """Проверка на разреженность данных (пропуски)"""
        try:
            if len(ts) <= 1:
                return ValidationIssue(
                    type=ValidationType.SPARSE_DATA,
                    is_severity=True,
                    message="Недостаточно данных для анализа пропусков"
                )

            # Определяем частотность
            freq = detect_frequency(ts)
            if freq in ["SINGLE_OBSERVATION", "UNDETERMINED", "MIXED_FREQUENCY"]:
                return ValidationIssue(
                    type=ValidationType.SPARSE_DATA,
                    is_severity=True,
                    message=f"Невозможно анализировать пропуски из-за проблем с частотностью: {freq}"
                )

            # Создаем DataFrame для анализа пропусков
            dates = pd.to_datetime(ts.index, dayfirst=True, format="%d.%m.%Y")
            df = pd.DataFrame({'date': dates, 'obs': ts.values})
            df = df.sort_values('date')

            # Анализируем разницы между наблюдениями
            diffs_series = df["date"].diff().value_counts()
            diffs = Fcount(diffs_series)

            # Если есть нестандартные интервалы, анализируем пропуски
            if diffs["#"] > 0:
                # Создаем полный временной ряд для анализа пропусков
                # Преобразуем частотность в pandas frequency
                freq_mapping = {
                    "D": "D",
                    "W": "W",
                    "ME": "ME",  # Для месячных данных используем 'M'
                    "QE": "QE",  # Для квартальных данных используем 'Q'
                    "YE": "YE"  # Для годовых данных используем 'Y'
                }

                pandas_freq = freq_mapping.get(freq, "D")
                try:
                    df_full = df.set_index('date').asfreq(pandas_freq)
                    total_gap_ratio = df_full['obs'].isnull().sum() / len(df_full)

                    # Вычисляем максимальный пропуск
                    max_gap = df_full['obs'].isnull().astype(int).groupby(
                        df_full['obs'].notnull().cumsum()
                    ).sum().max()

                    # Проверяем условия для разных частотностей
                    is_severity = False
                    message = ""

                    if freq == "D" and (total_gap_ratio > 0.4 or max_gap > 14):
                        is_severity = True
                        message = "Ряд слишком разряжен или слишком длинный пропуск. Дневные данные."
                    elif freq == "W" and (total_gap_ratio > 0.1 or max_gap > 4):
                        is_severity = True
                        message = "Ряд слишком разряжен или слишком длинный пропуск. Недельные данные."
                    elif freq in ["ME", "QE"] and (total_gap_ratio > 0.1 or max_gap > 2):
                        is_severity = True
                        message = "Ряд слишком разряжен или слишком длинный пропуск. Месячные/квартальные данные."
                    elif freq == "YE" and (total_gap_ratio > 0.3 or max_gap > 2):
                        is_severity = True
                        message = "Ряд слишком разряжен или слишком длинный пропуск. Годовые данные."
                    else:
                        # Если пропуски незначительные, возвращаем None
                        return None

                    return ValidationIssue(
                        type=ValidationType.SPARSE_DATA,
                        is_severity=is_severity,
                        message=message
                    )
                except Exception as freq_error:
                    return ValidationIssue(
                        type=ValidationType.SPARSE_DATA,
                        is_severity=True,
                        message=f"Ошибка при анализе пропусков с частотностью {freq}: {str(freq_error)}"
                    )
            else:
                # Если нет нестандартных интервалов, возвращаем None
                return None

        except Exception as e:
            return ValidationIssue(
                type=ValidationType.SPARSE_DATA,
                is_severity=True,
                message=f"Ошибка при анализе пропусков: {str(e)}"
            )


@ValidationVisitor.register(ValidationType.FUTURE_VALUES)
class FutureValuesValidator(ValidationStrategyI):
    def check(self, ts: pd.Series) -> Optional[ValidationIssue]:
        """Проверка на наличие будущих значений относительно текущей даты"""
        try:
            # Преобразуем индекс в datetime
            dates = pd.to_datetime(ts.index, dayfirst=True, format="%d.%m.%Y")
            today = pd.to_datetime(datetime.today().date())

            # Находим будущие даты
            future_dates = dates[dates > today]
            is_severity = len(future_dates) > 0

            if not is_severity:
                # Если будущих значений нет, возвращаем None
                return None

            return ValidationIssue(
                type=ValidationType.FUTURE_VALUES,
                is_severity=is_severity,
                message=f"В ряде содержатся значения из будущего, относительно даты {today.strftime('%d.%m.%Y')}"
            )

        except Exception as e:
            return ValidationIssue(
                type=ValidationType.FUTURE_VALUES,
                is_severity=True,
                message=f"Ошибка при проверке будущих значений: {str(e)}"
            )


@ValidationVisitor.register(ValidationType.MIXED_FREQUENCY)
class MixedFrequencyValidator(ValidationStrategyI):
    def check(self, ts: pd.Series) -> Optional[ValidationIssue]:
        """Проверка на смешанную частотность"""
        if len(ts) <= 1:
            return ValidationIssue(
                type=ValidationType.MIXED_FREQUENCY,
                is_severity=True,
                message="Недостаточно данных для анализа частотности"
            )

        freq = detect_frequency(ts)
        is_severity = freq == "MIXED_FREQUENCY"

        if not is_severity:
            # Если смешанной частотности нет, возвращаем None
            return None

        # Определяем основную частотность для сообщения
        dates = pd.to_datetime(ts.index, dayfirst=True, format="%d.%m.%Y")
        sorted_dates = sorted(dates)
        diffs_series = pd.Series(sorted_dates).diff().value_counts()
        diffs = Fcount(diffs_series)
        max_freq = max(diffs, key=diffs.get)

        frequency_names = {
            "D": "дневная",
            "W": "недельная",
            "ME": "месячная",
            "QE": "квартальная",
            "YE": "годовая"
        }
        message = f"В ряде присутствуют 2 частотности. Основополагающая частотность {frequency_names.get(max_freq, 'неизвестная')}"

        return ValidationIssue(
            type=ValidationType.MIXED_FREQUENCY,
            is_severity=is_severity,
            message=message
        )
