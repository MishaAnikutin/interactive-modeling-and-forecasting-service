# Параметры для перебора
total_points = 401
FORECAST_HORIZONS = [1, 6, 12, 24, 36]
TEST_SIZES = [0, 12, 24, 36]
VAL_SIZES = [0, 12, 24, 36]


total_for_extended = 30
FORECAST_HORIZONS_2 = [i for i in range(total_for_extended)]
TEST_SIZES_2 = [i for i in range(total_for_extended)]
VAL_SIZES_2 = [i for i in range(total_for_extended)]


def generate_valid_combinations(f, t, v, total):
    """Генерирует допустимые комбинации параметров"""
    combinations = []
    for h in f:
        for test_size in t:
            for val_size in v:
                train_size = total - val_size - test_size

                # Проверка ограничения 3
                if 4 * (h + test_size) > train_size:
                    continue

                # Проверка ограничения 1
                if 0 < val_size < h + test_size:
                    continue

                # Проверка минимального размера train
                if train_size < 10:  # Минимум 10 точек для обучения
                    continue

                if h + test_size == 0:
                    continue

                combinations.append((h, test_size, val_size))

    return combinations

VALID_COMBINATIONS = generate_valid_combinations(
    FORECAST_HORIZONS, TEST_SIZES, VAL_SIZES, total_points
)

VALID_COMBINATIONS_EXTENDED = generate_valid_combinations(
    FORECAST_HORIZONS_2, TEST_SIZES_2, VAL_SIZES_2, total_for_extended
)

aligned_size = 29
FORECAST_HORIZONS_exog = [i for i in range(aligned_size)]
TEST_SIZES_exog = [i for i in range(aligned_size)]
VAL_SIZES_exog = [i for i in range(aligned_size)]

VALID_COMBINATIONS_EXTENDED_exog = generate_valid_combinations(
    FORECAST_HORIZONS_exog, TEST_SIZES_exog, VAL_SIZES_exog, aligned_size
)