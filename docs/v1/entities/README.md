Тут лежат все схемы и модели для баз данных
# Сущности проекта

```mermaid
erDiagram
    users {
        integer id PK
        integer forecast_models_id
    }

    forecast_models {
        integer id PK
        text title
        text description
        boolean is_public
        integer user_id FK
        integer model_type FK
        integer dependent_variables_id FK
        integer explanatory_variables_id FK
        integer hyperparameters_id FK
        integer metrics_id FK
    }

    metrics {
        integer id PK
        varchar name
        float value
    }

    hyperparameters {
        integer id PK
        varchar type
        float value
    }

    variable {
        integer id PK
        text alias
        integer forecast_model_id FK
        integer timeseries_id
        integer preprocessing_sequence_id FK
    }

    preprocess {
        integer id PK
        text type
        integer position
        integer parameter_value_id FK
    }

    preprocess_parameters {
        integer id PK
        text name
        float value
    }

    model_type {
        integer id PK
        text name
        text class
    }

    users ||--o{ forecast_models : "Создает"
    forecast_models }o--|| model_type : "Тип модели"
    forecast_models }o--o{ metrics : "Метрики качества"
    forecast_models }o--o{ hyperparameters : "Гиперпараметры"
    forecast_models }o--o{ variable : "Зависимые переменные"
    forecast_models }o--o{ variable : "Объясняющие переменные"
    variable ||--o{ preprocess : "Предобработки"
    preprocess ||--|| preprocess_parameters : "Параметры предобработок"
```


# Схемы
## Ряды:
1. [Timeseries](./Timeseries.md) - временной ряд
2. [Preprocess](./Preprocess.md) - единица предобработки ряда
## Модели:
1. [ArimaxParams](./ArimaxParams.md) - параметры ARIMAX
2. [Forecasts](./Forecasts.md) - прогнозы модели
3. [Metric](./Metric.md) - метрика качесвтва модели
4. [Coefficient](./Coefficient.md) - коэффициент перед параметром в линейной модели (ARIMA/VAR)