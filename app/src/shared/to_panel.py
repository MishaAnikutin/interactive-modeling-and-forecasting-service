import pandas as pd
from fastapi import HTTPException


def to_panel(
        target: pd.Series,
        exog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if isinstance(target, pd.DataFrame):
        target = target.iloc[:, 0]
    df = pd.DataFrame(
        {
            "unique_id": 'ts',
            "ds": target.index,
            "y": target.values,
        }
    )
    if exog is not None and not exog.empty:
        # Проверка конфликта имен
        conflict_columns = set(exog.columns) & {'unique_id', 'ds', 'y'}
        if conflict_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Конфликт имен в экзогенных переменных: {conflict_columns}"
            )

        # Объединяем с экзогенными переменными
        df = df.set_index('ds')
        df = df.join(exog, how='left')
        df = df.reset_index()
    return df