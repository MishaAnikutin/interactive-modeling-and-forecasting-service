import pandas as pd

from src.core.domain.correlation.correlation import CorrelationMethod, CorrelationMatrix, Correlation


class CorrelationInteractor:
    def calculate(self, dataframe: pd.DataFrame, method: CorrelationMethod) -> CorrelationMatrix:
        values = dataframe.corr(method=method.value)
        columns = dataframe.columns.tolist()
        n = len(columns)

        correlation_matrix = [
            [
                Correlation(
                    value=values.iloc[i, j],
                    variable_1=columns[i],
                    variable_2=columns[j]
                )
                for j in range(n)
            ]
            for i in range(n)
        ]

        return CorrelationMatrix(values=correlation_matrix)