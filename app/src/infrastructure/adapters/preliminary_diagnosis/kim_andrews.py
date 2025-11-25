from src.core.application.preliminary_diagnosis.schemas.common import ResultValues
from src.core.application.preliminary_diagnosis.schemas.kim_andrews import KimAndrewsResult
from src.core.domain import Timeseries


from typing import Any
import numpy as np
from numpy import floating
from numpy.typing import NDArray
import pandas as pd
import statsmodels.formula.api as smf

from numpy.linalg import inv, pinv


class EOSResult:
    """
    Класс для хранения результатов применения теста на наличие структурных изменений
    в конце выборки.

    Attributes
    __________
    stats : dict[str, float]
    Словарь, содержащий значения статистик. Названия статистик хранятся в
    ключах словаря.

    pval : dict[str, float]
    Словарь, содержащий p-значения теста.

    substats: dict[str, list[float]]
    Словарь, содержащий списки значений промежуточных вспомогательных статистик.
    Названия статистик хранятся в ключах словаря.
    """

    def __init__(self):
        self.stats = {}
        self.pval = {}
        self.substats = {}

        for v in ["Sa", "Sb", "Sc", "Sd", "R"]:
            self.stats[v] = -1.0
            self.pval[v] = -1.0
            self.substats[v] = []


class ModelError(Exception):
    pass


def to_array(x: pd.Series | np.ndarray | list) -> NDArray[floating[Any]]:
    """
    Данная функция предназначена для преобразования объекта в вектор-столбец
    типа NDArray.

    Parameters
    ----------
    x : pd.Series | np.ndarray | list
        Объект со входными данными. Может быть столбцом датасета Pandas,
        массивом Numpy или простым списком.

    Returns
    -------
    NDArray
        Данные в виде вектор-столбца.

    Raises
    ------
    ValueError
        Данные имеют недопустимую форму или исходный тип.
    """
    match x:
        case pd.Series():
            return to_array(x.to_numpy())
        case list():
            return to_array(np.array(x))
        case np.ndarray():
            match x.shape:
                case (sh,):
                    return x.reshape((sh, 1))
                case (1, sh):
                    return x.T
                case (sh, 1):
                    return x
                case _:
                    raise ValueError("u of wrong shape")
        case _:
            raise ValueError("u of wrong type")


def sigma(u: pd.Series | np.ndarray | list, m: int) -> NDArray[floating[Any]]:
    """
    Данная функция предназначена для получения оценки ковариационной матрицы
    ошибок хвоста выборки.

    Parameters
    ----------
    u: pd.Series | np.ndarray | list
        Вектор остатков оценивания модели на всей выборке, включающей как
        начальную подвыборку, так и тестируемый "хвост" выборки.

    m : int
        Длина тестируемого хвоста.

    Returns
    -------
    NDArray[floating[Any]]
        Оцененное значение ковариационной матрицы размера m*m.
    """
    U = to_array(u)
    n = U.shape[0] - m
    return np.asarray(
        sum(U[j : j + m, :] @ U[j : j + m, :].T for j in range(n + 1)) / (n + 1)
    )


def proj(m: int, x: np.ndarray, sigma: np.ndarray) -> NDArray[floating[Any]]:
    """
    Функция для расчета проективной матрицы на пространство, определяемое
    матрицей x.

    Parameters
    ----------
    m : int
        Длина тестируемого хвоста.

    x : np.ndarray
        Матрица объясняющих переменных x. Должна иметь число строк, равное длине
        тестируемого хвоста.

    sigma : np.ndarray
        Ковариационная матрица размера m*m.

    Returns
    -------
    np.ndarray
        Проективная матрица.
    """
    assert isinstance(x, np.ndarray), "X should be NDarray"
    assert m == x.shape[0], "not enough observations"

    isigma = inv(sigma)

    if m < x.shape[1]:
        return isigma

    return isigma @ x @ inv(x.T @ isigma @ x) @ x.T @ isigma

def inverse(A):
    try:
        b = inv(A.T @ A)
    except np.linalg.LinAlgError:
        b = pinv(A.T @ A)
    return b


def andrews_eos_test(formula: str, data: pd.DataFrame, m: int = 1) -> EOSResult:
    """
    Основная функция, в которой реализуется тест на наличие структурных изменений
    в конце выборки[1]_[2]_.

    Parameters
    ----------
    formula : str
        Строка, описывающая модель в формате Patsy[3]_.

    data : pd.DataFrame
        Датафрейм Pandas, содержащий данные для модели.

    m : int
        Длина тестируемого хвоста.

    Returns
    -------
    EOSResult
        Объект типа EOSResult, включающий статистики Sa, Sb (=P), Sc, Sd, R.

    Notes
    -----
    Данная реализация теста имеет ограничение в том, что она в состоянии проводить тест
    только тех моделей, что могут быть представлены в пригодной для обычного МНК
    форме. Модели, требующие для оценивания более продвинутых методов, текущей реализацией
    не покрываются. Таким образом, де-факто, тестировать можно линейные межобъектные модели,
    а также модели авторегрессии. Однако для текущих целей этого достаточно.

    Отдельно следует отметить, что текущая реализация не фильтрует наблюдения с
    пропусками. Это будет реализовано в дальнейшем.

    References
    ----------
    .. [1] Andrews, Donald W. K. 2002.
       «End-of-Sample Instability Tests».
       Cowles Foundation Discussion Paper 1369. Cowles Foundation Discussion Papers.
       Cowles Foundation for Research in Economics, Yale University.
       https://ideas.repec.org//p/cwl/cwldpp/1369.html.
    .. [2] Andrews, Donald W. K, и Jae-Young Kim. 2006.
       «Tests for Cointegration Breakdown Over a Short Time Period».
       Journal of Business & Economic Statistics 24 (4): 379–94.
       https://doi.org/10.1198/073500106000000297.
    .. [3] https://patsy.readthedocs.io/en/latest/
    """
    assert formula is not None
    assert data is not None

    result = EOSResult()

    _model = smf.ols(formula, data)

    n = int(_model.nobs) - m
    Y, X = _model.endog.reshape((n + m, 1)), _model.exog

    assert Y is not None
    assert X is not None

    # Statistics
    _X0, _Y0 = X[:-m, :], Y[:-m, :]
    _X1, _Y1 = X[-m:, :], Y[-m:, :]

    _b0 = inverse(_X0) @ _X0.T @ _Y0
    _b = inverse(X) @ X.T @ Y

    SIGMA = sigma(Y - X @ _b, m)
    PROJ = proj(m, _X1, SIGMA)

    _resid0 = _Y1 - _X1 @ _b0
    _resid = _Y1 - _X1 @ _b

    result.stats["Sa"] = (_resid0.T @ _resid0).item()
    result.stats["Sb"] = (_resid.T @ _resid).item()
    result.stats["Sc"] = (_resid0.T @ PROJ @ _resid0).item()
    result.stats["Sd"] = (_resid.T @ PROJ @ _resid).item()
    result.stats["R"] = float(np.sum(np.cumsum(_resid[::-1, :]) ** 2))

    # SubSampling
    for t in range(n - m + 1):
        smpl1 = [i for i in range(n) if not (t <= i < t + m)]
        smpl2 = [i for i in range(n) if not (t <= i < t + np.ceil(m / 2))]
        smpl = list(range(t, t + m))

        _X1, _X2, _X = X[smpl1, :], X[smpl2, :], X[smpl, :]
        _Y1, _Y2, _Y = Y[smpl1, :], Y[smpl2, :], Y[smpl, :]

        _PROJ = proj(m, _X, SIGMA)

        _b1 = inverse(_X1) @ _X1.T @ _Y1
        _b2 = inverse(_X2) @ _X2.T @ _Y2

        _resid1 = _Y - _X @ _b1
        _resid2 = _Y - _X @ _b2

        result.substats["Sa"].append((_resid1.T @ _resid1).item())
        result.substats["Sb"].append((_resid2.T @ _resid2).item())
        result.substats["Sc"].append((_resid1.T @ _PROJ @ _resid1).item())
        result.substats["Sd"].append((_resid2.T @ _PROJ @ _resid2).item())
        result.substats["R"].append(float(np.sum(np.cumsum(_resid2[::-1, :]) ** 2)))

    for s in ["Sa", "Sb", "Sc", "Sd", "R"]:
        try:
            result.pval[s] = sum(
                result.stats[s] <= val for val in result.substats[s]
            ) / len(result.substats[s])
        except ZeroDivisionError:
            result.pval[s] = None

    return result


class KimAndrewsAdapter:
    @staticmethod
    def get_formula(*, dep_var="y", ar=1, const=False, trend=False):
        chunks = []
        if const:
            chunks.append("1")
        else:
            chunks.append("-1")
        if trend:
            chunks.append("t")
        if ar > 0:
            chunks.extend(f"{dep_var}.shift({i})" for i in range(1, ar + 1))

        formula = f"{dep_var} ~ " + " + ".join(chunks)
        return formula

    def run(self, ts: Timeseries, n: int, m: int, shift: int, trend: bool, const: bool) -> KimAndrewsResult:
        formula = self.get_formula(dep_var="y", ar=shift, trend=trend, const=const)

        df = pd.DataFrame({"y": ts.values})
        if trend:
            df["t"] = range(len(df))
        df = df.iloc[-(n + m + shift):]

        results = andrews_eos_test(formula, df, m)
        return KimAndrewsResult(
            Sa_values=ResultValues(
                p_value=results.pval["Sa"],
                stat_value=results.stats["Sa"],
            ),
            Sb_values=ResultValues(
                p_value=results.pval["Sb"],
                stat_value=results.stats["Sb"],
            ),
            Sc_values=ResultValues(
                p_value=results.pval["Sc"],
                stat_value=results.stats["Sc"],
            ),
            Sd_values=ResultValues(
                p_value=results.pval["Sd"],
                stat_value=results.stats["Sd"],
            ),
            R_values=ResultValues(
                p_value=results.pval["R"],
                stat_value=results.stats["R"],
            ),
        )