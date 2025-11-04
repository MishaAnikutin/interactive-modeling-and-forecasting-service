from enum import StrEnum


class Distribution(StrEnum):
    norm: str = "norm"
    expon: str = "expon"
    pareto: str = "pareto"
    dweibull: str = "dweibull"
    t: str = "t"
    genextreme: str = "genextreme"
    gamma: str = "gamma"
    lognorm: str = "lognorm"
    beta: str = "beta"
    uniform: str = "uniform"
    loggamma: str = "loggamma"
