import scipy.stats as stats
from src.core.domain.distributions import Distribution, DistributionServiceI

from .factory import DistributionFactory


@DistributionFactory.register(name=Distribution.norm)
class Normal(DistributionServiceI):
    impl = stats.norm


@DistributionFactory.register(name=Distribution.expon)
class Exponential(DistributionServiceI):
    impl = stats.expon


@DistributionFactory.register(name=Distribution.pareto)
class Pareto(DistributionServiceI):
    impl = stats.pareto


@DistributionFactory.register(name=Distribution.dweibull)
class Dweibull(DistributionServiceI):
    impl = stats.dweibull


@DistributionFactory.register(name=Distribution.t)
class StudentT(DistributionServiceI):
    impl = stats.t


@DistributionFactory.register(name=Distribution.genextreme)
class GenExtreme(DistributionServiceI):
    impl = stats.genextreme


@DistributionFactory.register(name=Distribution.gamma)
class Gamma(DistributionServiceI):
    impl = stats.gamma


@DistributionFactory.register(name=Distribution.lognorm)
class LogNorm(DistributionServiceI):
    impl = stats.lognorm


@DistributionFactory.register(name=Distribution.beta)
class Beta(DistributionServiceI):
    impl = stats.beta


@DistributionFactory.register(name=Distribution.uniform)
class Uniform(DistributionServiceI):
    impl = stats.uniform


@DistributionFactory.register(name=Distribution.loggamma)
class LogGamma(DistributionServiceI):
    impl = stats.loggamma
