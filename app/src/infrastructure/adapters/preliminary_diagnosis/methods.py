import numpy as np
import scipy.stats as stats
from src.core.domain.preliminary_diagnosis.service import DistributionServiceI
from src.infrastructure.adapters.preliminary_diagnosis.pp_plot_factory import (
    PPplotFactory,
)

@PPplotFactory.register(name="normal")
class Normal(DistributionServiceI):
    def get_theoretical_probs(self, ts: np.array) -> list[float]:
        params = stats.norm.fit(ts)
        sorted_data = np.sort(ts)
        theoretical_probs = stats.norm.cdf(sorted_data, *params)
        return theoretical_probs.tolist()


@PPplotFactory.register(name="exponential")
class Exponential(DistributionServiceI):
    def get_theoretical_probs(self, ts: np.array) -> list[float]:
        params = stats.expon.fit(ts)
        sorted_data = np.sort(ts)
        theoretical_probs = stats.expon.cdf(sorted_data, *params)
        return theoretical_probs.tolist()


@PPplotFactory.register(name="uniform")
class Uniform(DistributionServiceI):
    def get_theoretical_probs(self, ts: np.array) -> list[float]:
        params = stats.uniform.fit(ts)
        sorted_data = np.sort(ts)
        theoretical_probs = stats.uniform.cdf(sorted_data, *params)
        return theoretical_probs.tolist()


@PPplotFactory.register(name="chi2")
class Chi2(DistributionServiceI):
    def get_theoretical_probs(self, ts: np.array) -> list[float]:
        params = stats.chi2.fit(ts)
        sorted_data = np.sort(ts)
        theoretical_probs = stats.chi2.cdf(sorted_data, *params)
        return theoretical_probs.tolist()