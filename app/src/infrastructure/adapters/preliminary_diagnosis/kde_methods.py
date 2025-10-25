import numpy as np
from scipy import stats

from src.core.domain.preliminary_diagnosis.kde_service import KdeServiceI
from src.infrastructure.adapters.preliminary_diagnosis.kde_factory import KdeFactory


@KdeFactory.register(name="silverman")
class Silverman(KdeServiceI):
    def calculate_kde(self):
        x_grid = self.get_x_grid()
        kde = stats.gaussian_kde(self.ts_)
        kde.set_bandwidth(bw_method='silverman')
        density = kde(x_grid)
        bandwidth = kde.factor * np.std(self.ts_)

        return x_grid, density, bandwidth


@KdeFactory.register(name="scott")
class Scott(KdeServiceI):
    def calculate_kde(self):
        x_grid = self.get_x_grid()
        kde = stats.gaussian_kde(self.ts_)
        kde.set_bandwidth(bw_method='scott')
        density = kde(x_grid)
        bandwidth = kde.factor * np.std(self.ts_)

        return x_grid, density, bandwidth