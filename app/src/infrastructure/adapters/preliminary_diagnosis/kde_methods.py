from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import NearestNeighbors, KernelDensity

from src.core.application.preliminary_diagnosis.schemas.kde import SilvermanMethod, ScottMethod, KnnMethod, CrossValidationMethod
from src.core.domain.preliminary_diagnosis.kde_service import KdeServiceI
from src.infrastructure.adapters.preliminary_diagnosis.kde_factory import KdeFactory
import numpy as np


@KdeFactory.register(name="silverman")
class Silverman(KdeServiceI):
    def calculate_bandwidth(self, method: SilvermanMethod) -> float:
        sigma = np.std(self.ts_, ddof=1)
        iqr = np.subtract(*np.percentile(self.ts_, [75, 25]))
        sigma_r = min(sigma, iqr / 1.34) if iqr > 0 else sigma
        return 0.9 * sigma_r * self.ts_ ** (-1 / 5)


@KdeFactory.register(name="scott")
class Scott(KdeServiceI):
    def calculate_bandwidth(self, method: ScottMethod) -> float:
        sigma = np.std(self.ts_, ddof=1)
        return sigma * self.ts_.size ** (-1 / 5)


@KdeFactory.register(name="knn")
class Knn(KdeServiceI):
    def calculate_bandwidth(self, method: KnnMethod) -> float:
        x = self.ts_.reshape(-1, 1)
        k = method.k
        n = self.ts_.size
        if k is None:
            k = max(2, int(np.sqrt(n)))
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(x)
        dists, _ = nbrs.kneighbors(x)
        kth = dists[:, min(k, dists.shape[1] - 1)]
        return float(np.median(kth))


@KdeFactory.register(name="cross validation")
class CrossValidation(KdeServiceI):
    def calculate_bandwidth(self, method: CrossValidationMethod) -> float:
        scott_bw   = KdeFactory.calculate_bandwidth(self.ts_, ScottMethod())
        silver_bw  = KdeFactory.calculate_bandwidth(self.ts_, SilvermanMethod())
        base = np.nanmean([scott_bw, silver_bw])

        if not np.isfinite(base) or base <= 0:
            base = np.std(self.ts_, ddof=1) * (self.ts_.size ** (-1 / 5))

        grid = np.unique(np.clip(base * np.logspace(-1.0, 1.0, 15), 1e-3, None))

        cv = KFold(
            n_splits=min(method.folds, len(self.ts_)),
            shuffle=True,
            random_state=123
        )
        search = GridSearchCV(KernelDensity(), {"bandwidth": grid}, cv=cv, n_jobs=1)
        search.fit(self.ts_.reshape(-1, 1))

        return float(search.best_params_["bandwidth"])