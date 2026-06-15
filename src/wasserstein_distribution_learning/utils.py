import numpy as np


def _detect_modes(X: np.ndarray, n_max: int) -> int:
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return 1
    best_k, best_bic = 1, float('inf')
    for k in range(1, n_max + 1):
        try:
            bic = GaussianMixture(n_components=k, random_state=0).fit(X).bic(X)
            if bic < best_bic:
                best_bic, best_k = bic, k
        except Exception:
            break
    return best_k
