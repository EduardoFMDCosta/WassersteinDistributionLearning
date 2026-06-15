import numpy as np


def _detect_modes(X: np.ndarray, n_max: int) -> int:
    """Select the number of Gaussian components by BIC.

    Requires scikit-learn.  Falls back to 1 mode if sklearn is unavailable
    or if fitting fails for any k.
    """
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
