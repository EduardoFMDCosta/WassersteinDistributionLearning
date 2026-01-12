#!/usr/bin/env python3
# from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo


# -----------------------------
# Config
# -----------------------------
MIN_ROWS = 35_000
MIN_CONTINUOUS_FEATURES = 10

ID_RANGE = range(1, 3)          # adjust upward if you want broader scan  # 2001
SLEEP_SEC = 0.05                  # be polite, but keep it moving
SAVE_CSV_PATH = "ucimlrepo_scan_results.csv"

EXCLUDE_INTEGER_VALUED = True     # set False if you want integer-valued numeric columns included


# -----------------------------
# Heuristics
# -----------------------------
def _drop_na(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    return x[~pd.isna(x)]


def is_binary_indicator(col: pd.Series) -> bool:
    """True if non-missing values are a subset of {0,1} (int/bool/float)."""
    x = _drop_na(col.to_numpy())
    if x.size == 0:
        return False
    vals = np.unique(x)
    if vals.size > 2:
        return False
    # handle bool and numeric uniformly
    allowed = {0, 1, 0.0, 1.0, False, True}
    return set(vals.tolist()).issubset(allowed)


def is_integer_valued(col: pd.Series, tol: float = 1e-8) -> bool:
    """True if all finite values are (near) integers."""
    x = _drop_na(col.to_numpy())
    if x.size == 0:
        return False
    # if object dtype sneaks in, this may throw; caller should ensure numeric
    return np.all(np.abs(x - np.round(x)) < tol)


def count_continuous_features(X: pd.DataFrame) -> Tuple[int, List[str]]:
    """
    Continuous heuristic:
      - numeric dtype
      - not binary indicator
      - (optional) not integer-valued
    """
    if X is None or X.shape[1] == 0:
        return 0, []

    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        return 0, []

    keep: List[str] = []
    for c in X_num.columns:
        s = X_num[c]
        if is_binary_indicator(s):
            continue
        if EXCLUDE_INTEGER_VALUED and is_integer_valued(s):
            continue
        keep.append(c)

    return len(keep), keep


# -----------------------------
# Result container
# -----------------------------
@dataclass(frozen=True)
class Hit:
    uci_id: int
    name: str
    n_rows: int
    n_features_total: int
    n_continuous: int


# -----------------------------
# Main
# -----------------------------
def try_fetch(id_: int):
    try:
        return fetch_ucirepo(id=id_)
    except Exception:
        return None


def main() -> None:
    hits: List[Hit] = []

    for i, uci_id in enumerate(ID_RANGE, start=1):
        ds = try_fetch(uci_id)
        if ds is None:
            continue

        X = getattr(ds.data, "features", None)
        if not isinstance(X, pd.DataFrame):
            continue

        n_rows, n_feat = X.shape
        if n_rows <= MIN_ROWS:
            continue

        n_cont, _ = count_continuous_features(X)
        if n_cont <= MIN_CONTINUOUS_FEATURES:
            continue

        # name location can vary; be defensive
        name = getattr(ds, "metadata", {}).get("name", f"dataset_{uci_id}")

        hit = Hit(
            uci_id=uci_id,
            name=str(name),
            n_rows=int(n_rows),
            n_features_total=int(n_feat),
            n_continuous=int(n_cont),
        )
        hits.append(hit)
        print(f"[HIT] id={hit.uci_id} | {hit.name} | rows={hit.n_rows} | cont={hit.n_continuous}")

        if SLEEP_SEC:
            time.sleep(SLEEP_SEC)

        if i % 200 == 0:
            print(f"Progress: scanned {i} ids, hits={len(hits)}")

    if len(hits) > 0:
        out = pd.DataFrame([asdict(h) for h in hits]).sort_values(
            by=["n_rows", "n_continuous"], ascending=False
        )

        out.to_csv(SAVE_CSV_PATH, index=False)

        print(f"\nDone. Hits: {len(out)}")
        print(f"Saved: {SAVE_CSV_PATH}")
        if not out.empty:
            print(out.head(25).to_string(index=False))
    else:
        print("Done. No hits found.")

if __name__ == "__main__":
    main()

