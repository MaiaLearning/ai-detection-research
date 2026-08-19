"""Exact-stratum + nearest-neighbor caliper matching.

Used to pair essays from a "treatment" group (e.g. ell_status == "Yes")
against a "control" group on a mix of exact-match covariates (e.g. prompt,
grade) and continuous covariates compared by Euclidean distance (e.g.
standardized word count and holistic score), so that downstream feature
comparisons aren't confounded by those covariates.
"""
import numpy as np
import pandas as pd


def nearest_neighbor_match(
    df: pd.DataFrame,
    id_col: str,
    treatment_col: str,
    treatment_value,
    control_value,
    exact_cols: list[str],
    distance_cols: list[str],
    caliper: float,
) -> pd.DataFrame:
    matches = []

    if exact_cols:
        groups = [group for _, group in df.groupby(exact_cols, dropna=False)]
    else:
        groups = [df]

    for group in groups:
        treated = group[group[treatment_col] == treatment_value].sort_values(id_col)
        controls = group[group[treatment_col] == control_value]
        if treated.empty or controls.empty:
            continue

        available = controls.set_index(id_col)
        for _, trow in treated.iterrows():
            if available.empty:
                continue
            t_vec = trow[distance_cols].to_numpy(dtype=float)
            c_vecs = available[distance_cols].to_numpy(dtype=float)
            dists = np.sqrt(((c_vecs - t_vec) ** 2).sum(axis=1))
            best_pos = int(np.argmin(dists))
            best_dist = float(dists[best_pos])
            if best_dist > caliper:
                continue
            best_id = available.index[best_pos]
            matches.append({
                f"{id_col}_treated": trow[id_col],
                f"{id_col}_control": best_id,
                "distance": best_dist,
            })
            available = available.drop(index=best_id)

    return pd.DataFrame(matches, columns=[f"{id_col}_treated", f"{id_col}_control", "distance"])
