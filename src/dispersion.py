"""Across-document dispersion and centroid statistics for Experiment 7:
distinguishing within-document polish (H1) from across-document
homogenization (H2) as drivers of detectability.
"""
import numpy as np


def standardize_by_reference(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """z-score X using the REFERENCE array's per-column mean/std, not X's
    own -- so dispersion is expressed in a fixed (e.g. human) unit."""
    ref_mean = reference.mean(axis=0)
    ref_std = reference.std(axis=0, ddof=1)
    return (X - ref_mean) / ref_std


def centroid(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0)


def mean_distance_to_centroid(X: np.ndarray) -> float:
    c = centroid(X)
    return float(np.mean(np.linalg.norm(X - c, axis=1)))


def covariance_trace(X: np.ndarray) -> float:
    return float(np.trace(np.cov(X, rowvar=False, ddof=1)))


def covariance_determinant(X: np.ndarray) -> float:
    cov = np.cov(X, rowvar=False, ddof=1)
    return float(np.linalg.det(np.atleast_2d(cov)))


def project_onto_vector(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    """Signed length of (point - origin) projected onto direction, which is
    normalized internally -- so callers can pass any non-unit vector."""
    unit = direction / np.linalg.norm(direction)
    return float(np.dot(point - origin, unit))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(angle) between two vectors. Both must already be expressed in the
    same coordinate space -- cosine similarity is not invariant to per-axis
    rescaling, so standardizing a and b with different scalers before
    calling this would silently produce a meaningless angle."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
