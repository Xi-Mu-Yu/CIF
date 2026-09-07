"""
MNE-based spatial preprocessing: ICA, SSP, and SSS-like inner subspace projection.

Trials are collected from train_loader as (n_epochs, n_channels, n_times).
Each method returns W (n_channels, n_channels) applied in TCN as x @ W per time step
(same convention as utils.csp).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import mne
    from mne.io import RawArray
    from mne.preprocessing import ICA
except ImportError as exc:
    raise ImportError(
        "ICA/SSP/SSS spatial modes require MNE-Python. Install with: pip install mne"
    ) from exc

from utils.csp import _collect_epochs_from_loader, _pad_epochs


def _make_info(n_channels: int, sfreq: float):
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    return mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")


def _epochs_to_raw(epochs, n_channels: int, sfreq: float):
    """Stack epochs along time into one MNE RawArray (n_channels, total_time)."""
    if not epochs:
        return None
    data = np.concatenate(epochs, axis=1)
    if data.shape[0] != n_channels:
        raise ValueError(f"Expected {n_channels} channels, got {data.shape[0]}")
    return RawArray(np.ascontiguousarray(data, dtype=np.float64), _make_info(n_channels, sfreq))


def _projector_matrix_from_projs(projs, n_channels: int) -> np.ndarray:
    """Build apply matrix from MNE projection list (I - sum v v^T / v^T v)."""
    P = np.eye(n_channels, dtype=np.float64)
    for proj in projs:
        vec = np.asarray(proj["data"]["data"], dtype=np.float64).reshape(-1)
        denom = float(vec @ vec)
        if denom <= 0:
            continue
        P -= np.outer(vec, vec) / denom
    return P


def _row_apply_matrix(square: np.ndarray) -> np.ndarray:
    """Left-multiply matrix M -> W for row vectors: y_row = x_row @ W."""
    return np.asarray(square, dtype=np.float64).T


def _ica_reconstruction_matrix(ica, n_channels: int, sfreq: float) -> np.ndarray:
    """
    Build full (n_channels, n_channels) left-apply matrix M: x_clean = M @ x.

    Uses unit spatial patterns so it stays consistent with ica.apply() across MNE versions.
    """
    info = _make_info(n_channels, sfreq)
    n_times = 32
    M = np.zeros((n_channels, n_channels), dtype=np.float64)
    exclude = list(getattr(ica, "exclude", []) or [])

    for ch in range(n_channels):
        pattern = np.zeros((n_channels, n_times), dtype=np.float64)
        pattern[ch, :] = 1.0
        raw = RawArray(pattern, info)
        cleaned = ica.apply(raw, exclude=exclude, verbose=False)
        M[:, ch] = cleaned.get_data()[:, 0]

    return M


def fit_ica_matrix_from_batches(
    train_loader,
    n_channels: int,
    sfreq: float = 256.0,
    n_components: Optional[int] = None,
    n_exclude: int = 1,
    random_state: int = 97,
):
    """
    Fit MNE ICA on training trials and return reconstruction matrix with excluded components.

    Components with highest kurtosis are treated as artifacts when n_exclude > 0.
    """
    eye = np.eye(n_channels, dtype=np.float64)
    epochs, _ = _collect_epochs_from_loader(train_loader, n_channels)
    if not epochs:
        return eye

    raw = _epochs_to_raw(epochs, n_channels, sfreq)
    if raw is None:
        return eye

    n_comp = n_components or n_channels
    n_comp = int(max(2, min(n_comp, n_channels)))

    ica = ICA(
        n_components=n_comp,
        max_iter="auto",
        random_state=random_state,
        method="fastica",
    )
    try:
        ica.fit(raw)
    except Exception:
        return eye

    if n_exclude > 0 and ica.unmixing_matrix_ is not None:
        try:
            sources = ica.get_sources(raw).get_data()
            if sources.shape[0] > 0:
                from scipy.stats import kurtosis

                k = kurtosis(sources, axis=1, fisher=True, nan_policy="omit")
                k = np.nan_to_num(k, nan=-np.inf)
                n_drop = min(n_exclude, len(k))
                ica.exclude = np.argsort(k)[-n_drop:].tolist()
        except Exception:
            ica.exclude = []

    try:
        recon = _ica_reconstruction_matrix(ica, n_channels, sfreq)
    except Exception:
        return eye

    if recon.shape != (n_channels, n_channels):
        return eye

    return _row_apply_matrix(recon)


def fit_ssp_matrix_from_batches(
    train_loader,
    n_channels: int,
    sfreq: float = 256.0,
    n_proj: int = 1,
):
    """
    Fit MNE signal-space projectors (PCA on training data) and return apply matrix.

    Projects out the n_proj directions of largest spatial variance (common-mode / noise).
    """
    eye = np.eye(n_channels, dtype=np.float64)
    epochs, _ = _collect_epochs_from_loader(train_loader, n_channels)
    if not epochs:
        return eye

    raw = _epochs_to_raw(epochs, n_channels, sfreq)
    if raw is None:
        return eye

    n_proj = int(max(0, min(n_proj, n_channels - 1)))
    if n_proj == 0:
        return eye

    try:
        projs = mne.compute_proj_raw(
            raw,
            n_eeg=n_proj,
            n_grad=0,
            n_mag=0,
            verbose=False,
        )
        proj_mat = _projector_matrix_from_projs(projs, n_channels)
    except Exception:
        return eye

    return _row_apply_matrix(proj_mat)


def fit_sss_matrix_from_batches(
    train_loader,
    n_channels: int,
    inner_rank: Optional[int] = None,
):
    """
    SSS-inspired inner subspace reconstruction for channel-only data (no MEG geometry).

    Estimates spatial covariance on training trials, keeps the top inner_rank eigenvectors
    as the "inner" (signal) subspace, and returns projection W s.t. x_clean = x @ W.

    True Maxwell SSS requires MEG sensor layouts; this is the EEG-compatible analogue used
    when only (time, channel) arrays are available.
    """
    eye = np.eye(n_channels, dtype=np.float64)
    epochs, _ = _collect_epochs_from_loader(train_loader, n_channels)
    if not epochs:
        return eye

    X = _pad_epochs(epochs, n_channels)
    if X is None or X.shape[0] == 0:
        return eye

    rank = inner_rank or max(1, n_channels // 2)
    rank = int(max(1, min(rank, n_channels)))

    R = np.zeros((n_channels, n_channels), dtype=np.float64)
    for i in range(X.shape[0]):
        ep = X[i]
        T = ep.shape[1]
        if T == 0:
            continue
        R += ep @ ep.T / T
    R /= max(X.shape[0], 1)

    try:
        evals, evecs = np.linalg.eigh(R)
    except np.linalg.LinAlgError:
        return eye

    order = np.argsort(evals)[::-1]
    U = evecs[:, order[:rank]]
    P = U @ U.T
    return _row_apply_matrix(P)


def fit_ica_matrix_from_numpy(X, sfreq=256.0, n_components=None, n_exclude=1):
    """Fit ICA from X (n_trials, n_time, n_channels)."""
    n_trials, _, n_channels = X.shape
    epochs = [np.ascontiguousarray(X[i].T, dtype=np.float64) for i in range(n_trials)]

    class _Loader:
        def __iter__(self):
            bx = np.stack([e.T for e in epochs], axis=0)
            yield bx, np.zeros(n_trials), None

    return fit_ica_matrix_from_batches(
        _Loader(), n_channels, sfreq=sfreq, n_components=n_components, n_exclude=n_exclude
    )
