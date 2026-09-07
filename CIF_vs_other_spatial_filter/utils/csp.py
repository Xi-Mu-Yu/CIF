"""
Common Spatial Patterns (CSP) via MNE-Python.

Training trials are collected as (n_epochs, n_channels, n_times). Multiclass
labels are reduced to binary CSP (smallest label vs all others). The returned
filter matrix W has shape (n_channels, n_channels) and is applied in TCN as
x @ W per time step (W = csp.filters_.T).
"""

import numpy as np

try:
    from mne.decoding import CSP
except ImportError as exc:
    raise ImportError(
        "CSP spatial mode requires MNE-Python. Install with: pip install mne"
    ) from exc


def _trials_from_batch(batch, n_channels):
    if len(batch) == 3:
        batch_x, targets, padding_masks = batch
    else:
        batch_x, targets = batch
        padding_masks = None

    bx = batch_x.detach().cpu().numpy()
    y = np.asarray(targets).reshape(-1)

    if padding_masks is not None:
        pm = padding_masks.detach().cpu().numpy().astype(bool)
    else:
        pm = None

    epochs = []
    labels = []
    for i in range(bx.shape[0]):
        if pm is not None:
            valid = pm[i]
            if not np.any(valid):
                continue
            xi = bx[i, valid, :]
        else:
            xi = bx[i]

        if xi.shape[-1] != n_channels:
            raise ValueError(
                f"CSP: trial has {xi.shape[-1]} channels, expected enc_in={n_channels}"
            )

        epochs.append(np.ascontiguousarray(xi.T, dtype=np.float64))
        labels.append(int(y[i]))

    return epochs, labels


def _collect_epochs_from_loader(train_loader, n_channels):
    epochs = []
    labels = []
    for batch in train_loader:
        batch_epochs, batch_labels = _trials_from_batch(batch, n_channels)
        epochs.extend(batch_epochs)
        labels.extend(batch_labels)
    return epochs, np.asarray(labels, dtype=np.int64)


def _binary_labels(y):
    label_min = int(y.min())
    return (y == label_min).astype(np.int64), label_min


def _pad_epochs(epochs, n_channels):
    if not epochs:
        return None
    max_t = max(epoch.shape[1] for epoch in epochs)
    X = np.zeros((len(epochs), n_channels, max_t), dtype=np.float64)
    for i, epoch in enumerate(epochs):
        X[i, :, : epoch.shape[1]] = epoch
    return X


def _fit_mne_csp(X, y, n_channels, reg=1e-6):
    """
    Fit MNE CSP on X (n_epochs, n_channels, n_times), y binary (0/1).

    Returns W (n_channels, n_channels) for row-wise application x @ W.
    """
    eye = np.eye(n_channels, dtype=np.float64)
    if X is None or X.shape[0] == 0:
        return eye
    if len(np.unique(y)) < 2:
        return eye

    csp = CSP(
        n_components=n_channels,
        reg=reg,
        norm_trace=True,
        transform_into="csp_space",
        log=None,
    )
    try:
        csp.fit(X, y)
    except Exception:
        return eye

    if csp.filters_ is None:
        return eye

    # MNE transform: filters_[:n_components] @ epoch  ->  W = filters_.T for x @ W
    return np.asarray(csp.filters_, dtype=np.float64).T


def fit_csp_filters_from_batches(train_loader, n_channels, reg=1e-6):
    """
    Collect training trials from train_loader and fit MNE CSP filters.

    Groups: trials with global minimum label vs all others (multiclass -> binary CSP).

    Args:
        train_loader: iterable yielding (batch_x, targets, padding_masks) batches.
        n_channels: enc_in
        reg: passed to mne.decoding.CSP (shrinkage in [0, 1] when float)

    Returns:
        W: (n_channels, n_channels) float64
    """
    epochs, y = _collect_epochs_from_loader(train_loader, n_channels)
    if not epochs:
        return np.eye(n_channels, dtype=np.float64)

    y_bin, _ = _binary_labels(y)
    X = _pad_epochs(epochs, n_channels)
    return _fit_mne_csp(X, y_bin, n_channels, reg=reg)


def fit_csp_filters_from_numpy(X, y, reg=1e-6):
    """
    Fit MNE CSP from numpy arrays.

    Args:
        X: (n_trials, n_time, n_channels)
        y: (n_trials,) int labels — smallest label vs rest for binary CSP.

    Returns:
        W: (n_channels, n_channels) float64
    """
    n_trials, _, n_channels = X.shape
    if n_trials == 0:
        return np.eye(n_channels, dtype=np.float64)

    y = np.asarray(y).reshape(-1)
    y_bin, _ = _binary_labels(y)
    epochs = [np.ascontiguousarray(X[i].T, dtype=np.float64) for i in range(n_trials)]
    X_ct = _pad_epochs(epochs, n_channels)
    return _fit_mne_csp(X_ct, y_bin, n_channels, reg=reg)
