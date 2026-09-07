import torch
import torch.nn as nn


def load_cif_select_indices_from_csv(csv_path, splits=None, min_delta=0.0):
    """
    Derive local changed-channel indices with mean trial SNR gain > min_delta.

    csv_path: Trail_snr *_cif_trial_snr_gain_per_channel.csv
    splits: optional list like ["TRAIN"]; default in run.py is TRAIN only.
    Returns sorted list of local indices within the n changed channels.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if splits:
        df = df[df["split"].isin(splits)]
    if df.empty:
        raise ValueError(f"No rows in {csv_path} for splits={splits}")

    grouped = df.groupby("channel_changed_local_idx")["snr_delta_ch_db"].mean()
    selected = [int(idx) for idx, delta in grouped.items() if float(delta) > float(min_delta)]
    return sorted(selected)


def init_cif(module, configs):
    """Attach optional CIF (Channel Input Fusion) parameters to a model module."""
    module.use_cif = getattr(configs, "use_cif", False)
    module.t = configs.t
    module.n = configs.n
    module.cif_select = bool(getattr(configs, "cif_select", False))

    if configs.learnab:
        module.a = nn.Parameter(torch.tensor(configs.a, dtype=torch.float32))
        module.b = nn.Parameter(torch.tensor(configs.b, dtype=torch.float32))
    else:
        module.a = configs.a
        module.b = configs.b

    mask_indices = getattr(configs, "cif_channel_mask", None)
    if mask_indices is not None and len(mask_indices) > 0:
        mask = torch.zeros(module.n, dtype=torch.bool)
        for idx in mask_indices:
            local_idx = int(idx)
            if local_idx < 0 or local_idx >= module.n:
                raise ValueError(
                    f"cif_channel_mask index {local_idx} out of range for n={module.n}"
                )
            mask[local_idx] = True
        module.register_buffer("cif_channel_mask", mask)
    else:
        module.register_buffer("cif_channel_mask", None)


def apply_cif(module, x_enc):
    """Apply CIF fusion when enabled; otherwise return input unchanged."""
    if not module.use_cif:
        return x_enc

    n = module.n
    front = x_enc[:, :, :n]
    back = x_enc[:, :, -n:]
    x_enc_new = x_enc.clone()
    added = front * module.a + back * module.b

    if module.cif_select:
        if module.cif_channel_mask is None:
            raise ValueError("cif_select=True requires cif_channel_mask to be set")
        mask = module.cif_channel_mask.to(device=x_enc.device, dtype=torch.bool)
        if module.t > 0:
            x_enc_new[:, :, :n][:, :, mask] = added[:, :, mask]
        else:
            x_enc_new[:, :, -n:][:, :, mask] = added[:, :, mask]
    elif module.t > 0:
        x_enc_new[:, :, :n] = added
    else:
        x_enc_new[:, :, -n:] = added
    return x_enc_new
