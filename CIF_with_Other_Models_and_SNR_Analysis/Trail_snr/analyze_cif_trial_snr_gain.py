import argparse
import os
import sys
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import torch

# Ensure repo root is on sys.path when running as:
#   python Trail_snr/analyze_*.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_provider.data_loader import APAVALoader, ADFTDLoader, PTBLoader, TDBRAINLoader


DATASET_LOADERS = {
    "APAVA": APAVALoader,
    "ADFTD": ADFTDLoader,
    "PTB": PTBLoader,
    "TDBRAIN": TDBRAINLoader,
}


def trial_avg_snr_db(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T, C] (changed channels only)
    Replicates models/HMBiTCN.py::_trial_avg_snr_db:
      signal = mean over trials (dim=0)
      noise  = residual
      SNR(dB) = 10*log10(Var(signal)/Var(noise))
    Returns scalar tensor.
    """
    if x.size(0) < 2:
        return torch.tensor(float("nan"), device=x.device)
    # x: [B,T,C]
    # signal = mean over trials: [T,C]
    mean_b = x.mean(dim=0)  # [T,C]
    # Var(signal) over all elements (T,C): matches torch.var over signal with unbiased=False.
    signal_power = mean_b.var(unbiased=False).clamp_min(1e-12)
    # noise = x - mean_b, Var(noise) over all elements equals:
    #   mean_{t,c} Var_b(x_{b,t,c}) = mean_{t,c}(E_b[x^2] - (E_b[x])^2)
    mean_x2_b = x.pow(2).mean(dim=0)  # [T,C]
    residual_var = (mean_x2_b - mean_b.pow(2)).clamp_min(0.0)  # [T,C]
    noise_power = residual_var.mean().clamp_min(1e-12)  # scalar
    return 10.0 * torch.log10(signal_power / noise_power)


def trial_avg_snr_db_per_channel(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T, C] (changed channels only)
    Replicates models/HMBiTCN.py::_trial_avg_snr_db_per_channel.
    Returns per-channel SNR: [C]
    """
    if x.size(0) < 2:
        return torch.full((x.size(2),), float("nan"), device=x.device)
    # x: [B,T,C]
    mean_b = x.mean(dim=0)  # [T,C]
    mean_x2_b = x.pow(2).mean(dim=0)  # [T,C]
    # signal_power_c = Var_t(mean_b_{t,c})
    signal_power = mean_b.var(dim=0, unbiased=False).clamp_min(1e-12)  # [C]
    # noise_power_c = mean_t( E_b[x^2] - (E_b[x])^2 )
    residual_var = (mean_x2_b - mean_b.pow(2)).clamp_min(0.0)  # [T,C]
    noise_power = residual_var.mean(dim=0).clamp_min(1e-12)  # [C]
    return 10.0 * torch.log10(signal_power / noise_power)


def load_split_dataset(dataset_name: str, root_path: str, split: str, device: str):
    loader_cls = DATASET_LOADERS[dataset_name]
    dummy_args = SimpleNamespace()
    ds = loader_cls(dummy_args, root_path=root_path, flag=split)
    # ds.X is a numpy array: [N_trials, T, C]
    X = ds.X
    y = ds.y
    return X, y


def apply_cif_fusion(x_enc: torch.Tensor, a: float, b: float, t: int, n: int):
    """
    Replicate models/HMBiTCN.py fusion CIF snippet:
      front_ = x_enc[:, :, :n]
      back_  = x_enc[:, :, -n:]
      added  = front_*a + back_*b
      if t > 0: x_enc_new[:, :, :n] = added
      else:     x_enc_new[:, :, -n:] = added
    Then return (changed_before, changed_after) for the side controlled by t.
    """
    front = x_enc[:, :, :n]
    back = x_enc[:, :, -n:]
    added = front * a + back * b
    if t > 0:
        return front, added
    return back, added


@torch.no_grad()
def _accumulate_mean_and_mean2(
    X: np.ndarray,
    device: torch.device,
    a: float,
    b: float,
    t: int,
    n: int,
    chunk_size: int,
):
    """
    Stream over trials to compute:
      mean_b    = E_b[x]      over trials  -> [T,n]
      mean_x2_b = E_b[x^2]    over trials  -> [T,n]
    for BOTH before and after (CIF).
    """
    n_trials, T, C = X.shape
    if chunk_size <= 0:
        chunk_size = n_trials

    sum_before = None
    sum_before2 = None
    sum_after = None
    sum_after2 = None

    for s in range(0, n_trials, chunk_size):
        e = min(n_trials, s + chunk_size)
        xb = torch.from_numpy(X[s:e]).to(device=device, dtype=torch.float32)  # [B,T,C]
        changed_before, changed_after = apply_cif_fusion(xb, a=a, b=b, t=t, n=n)  # [B,T,n]
        cb = changed_before  # [B,T,n]
        ca = changed_after

        sb = cb.sum(dim=0)  # [T,n]
        sb2 = cb.pow(2).sum(dim=0)
        sa = ca.sum(dim=0)
        sa2 = ca.pow(2).sum(dim=0)

        if sum_before is None:
            sum_before = sb
            sum_before2 = sb2
            sum_after = sa
            sum_after2 = sa2
        else:
            sum_before += sb
            sum_before2 += sb2
            sum_after += sa
            sum_after2 += sa2

    denom = float(n_trials)
    mean_before = sum_before / denom
    mean_before2 = sum_before2 / denom
    mean_after = sum_after / denom
    mean_after2 = sum_after2 / denom
    return mean_before, mean_before2, mean_after, mean_after2


def snr_from_mean_stats(mean_b: torch.Tensor, mean_x2_b: torch.Tensor) -> torch.Tensor:
    """
    mean_b:    [T,n]  E_b[x]
    mean_x2_b: [T,n]  E_b[x^2]
    Returns scalar SNR(dB) using the same definition as HMBiTCN::_trial_avg_snr_db.
    """
    signal_power = mean_b.var(unbiased=False).clamp_min(1e-12)
    residual_var = (mean_x2_b - mean_b.pow(2)).clamp_min(0.0)
    noise_power = residual_var.mean().clamp_min(1e-12)
    return 10.0 * torch.log10(signal_power / noise_power)


def snr_per_channel_from_mean_stats(mean_b: torch.Tensor, mean_x2_b: torch.Tensor) -> torch.Tensor:
    """
    Returns per-channel SNR(dB): [n]
    """
    signal_power = mean_b.var(dim=0, unbiased=False).clamp_min(1e-12)  # [n]
    residual_var = (mean_x2_b - mean_b.pow(2)).clamp_min(0.0)  # [T,n]
    noise_power = residual_var.mean(dim=0).clamp_min(1e-12)  # [n]
    return 10.0 * torch.log10(signal_power / noise_power)


def _normalize_splits(splits: List[str]) -> List[str]:
    """Expand ALL -> TRAIN, VAL, TEST; dedupe while preserving order."""
    expanded: List[str] = []
    for s in splits:
        if s == "ALL":
            expanded.extend(["TRAIN", "VAL", "TEST"])
        else:
            expanded.append(s)
    seen = set()
    out: List[str] = []
    for s in expanded:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main():
    parser = argparse.ArgumentParser("Compute CIF trial-averaged SNR gain from raw data (no snr txt parsing).")
    parser.add_argument("--datasets_root", type=str, default="../Medformer_train/dataset", help="Parent dir containing APAVA/ADFTD/PTB/TDBRAIN")
    parser.add_argument("--datasets", type=str, nargs="*", default=["APAVA", "ADFTD", "PTB", "TDBRAIN"])
    parser.add_argument(
        "--split",
        dest="splits",
        nargs="+",
        default=["TEST"],
        metavar="SPLIT",
        choices=["TRAIN", "VAL", "TEST", "ALL"],
        help="One or more splits. Use ALL for TRAIN+VAL+TEST in one run. Default: TEST.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    # CIF parameters (same names as run.py)
    parser.add_argument("--t", type=int, default=1, help="If t>0: operate on front channels; else back channels")
    parser.add_argument("--n", type=int, default=8, help="Number of changed channels")
    parser.add_argument("--a", type=float, default=1.0, help="CIF fusion coefficient a")
    parser.add_argument("--b", type=float, default=1.0, help="CIF fusion coefficient b")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Trials per chunk moved to device (for GPU/CPU streaming)")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./Trail_snr",
        help="Output directory; writes {dataset}_cif_trial_snr_gain_summary.csv and _per_channel.csv per dataset",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    split_list = _normalize_splits(list(args.splits))

    for ds_name in args.datasets:
        if ds_name not in DATASET_LOADERS:
            raise ValueError(f"Unknown dataset: {ds_name}")
        root_path = os.path.join(args.datasets_root, ds_name)
        rows = []
        per_channel_rows = []

        for split in split_list:
            X, y = load_split_dataset(ds_name, root_path=root_path, split=split, device=str(device))

            if not isinstance(X, np.ndarray):
                X = np.asarray(X)

            if X.ndim != 3:
                raise ValueError(f"{ds_name} X must be [N,T,C], got shape {X.shape}")
            n_trials, T, C = X.shape
            if args.n > C:
                raise ValueError(f"{ds_name}: n={args.n} can't be > C={C}")
            if n_trials < 2:
                raise ValueError(f"{ds_name}: split={split} has too few trials: {n_trials}")

            mean_b, mean_b2, mean_a, mean_a2 = _accumulate_mean_and_mean2(
                X,
                device=device,
                a=args.a,
                b=args.b,
                t=args.t,
                n=args.n,
                chunk_size=args.chunk_size,
            )

            snr_before = snr_from_mean_stats(mean_b, mean_b2)
            snr_after = snr_from_mean_stats(mean_a, mean_a2)
            snr_delta = snr_after - snr_before

            snr_before_ch = snr_per_channel_from_mean_stats(mean_b, mean_b2)
            snr_after_ch = snr_per_channel_from_mean_stats(mean_a, mean_a2)
            snr_delta_ch = snr_after_ch - snr_before_ch

            before_mean_ch = torch.nanmean(snr_before_ch)
            after_mean_ch = torch.nanmean(snr_after_ch)
            delta_mean_ch = torch.nanmean(snr_delta_ch)

            rows.append(
                {
                    "dataset": ds_name,
                    "split": split,
                    "device": str(device),
                    "n_trials": int(n_trials),
                    "T": int(T),
                    "C": int(C),
                    "t": int(args.t),
                    "n": int(args.n),
                    "a": float(args.a),
                    "b": float(args.b),
                    "changed_side": "front" if args.t > 0 else "back",
                    "snr_before_db": float(snr_before.detach().cpu().item()),
                    "snr_after_db": float(snr_after.detach().cpu().item()),
                    "snr_delta_db": float(snr_delta.detach().cpu().item()),
                    "snr_before_mean_ch_db": float(before_mean_ch.detach().cpu().item()),
                    "snr_after_mean_ch_db": float(after_mean_ch.detach().cpu().item()),
                    "snr_delta_mean_ch_db": float(delta_mean_ch.detach().cpu().item()),
                }
            )

            for ci in range(args.n):
                per_channel_rows.append(
                    {
                        "dataset": ds_name,
                        "split": split,
                        "t": int(args.t),
                        "n": int(args.n),
                        "a": float(args.a),
                        "b": float(args.b),
                        "channel_changed_local_idx": int(ci),
                        "snr_before_ch_db": float(snr_before_ch[ci].detach().cpu().item()),
                        "snr_after_ch_db": float(snr_after_ch[ci].detach().cpu().item()),
                        "snr_delta_ch_db": float(snr_delta_ch[ci].detach().cpu().item()),
                    }
                )

        prefix = f"{ds_name}_cif_trial_snr_gain"
        out_df = pd.DataFrame(rows)
        out_path = os.path.join(args.output_dir, f"{prefix}_summary.csv")
        out_df.to_csv(out_path, index=False)

        per_df = pd.DataFrame(per_channel_rows)
        per_path = os.path.join(args.output_dir, f"{prefix}_per_channel.csv")
        per_df.to_csv(per_path, index=False)

        print(f"Wrote: {out_path} (rows={len(out_df)})")
        print(f"Wrote: {per_path} (rows={len(per_df)})")


if __name__ == "__main__":
    main()

# Examples:
#   --split TEST
#   --split ALL
#   --split TRAIN VAL TEST
# python Trail_snr/analyze_cif_trial_snr_gain.py --datasets_root ../Medformer_train/dataset --datasets APAVA --split ALL --device cpu --t 1 --n 9 --a -0.8 --b -0.6

#  python Trail_snr/analyze_cif_trial_snr_gain.py --datasets_root ../Medformer_train/dataset --datasets PTB --split ALL --device cpu --t 1 --n 8 --a 0.21 --b -0.5


# python Trail_snr/analyze_cif_trial_snr_gain.py --datasets_root ../Medformer_train/dataset --datasets TDBRAIN --split ALL --device cpu --t -1 --n 25 --a 1 --b 1.1


# python Trail_snr/analyze_cif_trial_snr_gain.py ... --split ALL