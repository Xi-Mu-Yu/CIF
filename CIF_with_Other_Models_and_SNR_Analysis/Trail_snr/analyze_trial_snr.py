import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


DATASETS_DEFAULT = ["APAVA", "ADFTD", "PTB", "TDBRAIN"]


def parse_dB_list(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if s == "" or s.lower() == "nan":
        return []
    out = []
    for p in s.split(","):
        p = p.strip()
        if p == "" or p.lower() == "nan":
            continue
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out


def extract_seed_from_filename(path: str):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def infer_dataset_from_path(path: str, datasets):
    up = path.upper()
    for d in datasets:
        if d.upper() in up:
            return d
    return "UNKNOWN"


def infer_run_id(path: str, root_dir: str):
    # use relative directory containing the snr file as run identifier
    rel = os.path.relpath(os.path.dirname(path), root_dir)
    return rel.replace(os.sep, "/")


def trial_averaged_snr_from_df(df: pd.DataFrame) -> float:
    """
    Return trial-averaged SNR delta (dB) for a single snr_iter_stats_seed*.txt file.
    Priority:
      1) mean over iters of 'trial_delta_ch_mean_dB' if present
      2) mean over iters of mean(trial_delta_ch_dB_list) if present
    """
    if "trial_delta_ch_mean_dB" in df.columns:
        vals = pd.to_numeric(df["trial_delta_ch_mean_dB"], errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else float("nan")

    if "trial_delta_ch_dB_list" in df.columns:
        lists = df["trial_delta_ch_dB_list"].apply(parse_dB_list)
        per_iter_mean = lists.apply(lambda x: float(np.mean(x)) if len(x) else np.nan).dropna()
        return float(per_iter_mean.mean()) if len(per_iter_mean) else float("nan")

    return float("nan")


def main():
    parser = argparse.ArgumentParser("Analyze CIF Trial-averaged SNR for multiple datasets")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./checkpoints",
        help="Search under this directory recursively for snr_iter_stats_seed*.txt",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="snr_iter_stats_seed*.txt",
        help="Glob pattern for input txt files",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=DATASETS_DEFAULT,
        help="Dataset keywords to match in file paths",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./Trail_snr",
        help="Output directory for csv files",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.root_dir, "**", args.pattern), recursive=True))
    if not files:
        raise FileNotFoundError(f"No files matched pattern={args.pattern} under root_dir={args.root_dir}")

    rows = []
    for fp in files:
        seed = extract_seed_from_filename(fp)
        ds = infer_dataset_from_path(fp, args.datasets)
        run_id = infer_run_id(fp, args.root_dir)

        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception:
            continue

        snr = trial_averaged_snr_from_df(df)
        rows.append(
            {
                "dataset": ds,
                "seed": seed,
                "trial_avg_snr_delta_dB": snr,
                "run_id": run_id,
                "file": os.path.relpath(fp, args.root_dir).replace(os.sep, "/"),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.dropna(subset=["seed"])
    os.makedirs(args.output_dir, exist_ok=True)

    by_seed_path = os.path.join(args.output_dir, "trial_snr_by_seed.csv")
    out_df.sort_values(["dataset", "run_id", "seed"]).to_csv(by_seed_path, index=False)

    # dataset summary
    summary = (
        out_df[out_df["dataset"] != "UNKNOWN"]
        .groupby("dataset", as_index=False)["trial_avg_snr_delta_dB"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n_seeds", "mean": "mean_dB", "std": "std_dB"})
    )
    summary_path = os.path.join(args.output_dir, "trial_snr_summary_by_dataset.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Wrote: {by_seed_path} (rows={len(out_df)})")
    print(f"Wrote: {summary_path} (rows={len(summary)})")


if __name__ == "__main__":
    main()

