#!/usr/bin/env python3
"""Export CIF-select channel masks from Trail_snr per-channel SNR gain CSVs."""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.cif_utils import load_cif_select_indices_from_csv


def main():
    parser = argparse.ArgumentParser(
        description="Export local channel indices with positive trial SNR gain on TRAIN split.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./Trail_snr",
        help="Directory containing {DATASET}_cif_trial_snr_gain_per_channel.csv",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=["APAVA", "ADFTD", "PTB", "TDBRAIN"],
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=["TRAIN"],
    )
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument(
        "--output_json",
        type=str,
        default="./Trail_snr/cif_select_masks.json",
    )
    args = parser.parse_args()

    out = {}
    for ds in args.datasets:
        csv_path = os.path.join(args.input_dir, f"{ds}_cif_trial_snr_gain_per_channel.csv")
        if not os.path.isfile(csv_path):
            print(f"Skip {ds}: missing {csv_path}")
            continue
        indices = load_cif_select_indices_from_csv(
            csv_path,
            splits=list(args.splits),
            min_delta=args.min_delta,
        )
        out[ds] = {
            "csv": csv_path,
            "splits": list(args.splits),
            "min_delta_db": float(args.min_delta),
            "local_indices": indices,
            "k_selected": len(indices),
        }
        print(f"{ds}: {len(indices)} channels -> {indices}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)
    print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()
