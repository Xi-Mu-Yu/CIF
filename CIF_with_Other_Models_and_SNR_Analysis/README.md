
<div align="center">
<h1> Channel-Imposed Fusion: A Simple yet Effective Method for Medical Time Series Classification </h1>
</div>

Proposed model: **HMTCN** (in code: `--model HMBiTCN`).

---

## 1. Installation

```bash
conda create -n TCN python=3.8.10
conda activate TCN
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```

---

## 2. Dataset

Download data from [Medformer](https://github.com/DL4mHealth/Medformer), then place under `./dataset/` or set `--root_path` (scripts default to `../Medformer_train/dataset/`).

```
{root_path}/
├── Feature/
│   └── *.npy
└── Label/
    └── label.npy
```

Supported datasets: `APAVA`, `ADFTD`, `PTB`, `TDBRAIN`.

---

## 3. Usage

All experiments use `run.py`:

```bash
python -u run.py --task_name classification --is_training 1 [options...]
```

### Train HMTCN (paper, 4 datasets)

```bash
bash ./scripts/classification/HMBiTCN_CIF_select.sh
```

### Train baselines

```bash
bash ./scripts/classification/EEGNet.sh
bash ./scripts/classification/Medformer.sh
bash ./scripts/classification/TCN.sh
bash ./scripts_additional/ADformer.sh
bash ./scripts_additional/ResNet1d.sh
bash ./scripts_additional/Conformer.sh
# more scripts in scripts_additional/
```

### Train on a single dataset (APAVA example)

```bash
export CUDA_VISIBLE_DEVICES=0
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/APAVA/ \
  --model_id APAVA-Indep-HMBiTCN-CIF-select \
  --model HMBiTCN \
  --data APAVA \
  --e_layers 6 --batch_size 32 --d_model 128 --d_ff 256 \
  --des 'Exp_10' --itr 5 --learning_rate 0.0001 \
  --train_epochs 100 --patience 10 \
  --t 1 --n 9 --a -0.8 --b -0.6 \
  --learnab True --cif_select True \
  --cif_snr_gain_csv ./Trail_snr/APAVA_cif_trial_snr_gain_per_channel.csv \
  --cif_snr_splits TRAIN --gpu 0
```

### Test (load checkpoint)

```bash
bash ./scripts/classification/test.sh
```

Set `--is_training 0` with the same hyperparameters as training.

### Outputs

| Artifact | Path |
|----------|------|
| Checkpoints | `./checkpoints/classification/{model_id}/{model}/` |
| Logs | `./log/classification/{model_id}/{model}/` |
| Results | `./results/classification/{model_id}/{model}/result_classification.txt` |

Paper logs and results are included under `./log/` and `./results/`.

---

## 4. SNR Analysis

`Trail_snr/` provides tools to evaluate whether **CIF improves trial-averaged SNR**, and to derive **CIF-select** channel masks (apply CIF only on channels where fusion raises SNR).

**Trial-averaged SNR (dB):** treat trial mean as signal, trial-to-trial residual as noise, then compute `10·log10(Var(signal)/Var(noise))`. Compare SNR before vs. after CIF fusion.

### Precomputed files

```
Trail_snr/
├── APAVA_cif_trial_snr_gain_per_channel.csv
├── ADFTD_cif_trial_snr_gain_per_channel.csv
├── PTB_cif_trial_snr_gain_per_channel.csv
├── TDBRAIN_cif_trial_snr_gain_per_channel.csv
└── cif_select_masks.json
```

Training with `--cif_snr_gain_csv` auto-loads channels where `snr_delta_ch_db > 0` on the TRAIN split.

### Step 1: Offline CIF SNR gain (per channel)

Compute SNR before/after CIF on raw data splits:

```bash
python Trail_snr/analyze_cif_trial_snr_gain.py \
  --datasets_root ../Medformer_train/dataset \
  --datasets APAVA \
  --split TRAIN \
  --device cpu \
  --t 1 --n 9 --a -0.8 --b -0.6
```

Outputs `{DATASET}_cif_trial_snr_gain_per_channel.csv` and `_summary.csv` under `./Trail_snr/`.

Per-dataset CIF args: APAVA `(t=1,n=9,a=-0.8,b=-0.6)`, ADFTD `(1,10,-0.19,-0.27)`, PTB `(1,8,0.21,-0.5)`, TDBRAIN `(-1,25,1,1.1)`.

### Step 2: Export CIF-select masks

```bash
python Trail_snr/export_cif_select_masks.py \
  --input_dir ./Trail_snr \
  --datasets APAVA ADFTD PTB TDBRAIN \
  --output_json ./Trail_snr/cif_select_masks.json
```

### Step 3: Analyze SNR during training (optional)

Enable SNR logging in training to write `snr_iter_stats_seed*.txt` under checkpoints:

```bash
python -u run.py ... --log_snr True --use_cif True
```

Aggregate logs across runs:

```bash
python Trail_snr/analyze_trial_snr.py \
  --root_dir ./checkpoints \
  --datasets APAVA ADFTD PTB TDBRAIN \
  --output_dir ./Trail_snr
```

Outputs `trial_snr_by_seed.csv` and `trial_snr_summary_by_dataset.csv`.

More details: `Trail_snr/README.md`.

---

## Acknowledgements

Code based on [Medformer](https://github.com/DL4mHealth/Medformer) and [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
