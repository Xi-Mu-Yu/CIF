<div align="center">
<h1>Channel-Imposed Fusion (CIF) vs. Spatial Filtering Methods</h1>
<p>A unified benchmark framework for channel-level spatial preprocessing in medical time series classification</p>
</div>

This project implements a **unified interface for 10 channel-level spatial preprocessing methods** on top of an **HM-BiTCN** classification backbone, enabling systematic comparison of **CIF (Channel-Imposed Fusion)** against classical EEG/ECG spatial filtering strategies. All methods are switched via `--spatial_mode` and share the same training pipeline and evaluation protocol.

> **Key takeaway:** For multichannel medical time series, appropriate channel-level spatial preprocessing often matters more than stacking more complex temporal models. CIF fuses cross-channel information with very few parameters in an end-to-end learnable way, outperforming or matching traditional fixed filtering on most benchmarks.

---

## 1. What is CIF?

**Channel-Imposed Fusion (CIF)** is built on a simple physiological prior: when channels are ordered by functional region, the **front** and **back** segments of the channel sequence correspond to different brain areas or lead groups. Linearly fusing signals from both ends can improve the signal-to-noise ratio (SNR) of the target channels.

Given input `x ∈ R^{B×T×C}`, take the first `n` channels as `front` and the last `n` channels as `back`, then compute the fused block:

```
added = front × a + back × b
```

Write `added` back into one end of the channel sequence (controlled by `--t`: front or back); all other channels remain unchanged. Coefficients `a` and `b` can be made learnable with `--learnab True` and optimized jointly with the TCN under the classification loss.

**CIF at a glance:**

| Property | CIF |
|----------|-----|
| Parameters | 2 (`a`, `b`), or 0 (fixed coefficients) |
| Requires training-set fitting | No (coefficients learned by gradient or set manually) |
| End-to-end | Yes |
| Physiological prior | Channels ordered by functional region; front/back represent different areas |
| Output channels | Unchanged (`C` dimensions) |

---

## 2. Spatial Filtering Methods Overview

This repo inserts a spatial preprocessing layer **before** the TCN encoder. All methods preserve input/output shape `(B, T, C)`:

```
Raw multichannel signal  →  [Spatial preprocessing]  →  HM-BiTCN  →  Classification head
                                    ↑ spatial_mode
```

### 2.1 Method Taxonomy

| Category | `--spatial_mode` | Brief description | Matrix / params | Train-set fit | End-to-end |
|----------|------------------|-------------------|-----------------|---------------|------------|
| **CIF (ours)** | `cif` | Linear fusion of front/back channel blocks, written to one end | 2 scalars | No | Yes |
| Common average reference | `car` | Subtract global channel mean (CAR) | Fixed | No | No |
| Adjacent bipolar | `bipolar` | Adjacent difference along channel axis: `x_{i+1} - x_i` | Fixed | No | No |
| Graph Laplacian | `laplacian` | Path-graph discrete Laplacian: `2x - x_{left} - x_{right}` | Fixed | No | No |
| Standard limb leads | `ecg` | First 3 channels → Einthoven leads I/II/III | Fixed | No | No |
| Common Spatial Patterns | `csp` | Generalized eigen-decomposition on trial covariances (min class vs. rest) | `C×C` fixed | **Yes** | No |
| Independent Component Analysis | `ica` | MNE FastICA, reconstruct after dropping high-kurtosis components | `C×C` fixed | **Yes** | No |
| Signal-Space Projection | `ssp` | MNE PCA, project out high-variance spatial modes | `C×C` fixed | **Yes** | No |
| Signal-Space Separation | `sss` | Inner subspace reconstruction from spatial covariance (EEG-compatible analogue) | `C×C` fixed | **Yes** | No |
| Learnable linear mixing | `linear` | `nn.Linear(C, C, bias=False)`, identity initialization | `C²` learnable | No | **Yes** |

> **MNE dependency:** `csp`, `ica`, `ssp`, and `sss` require `pip install mne` (listed in `requirements.txt`). Spatial matrices are fitted on the training set before training and fixed at test time.

### 2.2 Key Differences from CIF

**vs. CAR / Bipolar / Laplacian (classical fixed re-referencing)**

- These three use **closed-form formulas** with no extra parameters and no dependence on labels or training statistics.
- CAR removes global common-mode noise; Bipolar / Laplacian emphasize local spatial differences.
- CIF does not subtract a global mean. Instead, it **selectively** fuses functional-region blocks at both ends of the sequence, preserving more region-specific information. With learnable `a` and `b`, it adapts to the task.

**vs. CSP (Common Spatial Patterns)**

- CSP estimates class-conditional covariance differences on the training set and extracts discriminative spatial filters—a BCI classic.
- For multiclass tasks, this implementation uses **binary CSP: smallest label vs. all others**.
- The CSP matrix is **fixed** after fitting and cannot be adjusted via backprop through the TCN; CIF uses only 2 parameters but is **fully differentiable**.
- Empirically, CIF clearly outperforms CSP on APAVA and PTB; on TDBRAIN, CSP is close to CIF.

**vs. ICA / SSP / SSS (MNE decomposition methods)**

- **ICA:** Blind source separation; automatically drops high-kurtosis artifact components (default: 1). Good for ocular/muscle artifacts, but does not use class labels.
- **SSP:** Projects out the highest-variance spatial directions on the training set (default: 1), similar to removing common-mode / bad channels.
- **SSS:** Keeps the top `rank` principal subspace of the spatial covariance (default: `C/2`), suppressing outer-subspace interference.
- All three are fitted once before training and **cannot be fine-tuned**; CIF has far fewer parameters than `linear` (`C²`) yet is more stable on most datasets.

**vs. Linear (learnable full-channel mixing)**

- `linear` mode provides a fully connected `C×C` channel mixer—a "no prior" upper-bound baseline for CIF.
- Parameter count grows as `C²` (e.g., 1024 params for TDBRAIN with `C=32`), making overfitting more likely.
- CIF uses 2 parameters plus region-ordering prior and beats or matches `linear` on APAVA, PTB, and TDBRAIN.

**vs. ECG (Einthoven leads)**

- Meaningful mainly for ECG datasets such as PTB: first three channels assumed RA/LA/LL, converted to standard limb leads.
- A domain-specific hard-coded transform; CIF is generic for both EEG and ECG without lead semantics.

---

## 3. Experimental Results

On APAVA, ADFTD, TDBRAIN, and PTB, we use the same HM-BiTCN configuration and only switch `--spatial_mode`. The table below reports **mean ± std Test Accuracy** over 5 random seeds (seeds 41–45):

| Method | APAVA | ADFTD | TDBRAIN | PTB |
|--------|-------|-------|---------|-----|
| laplacian | **86.40 ± 0.94** | 49.85 ± 1.69 | 78.83 ± 2.37 | 79.52 ± 2.50 |
| **cif** | 86.30 ± 1.05 | **58.56 ± 0.93** | **93.13 ± 1.41** | **88.29 ± 1.45** |
| bipolar | 86.18 ± 1.72 | 51.12 ± 0.74 | 77.19 ± 3.57 | 79.01 ± 4.12 |
| linear | 84.33 ± 1.65 | 53.16 ± 2.37 | 88.87 ± 2.98 | 84.98 ± 2.50 |
| ecg | 83.56 ± 1.58 | 49.45 ± 2.09 | 90.54 ± 1.41 | 84.10 ± 1.56 |
| csp | 83.48 ± 0.96 | 50.24 ± 0.98 | 90.37 ± 2.38 | 76.54 ± 1.03 |
| ica | 82.12 ± 2.41 | 52.29 ± 2.28 | 91.06 ± 2.34 | 82.15 ± 1.64 |
| car | 80.73 ± 2.38 | 53.10 ± 1.26 | 74.25 ± 1.51 | 83.39 ± 2.23 |
| sss | 79.06 ± 1.29 | 54.15 ± 0.98 | 86.33 ± 1.22 | 85.25 ± 1.46 |
| ssp | 77.96 ± 0.56 | 53.93 ± 1.48 | 72.10 ± 5.16 | 79.69 ± 1.35 |

**Main findings:**

1. **CIF achieves the best Test Accuracy on 3/4 datasets** (ADFTD, TDBRAIN, PTB), with the largest gains on TDBRAIN (+2.1% vs. next-best ica) and ADFTD (+4.4% vs. next-best sss).
2. **Fixed re-referencing methods** (CAR, SSP) are generally weaker than CIF; CAR drops nearly 19 points on TDBRAIN, suggesting global mean subtraction destroys spatial structure on that dataset.
3. **CSP is label-sensitive** and performs poorly on PTB (76.54%), likely because binary reduction (min class vs. rest) does not suit the multiclass structure of that task.
4. **Laplacian slightly exceeds CIF on APAVA** (86.40 vs. 86.30), but CIF is better on the other three datasets, making it more balanced overall.
5. **Linear (`C²` params) does not beat CIF**, indicating that CIF's region-ordering prior effectively constrains the search space and yields better generalization with far fewer parameters.

Full logs and per-seed results are in `./results/classification/` and `./log/classification/`.

---

## 4. Installation

```bash
conda create -n TCN python=3.8.10
conda activate TCN
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```

Datasets are available from [Medformer](https://github.com/DL4mHealth/Medformer). Place them under:

```
./dataset/
├── ADFTD/
├── APAVA/
├── PTB/
├── TDBRAIN/
└── PTB-XL/
```

---

## 5. Usage

### 5.1 Run CIF (default)

```bash
python run.py \
  --task_name classification \
  --is_training 1 \
  --model TCN \
  --data APAVA \
  --root_path ./dataset/APAVA/ \
  --spatial_mode cif \
  --t 1 --n 9 --a -0.8 --b -0.6 --learnab True
```

### 5.2 Switch to other spatial filtering methods

Change only `--spatial_mode`; keep all other training settings identical for fair comparison:

```bash
# Classical fixed filters
--spatial_mode car        # Common average reference
--spatial_mode bipolar    # Adjacent bipolar montage
--spatial_mode laplacian  # Laplacian
--spatial_mode ecg        # Einthoven leads (ECG)

# MNE-fitted methods (matrix estimated on training set before training)
--spatial_mode csp        # optional: --csp_reg 1e-6
--spatial_mode ica        # optional: --ica_n_components --ica_n_exclude 1
--spatial_mode ssp        # optional: --ssp_n_proj 1
--spatial_mode sss        # optional: --sss_inner_rank

# Learnable baseline
--spatial_mode linear     # C×C fully connected mixing
```

### 5.3 Batch comparison experiments

```bash
# Classical spatial filters: cif / car / bipolar / laplacian / csp / ecg
bash scripts/classification/TCN.sh

# MNE methods: ica / ssp / sss / linear
bash scripts/classification/TCN_mne.sh

# Quick subset comparison
bash scripts/classification/TCN_2.sh
```

### 5.4 Key CIF hyperparameters

| Argument | Meaning | Typical values |
|----------|---------|----------------|
| `--n` | Number of front/back channels to fuse | APAVA: 9, ADFTD: 10, TDBRAIN: 25, PTB: 8 |
| `--t` | Where to write fused block: `1`=front, `-1`=back | APAVA/ADFTD/PTB: 1, TDBRAIN: -1 |
| `--a`, `--b` | Initial fusion coefficients | Dataset-specific; see `scripts/classification/TCN.sh` |
| `--learnab` | Whether `a`, `b` are learnable | True / False |

Recommended CIF configs per dataset are in `scripts/classification/TCN.sh`.

---

## 6. Code Structure

```
models/TCN.py              # Spatial preprocessing + HM-BiTCN backbone
utils/csp.py               # MNE CSP fitting
utils/mne_spatial.py       # ICA / SSP / SSS fitting
exp/exp_classification.py  # Pre-training fit logic (csp/ica/ssp/sss)
run.py                     # --spatial_mode entry point
scripts/classification/    # Batch scripts per dataset × method
results/classification/    # Evaluation results
```

Spatial preprocessing lives in `models/TCN.py` → `forward()`: depending on `spatial_mode`, it applies CIF fusion, fixed transforms, or matrix multiplication `x @ W`.

---

## 7. When to Use Which Method?

| Scenario | Recommendation |
|----------|----------------|
| General EEG/ECG classification; accuracy and efficiency | **CIF** (default) |
| Artifact removal without labels | ICA or SSP |
| BCI binary classification with sufficient training trials | CSP |
| Known ECG lead order (RA/LA/LL) | `ecg` or CIF |
| Ablation: learnable but prior-free upper bound | `linear` |
| Fast common-mode removal, compute not a concern | CAR (may lose discriminative information) |

CIF's core advantage: **encode cross-channel physiological priors with 2 learnable parameters, optimize end-to-end, require no pre-fitting on the training set, and consistently outperform classical spatial filters across multiple datasets.**

---

## Acknowledgements

This code is built on [Medformer](https://github.com/DL4mHealth/Medformer) and [Time-Series-Library](https://github.com/thuml/Time-Series-Library). We thank the original authors for their valuable open-source work.
