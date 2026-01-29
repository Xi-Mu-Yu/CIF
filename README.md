
<div align="center">
<h1> Channel-Imposed Fusion: A Simple yet Effective Method for Medical Time Series Classification </h1>
</div>

**A simple model with good features beats a complex model with poor features.**

Medical time series (MedTS), such as EEG and ECG, are critical for clinical diagnosis but face two main challenges: generic model architectures fail to fully leverage physiological priors, and the inherently low signal-to-noise ratio (SNR) limits feature representation. To address these issues, we propose a simple data-centric framework that effectively combines physiological priors with deep learning.
First, **Channel-Imposed Fusion (CIF)** is inspired by physiological priors and is based on the idea that, ideally, information from other channels can enhance the signal-to-noise ratio (SNR) of the current channel. By linearly fusing signals across channels, CIF effectively enhances feature representations. To improve efficiency on multi-channel data, we adopt global fusion coefficients and reorder the channels according to functional regions, achieving physiologically meaningful fusion with minimal parameters.
Secondly, we build a simplified HM-BiTCN on top of TCNs, aiming to capture both forward and backward temporal dependencies with the simplest possible design.
Experimental results demonstrate that the combination of CIF and HM-BiTCN achieves state-of-the-art performance across multiple MedTS benchmarks, showing that competitive results can be obtained without complex model design.

**We hope this work encourages the community to reconsider the core of medical time series classification: should it be driven primarily by data-centric strategies, model-centric design, or a combination of both?**



## All code is available

## 1. Installation
```
conda create -n TCN python=3.8.10
conda activate TCN
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```
## 2. Dataset
All data can be accessed in [Medformer](https://github.com/DL4mHealth/Medformer).
```
├── ./dataset
    ├── [ADFTD]
    ├── [APAVA]
    ├── [PTB]
    ├── [TDBRAIN]
    ├── [PTB-XL]
```

## 3. Usage
**To test a model**

```

./checkpoints/classification

```
The training logs for the paper results can be found in:

```
./log/classification
```

The training results for the paper results can be found in:
```
./results/classification
```

if you want to test, you can follow the code below.
```
bash ./scripts/test.sh
```

**To train a model**
```
bash ./scripts/TCN.sh
``` 


## Acknowledgements
Our code is largely based on [Medformer](https://github.com/DL4mHealth/Medformer) and [Time-Series-Library](https://github.com/thuml/Time-Series-Library). Thanks for these authors for their valuable work, hope our work can also contribute to related research.

