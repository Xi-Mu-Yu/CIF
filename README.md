
<div align="center">
<h1> Channel-Imposed Fusion: A Simple yet Effective Method for Medical Time Series Classification </h1>
</div>

Medical time series (MedTS) such as EEG and ECG are critical for clinical diagnosis, yet existing deep learning approaches often struggle with two key challenges: the misalignment between domain-specific physiological knowledge and generic architectures, and the inherent low signal-to-noise ratio (SNR) of MedTS. To address these limitations, we shift from a conventional model-centric paradigm toward a data-centric perspective grounded in physiological principles. We propose Channel-Imposed Fusion (CIF), a method that explicitly encodes causal inter-channel relationships by linearly combining signals under domain-informed constraints, thereby enabling interpretable signal enhancement and noise suppression. To further demonstrate the effectiveness of data-centric design, we develop a simple yet powerful model, Hidden-layer Mixed Bidirectional Temporal Convolutional Network (HM-BiTCN), which, when combined with CIF, consistently outperforms Transformer-based approaches on multiple MedTS benchmarks and achieves new state-of-the-art performance on general time series classification datasets. Moreover, CIF is architecture-agnostic and can be seamlessly integrated into mainstream models such as Transformers, enhancing their adaptability to medical scenarios. Our work highlights the necessity of rethinking MedTS classification from a data-centric perspective and establishes a transferable framework for bridging physiological priors with modern deep learning architectures.



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

We have given the model of the corresponding results of our paper, **checkpoints are included in the supplementary material  we provided in openreview**.

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

