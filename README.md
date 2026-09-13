
<div align="center">
<h1> Channel-Imposed Fusion: A Simple yet Effective Method for Medical Time Series Classification </h1>
</div>



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
Our code is largely based on [Medformer](https://github.com/DL4mHealth/Medformer)  [ADformer](https://github.com/DL4mHealth/ADformer). [ecg_ptbxl_benchmarking
](https://github.com/helme/ecg_ptbxl_benchmarking) and [Time-Series-Library](https://github.com/thuml/Time-Series-Library). Thanks for these authors for their valuable work, hope our work can also contribute to related research.

