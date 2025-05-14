
<div align="center">
<h1> Channel-Imposed Fusion: A Simple yet Effective Method for Medical Time Series Classification </h1>
</div>

The automatic classification of medical time series signals, such as electroencephalogram (EEG) and electrocardiogram (ECG), plays a pivotal role in clinical decision support and early detection of diseases. Although Transformer based models have achieved notable performance by implicitly modeling temporal dependencies through self-attention mechanisms, their inherently complex architectures and opaque reasoning processes undermine their trustworthiness in high stakes clinical settings. In response to these limitations, this study shifts focus toward a modeling paradigm that emphasizes interpretability and structural transparency, aligning more closely with the intrinsic characteristics of medical data. We propose a novel method Channel Imposed Fusion (CIF) that explicitly models temporal relationships across multiple channels, effectively reducing redundancy and improving feature diversity, thus improving model interpretability. Furthermore, we integrate CIF with the Temporal Convolutional Network (TCN), known for its structural simplicity and controllable receptive field, to construct an efficient and interpretable classification framework. Experimental results on several public EEG and ECG datasets demonstrate that the proposed method not only surpasses current mainstream Transformer models in classification accuracy but also achieves a desirable balance between performance and interpretability, thus offering a novel perspective for medical time series classification.



<span style="color:red;">All code is available</span>

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

We have given the model of the corresponding results of our paper
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

