
<div align="center">
<h1> Channel-Imposed Fusion: A Simple yet Effective Method for Medical Time Series Classification </h1>
</div>

Co-training has achieved significant success in the field of semi-supervised learning; however, the *homogenization phenomenon*, which arises from multiple models tending towards similar decision boundaries, remains inadequately addressed. To tackle this issue, we propose a novel algorithm called **&beta;-FFT** from the perspectives of data diversity and training structure.First, from the perspective of data diversity, we introduce a nonlinear interpolation method based on the **Fast Fourier Transform (FFT)**. This method generates more diverse training samples by swapping low-frequency components between pairs of images, thereby enhancing the model's generalization capability. Second, from the structural perspective, we propose a differentiated training strategy to alleviate the homogenization issue in co-training. In this strategy, we apply additional training with labeled data to one model in the co-training framework, while employing linear interpolation based on the **Beta (&beta;)** distribution for the unlabeled data as a regularization term for the additional training. This approach enables us to effectively utilize the limited labeled data while simultaneously improving the model's performance on unlabeled data, ultimately enhancing the overall performance of the system.Experimental results demonstrate that **&beta;-FFT** outperforms current state-of-the-art (SOTA) methods on three public medical image datasets.




## 1. Installation
```
conda create -n TCN python=3.8.10
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```
## 2. Dataset
All data can be accessed in Medformer.
```
├── ./dataset
    ├── [ADFTD]
    ├── [APAVA]
    ├── [PTB]
    ├── [TDBRAIN]
    ├── [PTB-XL]
```

## 3. Usage
**To train a model**
```
I will organize my code after submitting another piece of work to NeurIPS 2025, and plan to make the code publicly available by the end of June
``` 
**To test a model**

We have given the model of the corresponding results of our paper
```
code/model
```

if you want to test, you can follow the code below.

```
python test_ACDC_beta_FFT.py  # for ACDC testing
python test_promise12_beta_FFT.py  # for PROMISE12 testing
python test_MSCMR_split1.py  # for MS-CMRSEG19_split1
python test_MSCMR_split2.py  # for MS-CMRSEG19_split2
```

## Acknowledgements
Our code is largely based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [BCP](https://github.com/DeepMed-Lab-ECNU/BCP), [DiffRect](https://github.com/CUHK-AIM-Group/DiffRect/),and [ABD](https://github.com/chy-upc/ABD). Thanks for these authors for their valuable work, hope our work can also contribute to related research.

