# [AAAI2026] MambaOVSR: Multiscale Fusion with Global Motion Modeling for Chinese Opera Video Super-Resolution

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2511.06172)

</div>

<div align="center">
  <img src="./figures/ours_demo.gif" style="width:100%; max-width:900px;" />
  <br>
</div>

### **Motivation**

</summary>

![results3](./figures/fig1_dataset.png)

</details>

<details open>
<summary>

### **MambaOVSR pipeline**

</summary>

![results3](./figures/framework.png)

</details>

<details open>
<summary>

## Get Started


### **Dataset download**
Please send a signed [dataset release agreement](./figures/LICENSE%20AGREEMENT_COVC.pdf) copy to changhua@wust.edu.cn. If your application is passed, we will send the download link of the dataset.

## Dataset Preparation
<!-- The COVC dataset we built will be made public later. -->
```
├── datasets
    ├── Train
        ├── sequences

    ├── Test
        ├── high
            ├── sequences
            ├── sequences_LR
        ├── medium
            ├── sequences
            ├── sequences_LR
        ├── low
            ├── sequences
            ├── sequences_LR
```

## Main Results
<details open>
<summary>

### **Quantitative Results**

</summary>

![results3](./figures/Quantitative%20comparison.png)

</details>

<details open>
<summary>

### **Qualitative Results**

</summary>

![results3](./figures/visual_com.png)

</details>


##  Code Usage
### Environment Setup
* **Dependencies**: 
  - CUDA 11.8
  - Python 3.10.13
  - pytorch 2.1.1
  
  Create Conda Environment
  ```
  conda create -y -n MambaOVSR python=3.10.13
  conda activate MambaOVSR
  pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
  pip install causal_conv1d==1.1.1
  pip install mamba-ssm==1.2.0.post1
  # copy mamba-ssm dir in vim to conda env site-package dir
  cp -rf mamba-1p1p1/mamba_ssm /opt/miniconda3/envs/mamba/lib/python3.10/site-packages
  ```
  
### Train
* Configure the dataset path in train.py
  ```python
  dataset_indexes=["Path to your dataset index file (.txt)",]
* Configure model parameters and training settings in train.py
  ```python
  model_args = model_arg_mamba(...)
  train_args = train_arg(...)
* running code
  ```python
  python train.py

### Test
  * Download the [pre-trained model](https://pan.baidu.com/s/1J33-eH0rLil2TMM3WkD3zw) Code[9527]
    <!-- Pre-training models will be disclosed later -->
  * Set the pre-trained model path in test.py
    ```python
    test_args = test_arg(...,checkpoint='Save path for pre-trained models',...)
  * Set the test set mode and index file path in test.py
    ```python
    mode="medium" # high,medium,low
    dataset_indexes=["Path to your test dataset index file (.txt)",]
  * running code
    ```python
    python test.py 

## Citation
If you find our work useful for your research, please cite our paper
```
@article{chang2025mambaovsr,
  title={MambaOVSR: Multiscale Fusion with Global Motion Modeling for Chinese Opera Video Super-Resolution},
  author={Chang, Hua and Xu, Xin and Liu, Wei and Wang, Wei and Yuan, Xin and Jiang, Kui},
  journal={arXiv preprint arXiv:2511.06172},
  year={2025}
}
```

## Acknowledgement

Our code is built upon [Cycmunet+](https://github.com/tongyuantongyu/cycmunet). Thanks to the contributors for their great work.
  
  

  
