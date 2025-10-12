# MCS_Net

This repository is the official implementation of MCS_Net : All-round enhancements! A Light-weight UNet for Thyroid Nodule Segmentation in ultrasound images using PyTorch.

![MCS-Net](Fig/MCS-Net.png)



## Main Environments

- python 3.9
- pytorch 2.1.0
- torchvision 0.16.0
- For generating GradCAM results, please follow the code on this [repository](https://github.com/jacobgil/pytorch-grad-cam)


## Requirements

Install from the `requirements.txt` using:

```
pip install -r requirements.txt
```



## Prepare the dataset.




- Then prepare the datasets in the following format for easy use of the code:

```
├── datasets
    ├── DDTI
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    ├── TN3k
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    ├── GlaS
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    └── Chest X-ray
        ├── Test_Folder
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── img
            └── labelcol 
         
```



## Train the Model

First, modify the model, dataset and training hyperparameters (including learning rate, batch size img size and optimizer etc) in `Config.py`

Then simply run the training code.

```
python3 train_model.py
```



#### 2. Test the Model

Please make sure the right model, dataset and hyperparameters setting  is selected in `Config.py`. 

Then change the test_session in `Config.py` .

Then simply run the evaluation code.

```
python3 test_model.py
```



## Reference
- [UNet](https://github.com/ZJUGiveLab/UNet-Version)
- [UNet++](https://github.com/ZJUGiveLab/UNet-Version)
- [UNet3+](https://github.com/ZJUGiveLab/UNet-Version)
- [MultiResUNet](https://github.com/makifozkanoglu/MultiResUNet-PyTorch)
- [TransUNet](https://github.com/Beckschen/TransUNet)
- [UCTransNet](https://github.com/McGregorWwww/UCTransNet)
- [ACC_UNet](https://github.com/qubvel/segmentation_models.pytorch)
- [MEW_UNet](https://github.com/JCruan519/MEW-UNet)
- [MISSFormer](https://github.com/ZhifangDeng/MISSFormer)
- [U2Net](https://github.com/NathanUA/U-2-Net)



