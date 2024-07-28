# MCS_Net

This repository is the official implementation of MCS_Net : All-round enhancements! A Light-weight UNet for Thyroid Nodule Segmentation in ultrasound images using PyTorch.

![MCS-Net](C:\Users\Administrator\Desktop\科研\MCS_Net\MCS-Net\MCS-Net.png)

## Main Environments

- python 3.9
- pytorch 2.1.0
- torchvision 0.16.0

## Requirements

Install from the `requirements.txt` using:

```
pip install -r requirements.txt
```

## Prepare the dataset.

- The DDTI and TN3K datasets, can be found here ([GoogleDrive](https://drive.google.com/drive/folders/1za9f38XKx-VYPxxb_xx83Dpk-Wg3Yaw8?usp=drive_link)), The GlaS  datasets, can be found here ([GoogleDrive](https://drive.google.com/drive/folders/1bfs6bgVM24fqyjO4aoX7ENi-1xKtNBGc?usp=drive_link)), The RITE datasets, can be found here ([GoogleDrive](https://drive.google.com/drive/folders/1Vofe2TSVry0FZYLNisvPKvR_67aSj0ml?usp=drive_link)), divided into a 7:1:2 ratio, 


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
    └── RITE
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

First, modify the model, dataset and training hyperparameters in `Config.py`

Then simply run the training code.

```
python3 train_model.py
```

## Evaluate the Model

Please make sure the right model and dataset is selected in `Config.py`

Then simply run the evaluation code.

```
python3 test_model.py
```




## Citation

If you find this work useful in your research or use this dataset in your work, please consider citing the following papers:
