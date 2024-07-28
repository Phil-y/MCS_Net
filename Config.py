import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

n_filts = 16
cosineLR = True # whether use cosineLR or not
n_channels = 3
n_labels = 1





print_frequency = 1
save_frequency = 5000
vis_frequency = 10
epochs = 300
early_stopping_patience = 50


pretrain = False







# model_name = 'ACC_UNet'
# model_name = 'AttUNet'
# model_name = 'AttentionSwinUnet'
# model_name = 'axial_deeplab'
# model_name = 'BPATUNet'
# model_name = 'BiFormer'
# model_name = 'BiONet'
# model_name = 'cait'
# model_name = 'SETR_cait'
# model_name = 'CC_Net'
# model_name = 'CE_Net'
# model_name = 'SETR_ConvFormer'
# model_name = 'coplenet'
# model_name = 'CPFNet'
# model_name = 'crossvit'
# model_name = 'CSNet'
# model_name = 'DAEFormer'
# model_name = 'DAF3D'
# model_name = 'DATransUNet'
# model_name = 'deeplabv3'
# model_name = 'deeplabv3_plus_MobileNetV2'
# model_name = 'deeplabv3_plus_ResNet'
# model_name = 'DeepPyramid_ResNet50'
# model_name = 'DeepPyramid_VGG16'
# model_name = 'DAELKAFormer'
# model_name = 'deit'
# model_name = 'SETR_deepvit'
# model_name = 'doubleunet'
# model_name = "DS_TransUNet"
# model_name = 'dvit'
# model_name = 'DUNet'
# model_name = 'EANet'
# model_name = 'EDANet'
# model_name = 'EGE_UNet'
# model_name = 'FANet'
# model_name = 'FastICENet'
# model_name = 'FAT_Net'
# model_name = 'FLA_Net'
# model_name = 'FocalTransformer'
# model_name = 'GT_UNet'
# model_name = 'H-DenseUnet'
# model_name = 'HiFormer'
# model_name = 'IB_UNet'
# model_name = 'ITUNet'
# model_name = 'LambdaUNet'
# model_name = 'linformer'
# model_name = 'LocalVit'
# model_name = 'M2UNet'
# model_name = 'MALUNet'
# model_name = 'maxim'
# model_name = 'MEWUNet'
# model_name = 'Medical_Transformer'
# model_name = "metaformer"
# model_name = 'MHA_UNet'
# model_name = 'MISSFormer'
# model_name = 'mmformer'
# model_name = 'MTUNet'
# model_name = 'MTNet'
# model_name = 'MultiResUnet'
# model_name = 'NestedFormer'
# model_name = 'NextViT'
# model_name = 'nnFormer'
# model_name = 'PCAMNet'
# model_name = 'PointUnet'
# model_name = "poolformer"
# model_name = 'PSPNet'
# model_name = 'pvt'
# model_name = 'R2AttUNet'
# model_name = 'SETR_refiner'
# model_name = 'Refiner_vit'
# model_name = 'ResUNet'
# model_name = 'resunetplusplus'
# model_name = 'SAR_UNet'
# model_name = 'ScaleFormer'
# model_name = 'SETR'
# model_name = 'SegNet'
# model_name = 'SegFormer'
# model_name = 'SGUNet'
# model_name = 'SMGARN'
# model_name = 'spanet'
# model_name = 'spnet'
# model_name = 'SwinUnet'
# model_name = 'SMESwinUnet'
# model_name = 't2t'
# model_name = 'TMUNet'
# model_name = 'TransBTS'
# model_name = 'TransCeption'
# model_name = 'transdeeplab'
# model_name = 'TransFuse'
# model_name = 'TransNorm'
# model_name = 'TransUNet'
# model_name = 'TRFENet'
# model_name = 'TRFEPLUS'
# model_name = 'U2NET'
# model_name = 'UCTransNet'
# model_name = 'UDTransNet'
# model_name = 'Uformer'
# model_name = 'U_Transformer'
# model_name = 'UNet'
# model_name = 'UNet_3Plus'
# model_name = 'UNet_2Plus'
# model_name = 'UNext'
# model_name = 'UTNet'


task_name = 'BUSI_with_GT'
# task_name = 'CHASE'
# task_name = 'Chest Xray'
# task_name = 'CVC-ClinicDB'
# task_name = 'ISIC2017'
# task_name = 'ISIC2018'
# task_name = 'Kvasir-Seg'
# task_name = 'RITE'
# task_name = 'MoNuSeg'
# task_name = 'GlaS'
# task_name = 'DDTI'
# task_name = 'TN3K'
# task_name = 'TG3K'

# used in testing phase, copy the session name in training phase
test_session = "Test_session_07.15_15h46"
# test_session = "Test_session"

# model_name = 'ACC_UNet' #2023
# model_name = 'AttUNet' #2018
# model_name = 'BPATUNet' #2023
# model_name = 'BiONet' #2020
# model_name = 'SETR_cait' #2021
# model_name = 'CE_Net' #2019
# model_name = 'SETR_ConvFormer' #2020
# model_name = 'CPFNet' #2020
# model_name = 'DAEFormer' #2023
# model_name = 'DATransUNet' #2023
# model_name = 'deeplabv3'
# model_name = 'deeplabv3_plus_MobileNetV2' #2018
# model_name = 'DAELKAFormer' #2023
# model_name = 'SETR_deepvit' #2021
# model_name = 'DUNet' #2019
# model_name = 'EGE_UNet' #2023
# # model_name = 'FastICENet' #2023
# model_name = 'FAT_Net' #2022
# model_name = 'M2UNet' #2018
# model_name = 'MALUNet' #2022
# model_name = 'MEWUNet' #2022
# model_name = 'MHA_UNet' #2023
# model_name = 'MISSFormer' #2021
# model_name = 'MTUNet' #2022
# model_name = 'MTNet'
# model_name = 'MultiResUnet' #2019
# model_name = 'PSPNet' #2017
# model_name = 'R2AttUNet' #2018
# model_name = 'SETR_refiner' #2021
# model_name = 'ResUNet' #2018
# model_name = 'resunetplusplus' #2021
# model_name = 'ScaleFormer' #2022
# model_name = 'SETR' #2021
# model_name = 'SegNet' #2017
# model_name = 'SGUNet' #2021
# model_name = 'SwinUnet' #2021
# model_name = 'SMESwinUnet' #2022
# model_name = 'TransCeption' #2023
# model_name = 'TransNorm' #2022
# model_name = 'TransUNet' #2021
# model_name = 'TRFENet' #2023
# model_name = 'TRFEPLUS' #2023
# model_name = 'U2NET' #2022
# model_name = 'UCTransNet' #2022
# model_name = 'UDTransNet' #2023
# model_name = 'U_Transformer' #2021
# model_name = 'UNet' #2015
# model_name = 'UNet_2Plus' #2018
# model_name = 'UNet_3Plus' #2020
# model_name = 'UNext' #2023


model_name = 'TDS_Net'
# model_name = 'DHR_Net'
# model_name = 'MCS_Net'
# model_name = 'DAC_Net'


# learning_rate = 1e-3
learning_rate = 1e-2
# learning_rate = 1e-4

# batch_size = 2
batch_size = 4
# batch_size = 8
# batch_size = 16

# img_size = 128
# img_size = 224
img_size = 256
# img_size = 512

# optimizer = 'AdamW'
# optimizer = 'Adam'
optimizer = 'SGD'

# channel_list = 8,16,24,32,48,64
# channel_list = 8,16,32,48,64,96
# channel_list = 16,24,32,48,64,128
# channel_list = 8,16,32,64,128,160
# channel_list = 16,32,48,64,128,256
# channel_list = 16,32,64,128,160,256
# channel_list = 16,32,64,128,256,512
# depth = 1,1,1,1
# depth = 1,1,2,2
# depth = 1,2,2,4
# depth = 2,2,4,4

train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
val_dataset = './datasets/'+ task_name+ '/Val_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'

session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')

# save_path          ="./data_train_test_session/" + task_name +'/'+ model_name +'/' \
#                     + 'channel_list_' + str(channel_list) + '/' \
#                     + 'lr_'+ str(learning_rate) +' batchsize_'+ str(batch_size) + ' ImgSize_'+ str(img_size) + ' ' + optimizer + '/' \
#                     + session_name + '/'

# save_path          ="./data_train_test_session/" + task_name +'/'+ model_name +'/' \
#                     + 'channel_list_' + str(channel_list) + '/' \
#                     + 'lr_'+ str(learning_rate) +' batchsize_'+ str(batch_size) + ' ImgSize_'+ str(img_size) + optimizer + '/' \
#                     + session_name + '/'

# save_path          ="./data_train_test_session/" + task_name +'/'+ model_name +'/' \
#                     + 'channel_list_' + str(channel_list) + '/' \
#                     + 'lr_'+ str(learning_rate) +'_batchsize_'+ str(batch_size) + '_ImgSize_'+ str(img_size) + '/' \
#                     + session_name + '/'

# save_path          ="./data_train_test_session/" + task_name +'/'+ model_name +'/' \
#                     + 'channel_list_' + str(channel_list) + '/' \
#                     + 'depth_' + str(depth) + '/' \
#                     + session_name + '/'

save_path          ="./data_train_test_session/" + task_name +'/'+ model_name +'/' + session_name + '/'


model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'








##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config




##########################################################################
# DTrans configs
##########################################################################
def get_DTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 2
    config.transformer.embedding_channels = 32 * config.transformer.num_heads
    config.KV_size = config.transformer.embedding_channels * 4
    config.KV_size_S = config.transformer.embedding_channels
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_sizes=[16,8,4,2]
    config.base_channel = 32
    config.decoder_channels = [32,64,128,256,512]
    return config