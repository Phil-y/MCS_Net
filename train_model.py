import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
from Load_Dataset import RandomGenerator,ValGenerator,ImageToImage2D

from networks.ACC_UNet import ACC_UNet
from networks.attention_unet import AttUNet
# from networks.AttSwinUnet.attention_swin_unet import AttentionSwinUnet
# from networks.axial_deeplab.axial_deeplab import axial26s,axial50s,axial50l,axial50m
# from networks.biformer.biformer import biformer_tiny,biformer_base,biformer_small
from networks.BPATUNet.BPATUNet import BPATUNet
from networks.BiONet import BiONet
# from networks.cait import cait_M36
from networks.SETR import Setr_cait
# from networks.CCNet.CCNet import Seg_Model
from networks.CE_Net.CE_Net import CE_Net
from networks.SETR import Setr_ConvFormer
# from networks.coplenet import COPLENet
from networks.TrfeNet.cpfnet import CPFNet
# from networks.CPFNet.CPFNet import CPFNet
# from networks.CrossVit.crossvit import crossvit_9_224
# from networks.CSNet import CSNet
from networks.DAEFormer.DAEFormer import DAEFormer
# from networks.DAF3D.DAF3D import DAF3D
from networks.DATransUNet import DATransUNet
from networks.DATransUNet import DATransUNet_CONFIGS as CONFIGS_ViT_seg
from networks.deeplabv3.deeplabv3 import DeepLabV3
from networks.deeplabv3_plus_MobileNetV2.deeplabv3_plus import deeplabv3_plus
# from networks.deeplabv3_plus_ResNet.deeplab_v3_plus import Deeplabv3plus
# from networks.DeepPyramid.DeepPyramid_VGG16 import DeepPyramid_VGG16
from networks.DAELKAFormer.DAELKAFormer import DAELKAFormer
# from networks.deit import deit_base_patch16_224
# from networks.dvit.deep_vision_transformer import deepvit_S
from networks.SETR import Setr_deepvit
# from networks.doubleunet import doubleunet
# from networks.DS_TransUNet import DS_TransUNet
from networks.DeformUNet import DUNet
# from networks.EANet import EANet
# from networks.FastICENet.FastICENet import EDANet
from networks.EGE_UNet import EGEUNet
# from networks.FANet.FANet import FANet
from networks.FastICENet.FastICENet import FastICENet
from networks.FAT_Net import FAT_Net
# from networks.FLA_Net.model import SPNet
# from networks.focal_transformer import FocalTransformer
# from networks.GT_UNet import GT_UNet
# from networks.HDenseUnet import DenseUnet
# from networks.HiFormer.HiFormer import HiFormer
# from networks.IB_UNet.ib_unet import IB_UNet_k3
# from networks.ITUNet import itunet_2d
# from networks.LambdaUNet.lit_model import LitLambdaUnet
# from networks.linformer import Linformer
# from networks.LocalViT.localvit import localvit_tiny_mlp6_act1
from networks.M2UNet import m2unet
from networks.MALUNet import MALUNet
# from networks.maxim.maxim_torch import MAXIM_dns_3s
from networks.MEWUNet import MEWUNet
# from networks.medical_T.model_codes import mix_net
# from networks.poolformer.metaformer import metaformer_id_s12
from networks.MHA_UNet import MHA_UNet
from networks.MISSFormer import MISSFormer
# from networks.mmformer.mmformer import mmformer
from networks.MTUNet import MTUNet
from networks.TrfeNet.mtnet import MTNet
from networks.MResUNet import MultiResUnet
# from networks.nestFormer.nested_former import NestedFormer
# from networks.nextvit.nextvit import NextViT
# from networks.nnFormer.nnFormer import nnFormer
# from networks.PCAMNet import PCAMNet
# from networks.PointUnet import PointUnet
# from networks.poolformer.poolformer import poolformer_s12
from networks.FLA_Net.PSPNet import PNSNet
# from networks.pvt import pvt_tiny
from networks.R2UNet import R2AttUNet
from networks.SETR import Setr_refiner
# from networks.Refiner_vit.refined_transformer import Refiner_ViT_S
from networks.ResUNet import ResUNet
from networks.resunetplusplus import resunetplusplus
# from networks.SAR_UNet import SAR_UNet
from networks.ScaleFormer import ScaleFormer
from networks.TrfeNet.segnet import SegNet
from networks.DAELKAFormer.segformer import SegFormer
from networks.SETR import Setr
from networks.TrfeNet.sgunet import SGUNet
# from networks.SMGARN.smgarn import SMGARN
# from networks.spanet import spanet_small
from networks.SwinUnet import SwinUnet
from networks.SMESwinUnet.SMESwinUnet import SMESwinUnet
# from networks.CrossVit.t2t import T2T
# from networks.TMUNet.TransMUNet import TransMUNet
# from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from networks.TransCeption.Transception import Transception
# from networks.transdeeplab.swin_deeplab import SwinDeepLab
# from networks.Transfuse.TransFuse import TransFuse_S
# from networks.Transnorm.vit_seg_modeling import VisionTransformer
from networks.TransUNet import TransUNet
from networks.TransUNet import TransUNet_CONFIGS as CONFIGS_ViT_seg
from networks.TrfeNet.trfe import TRFENet
from networks.TrfeNet.trfeplus import TRFEPLUS
from networks.u2net import U2NET
from networks.UCTransNet.UCTransNet import UCTransNet
from networks.UDTransNet.UDTransNet import UDTransNet
# from networks.UFormer import Uformer
from networks.U_Transformer import U_Transformer
from networks.UNet import UNet
from networks.UNet_2Plus import UNet_2Plus
from networks.UNet_3Plus import UNet_3Plus
from networks.UNeXt.UNeXt import UNext
# from networks.UTNet.resnet_utnet import ResNet_UTNet

from networks.MCS_Net import MCS_Net
from networks.TDS_Net import TDS_Net
from networks.DHR_Net import DHR_Net
from networks.DAC_Net import DAC_Net


from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch
import Config as config
from torchvision import transforms
from Utils import CosineAnnealingWarmRestarts, WeightedDiceBCE


import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--model_name', type=str,
                    default='hiformer-b', help='[hiformer-s, hiformer-b, hiformer-l]')
args = parser.parse_args()


'''/这段代码定义了一个名为logger_config的函数，用于配置和设置日志记录器（logger）'''
# 定义一个名为logger_config的函数，接收一个参数log_path，该参数表示日志文件的路径。函数返回配置好的日志记录器
def logger_config(log_path):
    '''
    config logger
    :param log_path: log file path
    :return: config logger
    '''
    # 获取或创建一个根日志记录器
    loggerr = logging.getLogger()
    # 设置日志记录器的日志级别为INFO，这意味着只有INFO级别及以上的日志消息会被记录
    loggerr.setLevel(level=logging.INFO)
    # 创建一个文件处理器，用于将日志消息写入到指定的日志文件log_path中，并设置文件编码为UTF-8
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    # 设置文件处理器的日志级别为INFO
    handler.setLevel(logging.INFO)
    # 创建一个日志格式化器，该格式化器定义了日志消息的输出格式，这里的格式仅包含日志消息内容。
    formatter = logging.Formatter('%(message)s')
    # 将上面定义的日志格式化器应用到文件处理器中。
    handler.setFormatter(formatter)
    # 创建一个流处理器，用于将日志消息输出到控制台
    console = logging.StreamHandler()
    # 设置流处理器的日志级别为INFO
    console.setLevel(logging.INFO)
    # 将文件处理器和流处理器添加到日志记录器中，这样日志消息既会被写入到文件中，也会输出到控制台
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    # 返回配置好的日志记录器loggerr。
    return loggerr


'''
这段代码定义了一个名为save_checkpoint的函数，用于保存模型检查点'''

# 定义一个名为save_checkpoint的函数，该函数接收两个参数：
# state：包含模型状态信息的字典。
# save_path：保存模型检查点的路径
def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    # 使用logger记录器输出日志信息，表示正在保存模型到指定路径save_path
    logger.info('\t Saving to {}'.format(save_path))
    # 检查指定的保存路径save_path是否存在，如果不存在，则创建该路径
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # 从state字典中获取模型的当前训练轮次（epoch）、是否为最佳模型（best_model）以及模型类型（model）
    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type
    # 根据best_model的值确定保存的模型文件名。如果当前模型是最佳模型，则文件名格式为best_model-模型类型.pth.tar；否则文件名格式为model-模型类型-轮次.pth.tar。
    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    # 使用PyTorch的torch.save函数将模型状态state保存到上面确定的文件名filename中
    torch.save(state, filename)

'''这段代码定义了一个名为worker_init_fn的函数，该函数用作PyTorch的DataLoader中的worker_init_fn参数，用于初始化每个数据加载进程的随机数生成器。'''
# 定义一个名为worker_init_fn的函数，该函数接收一个参数worker_id，该参数代表当前数据加载进程的ID
def worker_init_fn(worker_id):
    # 使用Python的random模块设置随机数生成器的种子。
    # 种子值是config.seed（这里假设config是一个配置对象或字典，包含了一个名为seed的种子值）加上当前worker_id。
    # 这样做的目的是为了确保每个数据加载进程有一个独特的随机数种子，从而在并行处理数据时可以保证随机性的一致性。
    random.seed(config.seed + worker_id)
'''这段代码的主要功能是为每个数据加载进程设置一个独特的随机数种子，以确保在并行处理数据时的随机性一致性'''

##################################################################################
#=================================================================================
#          Main Loop: load model,
#=================================================================================
##################################################################################
'''
定义了一个名为main_loop的函数，该函数是程序的主要执行逻辑
'''
# 定义一个名为main_loop的函数，该函数有三个参数：
# batch_size：批次大小，默认值为config.batch_size。
# model_type：模型类型，默认值为config.model_name。
# tensorboard：是否使用TensorBoard，默认为True
def main_loop(batch_size=config.batch_size, model_type=config.model_name, tensorboard=True):
    # 定义一个训练数据集的图像变换，使用RandomGenerator，它会对图像进行随机旋转和翻转
    # Load train and val data
    train_tf= transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    # print(train_tf)
    # 定义一个验证数据集的图像变换，使用ValGenerator，该变换可能会对图像进行缩放或其他操作
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    # 创建训练数据集，使用ImageToImage2D类，同时应用上面定义的训练数据集图像变换
    train_dataset = ImageToImage2D(config.train_dataset, train_tf,image_size=config.img_size)
    # print(train_dataset)#<Load_Dataset.ImageToImage2D object at 0x0000026A6A2D4BE0>
    # print(len(train_dataset))#24
    # 创建验证数据集，使用ImageToImage2D类，同时应用上面定义的验证数据集图像变换
    val_dataset = ImageToImage2D(config.val_dataset, val_tf,image_size=config.img_size)
    # print(val_dataset) #<Load_Dataset.ImageToImage2D object at 0x0000026A6A2D4C10>
    # 创建训练数据加载器，使用DataLoader类，设置批次大小、数据是否随机打乱、并行处理的工作进程数量、以及是否将数据加载到GPU的固定内存中
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)
    # 创建验证数据加载器，与上面的训练数据加载器设置相似
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)
    # print(train_loader) #<torch.utils.data.dataloader.DataLoader object at 0x0000011AAF7E4C10>
    # 获取学习率，从config配置中获取
    lr = config.learning_rate
    # 使用logger记录器输出模型类型
    logger.info(model_type)



    if  model_type == 'ACC_UNet':
        model = ACC_UNet()

    elif model_type == "AttUNet":
        model = AttUNet()

    # elif model_type == "AttentionSwinUnet":
    #     from types import SimpleNamespace
    #     configs = SimpleNamespace(config)
    #     model = AttentionSwinUnet(configs,num_classes=1)

    # elif model_type == "axial_deeplab":
    #     model = axial26s()

    elif model_type == "BPATUNet":
        model = BPATUNet()


    # elif model_type == "BiFormer":
    #     model = biformer_tiny()

    elif model_type == "BiONet":
        model = BiONet()

    # elif model_type == "cait":
    #     model = cait_M36()

    elif model_type == "SETR_cait":
        model = Setr_cait()

    # elif model_type == "CC_Net":
    #     model = Seg_Model()

    elif model_type == "CE_Net":
        model = CE_Net()

    elif model_type == "SETR_ConvFormer": #img_size = 256
        model = Setr_ConvFormer(n_channels=config.n_channels, n_classes=config.n_labels,imgsize=config.img_size)

    # elif model_type == "coplenet":
    #     model = COPLENet()

    elif model_type == "CPFNet":
        model = CPFNet()

    # elif model_type == "crossvit":
    #     model = crossvit_9_224()

    # elif model_type == "CSNet":
    #     model = CSNet()

    elif model_type == "DAEFormer":
        model = DAEFormer()

    # elif model_type == "DAF3D":
    #     model = DAF3D()

    elif model_type == "DATransUNet":
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = DATransUNet(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)

    elif model_type == "deeplabv3":
        model = DeepLabV3()

    elif model_type == "deeplabv3_plus_MobileNetV2":
        model = deeplabv3_plus(num_classes=config.n_labels)

    # elif model_type == "deeplabv3_plus_ResNet":
    #     model = Deeplabv3plus(3, 1, 32, 'resnet101')

    # elif model_type == "DeepPyramid_ResNet50":
    #     model = DeepPyramid_ResNet50()

    # elif model_type == "DeepPyramid_VGG16":
    #     model = DeepPyramid_VGG16(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == "DAELKAFormer":
        model = DAELKAFormer()

    # elif model_type == "deit":
    #     model = deit_base_patch16_224()

    elif model_type == "SETR_deepvit":
        model = Setr_deepvit(n_channels=config.n_channels, n_classes=config.n_labels, imgsize=config.img_size)

    # elif model_type == "dvit":
    #     model = deepvit_S()

    # elif model_type == "doubleunet":
    #     model = doubleunet()

    # elif model_type == "DS_TransUNet":
    #     model = DS_TransUNet(128,1)

    elif model_type == "DUNet":
        model = DUNet()

    # elif model_type == "EANet":
    #     model = EANet()

    # elif model_type == "EDANet":
    #     model = EDANet()

    elif model_type == "EGE_UNet":
        model = EGEUNet()

    # elif model_type == "FANet":
    #     model = FANet()

    elif model_type == "FastICENet":
        model = FastICENet(1)

    elif model_type == "FAT_Net":
        model = FAT_Net(n_channels=config.n_channels, n_classes=config.n_labels)

    # elif model_type == "FLA_Net":
    #     model = SPNet()

    # elif model_type == "FocalTransformer":
    #     model = FocalTransformer()

    # elif model_type == "GT_UNet":
    #     model = GT_UNet(3, 9)

    # elif model_type == "H-DenseUnet":
    #     model = DenseUnet()

    # elif model_type == "HiFormer":
    #     import networks.HiFormer.HiFormer_configs as configs
    #     CONFIGS = {
    #         'hiformer-s': configs.get_hiformer_s_configs(),
    #         'hiformer-b': configs.get_hiformer_b_configs(),
    #         'hiformer-l': configs.get_hiformer_l_configs(),
    #     }
    #     model = HiFormer(configs=CONFIGS[args.model_name], img_size=224, n_classes=1)

    # elif model_type == "IB_UNet":
    #     model = IB_UNet_k3()

    # elif model_type == "ITUNet":
    #     model = itunet_2d()

    # elif model_type == "LambdaUNet":
    #     model = LitLambdaUnet()

    # elif model_type == "LocalVit":
    #     model = localvit_tiny_mlp6_act1()

    # elif model_type == "linformer":
    #     model = Linformer()

    elif model_type == "M2UNet":
        model = m2unet()

    elif model_type == "MALUNet":
        model = MALUNet()

    # elif model_type == "maxim":
    #     model = MAXIM_dns_3s()

    elif model_type == "MEWUNet":
        model = MEWUNet()

    # elif model_type == "Medical_Transformer":
    #     model = mix_net()

    # elif model_type == "metaformer":
    #     model = metaformer_id_s12()

    elif model_type == 'MHA_UNet':
        model = MHA_UNet()

    elif model_type == "MISSFormer":
        model = MISSFormer()

    # elif model_type == "mmformer":
    #     model = mmformer()

    elif model_type == "MTUNet":
        model = MTUNet()

    elif model_type == "MTNet":
        model = MTNet(in_ch=3, out_ch=1)

    elif model_type == 'MultiResUnet':
        model = MultiResUnet(n_channels=config.n_channels, n_classes=config.n_labels)

    # elif model_type == "NestedFormer":
    #     model = NestedFormer()

    # elif model_type == "NextViT":
    #     model = NextViT()

    # elif model_type == "nnFormer":
    #     model = nnFormer()

    # elif model_type == "PCAMNet":
    #     model = PCAMNet()

    # elif model_type == "PointUnet":
    #     model = PointUnet()

    # elif model_type == "poolformer":
    #     model = poolformer_s12()

    elif model_type == "PSPNet":
        model = PNSNet()

    # elif model_type == "pvt":
    #     model = pvt_tiny()v

    elif model_type == "R2AttUNet":
        model = R2AttUNet()

    elif model_type == "SETR_refiner":
        model = Setr_refiner(n_channels=config.n_channels, n_classes=config.n_labels, imgsize=config.img_size)

    # elif model_type == "Refiner_vit":
    #     model = Refiner_ViT_S()

    elif model_type == "ResUNet":
        model = ResUNet()

    elif model_type == "resunetplusplus":
        model = resunetplusplus()

    # elif model_type == "SAR_UNet":
    #     model = SAR_UNet()

    elif model_type == "ScaleFormer":
        model = ScaleFormer()

    elif model_type == "SETR":

        model = Setr(n_channels=config.n_channels, n_classes=config.n_labels, imgsize=config.img_size)

    elif model_type == "SegNet":
        model = SegNet()

    # elif model_type == "SegFormer":
    #     model = SegFormer()

    elif model_type == "SGUNet":
        model = SGUNet()

    # elif model_type == "SMGARN":
    #     model = SMGARN()

    # elif model_type == "spanet":
    #     model = spanet_small()

    # elif model_type == "spnet":
    #     model = SPNet()

    elif model_type == 'SwinUnet':
        model = SwinUnet()
        model.load_from()
        lr = 5e-4

    elif model_type == 'SMESwinUnet':
        model = SMESwinUnet(n_channels=config.n_channels,n_classes=config.n_labels)
        model.load_from()
        lr = 5e-4

    # elif model_type == "t2t":
    #     model = T2T()

    # elif model_type == "TMUNet":
    #     model = TransMUNet()

    # elif model_type == "TransBTS":
    #     model = TransBTS()

    elif model_type == "TransCeption":
        model = Transception()

    # elif model_type == "transdeeplab":
    #     model = SwinDeepLab(
    #     config.EncoderConfig,
    #     config.ASPPConfig,
    #     config.DecoderConfig)

    # elif model_type == "TransFuse":
    #     model = TransFuse_S()

    # elif model_type == "TransNorm":
    #     config_vit = CONFIGS_ViT_seg[args.vit_name]
    #     config_vit.n_classes = args.num_classes
    #     config_vit.n_skip = args.n_skip
    #     if args.vit_name.find('R50') != -1:
    #         config_vit.patches.grid = (
    #             int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    #     model = VisionTransformer(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)

    elif model_type == "TransUNet":
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = TransUNet(config_vit,img_size=args.img_size, num_classes=config_vit.n_classes)

    elif model_type == "TRFENet":
        model = TRFENet(in_ch=3, out_ch=1)
    elif model_type == "TRFEPLUS":
        model = TRFEPLUS(in_ch=3, out_ch=1)

    elif model_type == "U2NET":
        model = U2NET()

    elif model_type == 'UCTransNet':
        config_vit = config.get_CTranS_config()
        model = UCTransNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)

    elif model_type == 'UDTransNet':
        config_vit = config.get_DTranS_config()
        model = UDTransNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)

    # elif model_type == "Uformer":
    #     model = Uformer()

    elif model_type == "U_Transformer":
        model = U_Transformer()

    elif model_type == 'UNet':
        model = UNet(n_channels=config.n_channels,n_classes=config.n_labels)

    elif model_type == 'UNet_2Plus':
        model = UNet_2Plus(n_channels=config.n_channels,n_classes=config.n_labels)

    elif model_type == 'UNet_3Plus':
        model = UNet_3Plus(n_channels=config.n_channels,n_classes=config.n_labels)
    elif model_type == "UNext":
        model = UNext()

    # elif model_type == "UTNet":
    #     model = ResNet_UTNet()

    elif model_type == 'DAC_Net':
        model = DAC_Net()
    elif model_type == 'TDS_Net':
        model = TDS_Net()
    elif model_type == 'DHR_Net':
        model = DHR_Net()
    elif model_type == 'MCS_Net':
        model = MCS_Net()

    else: raise TypeError('Please enter a valid name for the model type')

    torch.cuda.set_device(device=0)
    # 将模型移动到GPU上进行计算
    model = model.cuda()
    # print("model:",model)

    # 定义损失函数，使用WeightedDiceBCE，它是Dice损失和二元交叉熵损失的加权组合
    criterion = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5)
    # print(criterion)

    # 定义优化器，使用AdamW优化器，并只优化需要梯度的参数
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 根据配置决定是否使用余弦退火学习率调度器
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None

    # 如果tensorboard为True，则配置TensorBoard，创建日志目录并初始化TensorBoard写入器
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None
    # 初始化最佳Dice系数和最佳轮次
    max_dice = 0.0
    best_epoch = 1
    # 开始训练循环，遍历所有轮次
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        # 使用logger记录器输出当前轮次
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one
        # epoch
        # 设置模型为训练模式
        model.train(True)

        logger.info('Training with batch size : {}'.format(batch_size))
        # 训练一个轮次的模型
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)
        # 评估验证集上的模型性能
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                            optimizer, writer, epoch, lr_scheduler,model_type,logger)
            # print(val_loss, val_dice)
        # =============================================================
        #       Save best model
        # =============================================================
        # 保存性能最佳的模型。
        if val_dice > max_dice:
            if epoch+1 > 5:
                logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice,max_dice, best_epoch))
        # 计算早停次数
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count,config.early_stopping_patience))
        # 如果早停次数超过预设值，则提前停止训练
        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break
    # 返回训练好的模型
    return model
'''
这个main_loop函数的主要功能是加载数据、定义模型、损失函数、优化器和学习率调度器，然后进行多轮次的训练和验证，最后保存性能最佳的模型
'''

'''
这段代码是程序的主入口，当直接运行此Python文件时，if __name__ == '__main__': 下的代码块将被执行
'''
if __name__ == '__main__':
    # 这是Python中的一个常见惯用法，它表示以下的代码块只有在直接运行此Python脚本时才会执行，而不是作为模块被导入时执行
    deterministic = True
    # 根据deterministic的值，设置cudnn的行为。
    # 如果deterministic为False，则使用cudnn.benchmark = True来提高训练速度，但结果可能不是完全确定性的。如果deterministic为True，则强制cudnn为确定性模式。
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # 设置随机种子以确保结果的可重复性。这样做是为了在每次运行时得到相同的随机结果，方便调试和比较不同模型的性能
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)
    # 配置日志记录器，并将其赋值给logger变量。logger_config是之前定义的一个函数，用于设置和配置日志记录器
    logger = logger_config(log_path=config.logger_path)
    # 调用main_loop函数进行模型训练。model_type和tensorboard是函数的参数，这里使用config对象中的值。
    model = main_loop(model_type=config.model_name, tensorboard=True)
'''总结：这段代码主要是对训练过程进行初始化和配置，包括设置随机种子、cudnn行为、创建日志记录器、创建模型保存路径，并调用main_loop函数开始模型训练。'''