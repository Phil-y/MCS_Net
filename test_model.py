import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

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
from networks.Transnorm.vit_seg_modeling import VisionTransformer
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



from Utils import *
import cv2

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
args = parser.parse_args()
def show_image_with_dice(predict_save, labs, save_path):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    # fig, ax = plt.subplots()
    # plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))
    if config.task_name == "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save,(448,448))
        predict_save = cv2.resize(predict_save,(224,224))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path,predict_save * 255)
    else:
        cv2.imwrite(save_path,predict_save * 255)
    # plt.imshow(predict_save * 255,cmap='gray')
    # plt.text(x=10, y=24, s="Dice:" + str(dice_show), fontsize=5)
    # plt.axis("off")
    # #remove the white borders
    # height, width = predict_save.shape
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.savefig(save_path, dpi=2000)
    # plt.close()
    return dice_pred, iou_pred

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()

    output = model(input_img.cuda())
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')



    input_img.to('cpu')

    # input_img = input_img[0].transpose(0, -1).cpu().detach().numpy()
    # labs = labs[0]
    # output = output[0, 0, :, :].cpu().detach().numpy()

    # if (True):
    #     pickle.dump({
    #         'input': input_img,
    #         'output': (output >= 0.5) * 1.0,
    #         'ground_truth': labs,
    #         'dice': dice_pred_tmp,
    #         'iou': iou_tmp
    #     },
    #         open(vis_save_path + '.pkl', 'wb'))

    # if (True):
    #     plt.figure(figsize=(10, 3.3))
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(input_img)
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(labs, cmap='gray')
    #     plt.subplot(1, 3, 3)
    #     plt.imshow((output >= 0.5) * 1.0, cmap='gray')
    #     plt.suptitle(f'Dice score : {np.round(dice_pred_tmp, 3)}\nIoU : {np.round(iou_tmp, 3)}')
    #     plt.tight_layout()
    #     plt.savefig(vis_save_path)
    #     plt.close()


    return dice_pred_tmp, iou_tmp



if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    test_session = config.test_session
    if config.task_name == "BUSI_with_GT":
        test_num = 85
        model_type = config.model_name
        model_path = "./data_train_test_session/BUSI_with_GT/"+model_type + "/" +test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "CHASE":
        test_num = 52
        model_type = config.model_name
        model_path = "./data_train_test_session/CHASE/"+model_type+"/"   + test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "Chest Xray":
        test_num = 144
        model_type = config.model_name
        model_path = "./data_train_test_session/Chest Xray/"+model_type+"/" + test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "CVC-ClinicDB":
        test_num = 122
        model_type = config.model_name
        model_path = "./data_train_test_session/CVC-ClinicDB/" + model_type + "/"  + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "DDTI":
        test_num = 126
        model_type = config.model_name
        model_path = "./data_train_test_session/DDTI/" + model_type + "/"  + test_session + "/models/best_model-" + model_type + ".pth.tar"
        # model_path = "./data_train_test_session/DDTI/" + model_type + "/" + path + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "ISIC2017":
        test_num = 433
        model_type = config.model_name
        model_path = "./data_train_test_session/ISIC2017/" + model_type + "/"  + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "ISIC2018":
        test_num = 535
        model_type = config.model_name
        model_path = "./data_train_test_session/ISIC2018/" + model_type + "/"  + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "Kvasir-Seg":
        test_num = 200
        model_type = config.model_name
        model_path = "./data_train_test_session/Kvasir-Seg/" + model_type + "/"  + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "MoNuSeg":
        test_num = 14
        model_type = config.model_name
        model_path = "./data_train_test_session/MoNuSeg/"+model_type+"/" +test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "TG3K":
        test_num = 708
        model_type = config.model_name
        model_path = "./data_train_test_session/TG3K/"+model_type+"/" +test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "TN3K":
        test_num = 614
        model_type = config.model_name
        model_path = "./data_train_test_session/TN3K/"+model_type+"/" +test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "GlaS":
        test_num = 35
        model_type = config.model_name
        model_path = "./data_train_test_session/GlaS/"+model_type+"/" +test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "RITE":
        test_num = 8
        model_type = config.model_name
        model_path = "./data_train_test_session/RITE/"+model_type+"/" +test_session+"/models/best_model-"+model_type+".pth.tar"

    save_path  = "./data_train_test_session/" + config.task_name +'/'+ model_type +'/'  + test_session + '/'
    vis_path = "./data_train_test_session/" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    fp = open(save_path + 'test_result', 'a')
    fp.write(str(datetime.now()) + '\n')

    if model_type == 'ACC_UNet':
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

    elif model_type == "SETR_ConvFormer":  # img_size = 256
        model = Setr_ConvFormer()

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
        model = deeplabv3_plus()

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
        model = FastICENet()

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
        model = SMESwinUnet(n_channels=config.n_channels, n_classes=config.n_labels)
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

    elif model_type == "TransNorm":
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = VisionTransformer(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)

    elif model_type == "TransUNet":
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = TransUNet(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)

    elif model_type == "TRFENet":
        model = TRFENet(in_ch=3, out_ch=1)
    elif model_type == "TRFEPLUS":
        model = TRFEPLUS(in_ch=3, out_ch=1)

    elif model_type == "U2NET":
        model = U2NET()

    elif model_type == 'UCTransNet':
        config_vit = config.get_CTranS_config()
        model = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UDTransNet':
        config_vit = config.get_DTranS_config()
        model = UDTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

        # elif model_type == "Uformer":
        #     model = Uformer()

    elif model_type == "U_Transformer":
        model = U_Transformer()

    elif model_type == 'UNet':
        model = UNet(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNet_2Plus':
        model = UNet_2Plus(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNet_3Plus':
        model = UNet_3Plus(n_channels=config.n_channels, n_classes=config.n_labels)
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

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0
    acc_pred = 0.0


    import time
    from metrics import Metrics, evaluate

    all_start = time.time()
    metrics = Metrics(['precision', 'Sensitivity', 'specificity', 'accuracy', 'IOUJaccard', 'dice_or_f1', 'MAEMAEMAE', 'auc_roc','f1_score'])
    total_iou = 0
    total_cost_time = 0


    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']

            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path+str(i)+"_lab.jpg", dpi=300)
            plt.close()



            test_data = test_data.cuda()
            test_label = test_label.cuda()

            start = time.time()
            pred = model.forward(test_data)
            cost_time = time.time() - start


            Precision, Recall, Specificity, accuracy, IoU, DICE, MAE, auc_roc, f1_score= evaluate(pred, test_label)
            metrics.update(precision=Precision, Sensitivity=Recall, specificity=Specificity, accuracy=accuracy, IOUJaccard=IoU,
                           dice_or_f1=DICE,  MAEMAEMAE=MAE,  auc_roc=auc_roc, f1_score=f1_score)


            # total_iou += iou
            # total_cost_time += cost_time
            print(config.model_name)
            metrics_result = metrics.mean(test_num)
            metrics_result['inference_time'] = time.time() - all_start
            print("Test Result:")
            print(
                'precision: %.4f, recall_Sensitivity: %.4f, specificity: %.4f, accuracy: %.4f, iou: %.4f, dice: %.4f, mae: %.4f, auc: %.4f, f1_score: %.4f, inference_time: %.4f'
                % (metrics_result['precision'], metrics_result['Sensitivity'], metrics_result['specificity'],
                   metrics_result['accuracy'], metrics_result['IOUJaccard'], metrics_result['dice_or_f1'],
                     metrics_result['MAEMAEMAE'], metrics_result['auc_roc'], metrics_result['f1_score'],metrics_result['inference_time'])
                  )

            # print("total_cost_time:", total_cost_time)
            # print("loop_cost_time:", time.time() - all_start)


            # evaluation_dir = os.path.sep.join([save_path, 'metrics',  '/'])
            # if not os.path.exists(evaluation_dir):
            #     os.makedirs(evaluation_dir)



            keys_txt = ''
            values_txt = ''
            for k, v in metrics_result.items():
                if k != 'mae' and k != 'hd' and k != 'inference_time':
                    v = 100 * v

                # keys_txt += k + '\t'

                keys_txt  +='   ' + k + '\t'
                values_txt += '    %.2f' % v + '\t'+ '\t'

            name = keys_txt + '\n'
            text = values_txt + '\n'
            metrics_path= save_path + '/' + config.model_name + '_metrics' + '.txt'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(metrics_path, 'a+') as f:
                f.write(name)
                f.write(text)
            print(f'metrics saved in {metrics_path}')
            print("------------------------------------------------------------------")







            values_txt = ' '
            for k, v in metrics_result.items():
                if k != 'mae' and k != 'hd' and k != 'inference_time':
                    v = 100 * v

                # keys_txt += '   ' + k + '\t'
                values_txt += '%.2f' % v + '\t'

            # name = keys_txt + '\n'
            text = values_txt + '\n'
            metrics_path = save_path + '/' + config.model_name + '_metrics_draw' + '.txt'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(metrics_path, 'a+') as f:
                # f.write(name)
                f.write(text)
            print(f'metrics saved in {metrics_path}')
            print("------------------------------------------------------------------")






            input_img = torch.from_numpy(arr)
            dice_pred_t,iou_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                          vis_path+str(i),
                                               dice_pred=dice_pred, dice_ens=dice_ens)
            dice_pred+=dice_pred_t
            iou_pred+=iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print ("dice_pred",dice_pred/test_num)
    print ("iou_pred",iou_pred/test_num)

    fp.write(f"dice_pred : {dice_pred/test_num}\n")
    fp.write(f"iou_pred : {iou_pred/test_num}\n")
    fp.close()



