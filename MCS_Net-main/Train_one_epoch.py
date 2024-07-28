# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 2:14 下午
# @Author  : Haonan Wang
# @File    : Train_one_epoch.py
# @Software: PyCharm
import torch.optim
import os
import time
from Utils import *
import Config as config
import warnings
warnings.filterwarnings("ignore")


'''这是一个用于打印训练或测试摘要信息的函数print_summary。'''
def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    # 创建摘要信息的标题，包括模式（训练或测试）、当前轮次、当前批次以及总批次数
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    # 创建摘要信息的主体部分，包括当前批次和平均值的损失、IoU、Dice系数和准确度
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    string += 'Acc:{:.3f} '.format(acc)
    string += '(Avg {:.4f}) '.format(average_acc)
    # 如果模式是训练模式，添加学习率到摘要信息中
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # 添加当前批次和平均时间到摘要信息中
    string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    # 将摘要信息的标题和主体部分结合起来，然后使用logger对象打印摘要信息
    summary += string
    logger.info(summary)
    # print summary
'''总结：print_summary函数用于生成并打印训练或测试摘要信息，这些信息包括当前轮次、批次、损失、IoU、Dice系数、准确度、学习率以及运行时间等'''

##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
# 这是一个用于训练一个epoch的函数train_one_epoch。这个函数包含了如下几个主要部分
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    # 确定当前是训练模式还是验证模式
    logging_mode = 'Train' if model.training else 'Val'
    # 初始化时间和各种指标的累加器
    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0

    dices = []
    # 遍历数据加载器
    for i, (sampled_batch, names) in enumerate(loader, 1):
        # 尝试获取损失函数的名称，如果失败，则使用损失函数对象的名称
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # 将图像和标签移动到GPU
        images, masks = sampled_batch['image'], sampled_batch['label']
        images, masks = images.cuda(), masks.cuda()
        # print("images shape:",images.shape) #torch.Size([4, 3, 224, 224])
        # print("masks shape:",masks.shape) #torch.Size([4, 224, 224])

        # ====================================================
        #             Compute loss
        # ====================================================
        # 进行模型前向传播和损失计算
        preds = model(images)
        # print("preds shape:", preds.shape)
        out_loss = criterion(preds, masks.float())  # Loss
        # print(preds)
        # print(out_loss)
        # 如果模型处于训练模式，进行反向传播和权重更新
        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()


        # 计算当前批次的IoU、Dice系数和准确度
        train_iou = iou_on_batch(masks,preds)
        train_dice = criterion._show_dice(preds, masks.float())
        train_auc = auc_on_batch(masks,preds)
        # print(train_iou)
        # print(train_dice)
        batch_time = time.time() - end

        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        #     将当前批次的Dice系数添加到列表中
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        acc_sum += len(images) * train_auc
        dice_sum += len(images) * train_dice
        # 计算平均损失、IoU、Dice系数和准确度
        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()
        # 打印当前批次的摘要信息
        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, train_auc, train_acc_average, logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)
        # 如果配置了tensorboard，将当前批次的指标添加到tensorboard中
        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            writer.add_scalar(logging_mode + '_acc', train_auc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()
    # 更新学习率
    if lr_scheduler is not None:
        lr_scheduler.step()
    # if epoch + 1 > 10: # Plateau
    #     if lr_scheduler is not None:
    #         lr_scheduler.step(train_dice_avg)
    # 返回平均损失和平均Dice系数
    return average_loss, train_dice_avg

