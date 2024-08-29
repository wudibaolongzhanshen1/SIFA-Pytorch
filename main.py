"""Code for training SIFA."""
import logging
from datetime import datetime
import json
import numpy as np
import random
import os
import cv2
import time
from torchmetrics import JaccardIndex
import torch
import torchsummary
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F
import dataset
import model
import tensorflow as tf
from losses import *
import data_loader, losses, model
from stats_func import *
from torchviz import make_dot

torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
with open('./config_param.json') as config_file:
    config = json.load(config_file)
NUM_CLS = int(config['num_cls'])
target_val_dataset_path = config['target_val_dataset_path']
source_train_dataset_path = config['source_train_dataset_path']
target_train_dataset_path = config['target_train_dataset_path']
source_val_dataset_path = config['source_val_dataset_path']
OUTPUT_ROOT_DIR = config['output_root_dir']
if not os.path.isdir(OUTPUT_ROOT_DIR):
    os.makedirs(OUTPUT_ROOT_DIR)
    os.makedirs(os.path.join(OUTPUT_ROOT_DIR, 'logs'))
    os.makedirs(os.path.join(OUTPUT_ROOT_DIR, 'imgs'))
LOGS_DIR = os.path.join(OUTPUT_ROOT_DIR, 'logs')
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, current_time)
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'imgs')
NUM_IMGS_TO_SAVE = 20
POOL_SIZE = int(config['pool_size'])
LAMBDA_A = float(config['_LAMBDA_A'])
LAMBDA_B = float(config['_LAMBDA_B'])
SKIP = bool(config['skip'])
BASE_LR = float(config['base_lr'])
MAX_STEP = int(config['max_step'])
KEEP_RATE_VALUE = float(config['keep_rate_value'])
IS_TRAINING_VALUE = bool(config['is_training_value'])
BATCH_SIZE = int(config['batch_size'])
LR_GAN_DECAY = bool(config['lr_gan_decay'])
RESTORE_FROM = config['restore_from']
CHECKPOINT_DIR = config['checkpoint_dir']
if CHECKPOINT_DIR == '':
    CHECKPOINT_DIR = './checkpoints'
if not os.path.isdir(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

save_interval = 300
evaluation_interval = 100
log_interval = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model.MyModel().to(device)
if RESTORE_FROM is not None and RESTORE_FROM != '':
    net.load_state_dict(torch.load(RESTORE_FROM))
    print(f'load model from {config["restore_path"]}')



def train(config):
    filename = f'{LOGS_DIR}/{current_time}.log'
    logging.basicConfig(level=logging.INFO, filename=filename, filemode='w',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logger = logging.getLogger('my_logger')
    checkpoint_path = os.path.join(CHECKPOINT_DIR, current_time)
    os.makedirs(checkpoint_path)
    output_imgs_path = os.path.join(OUTPUT_ROOT_DIR,'imgs', current_time)
    os.makedirs(output_imgs_path)
    for key, value in config.items():
        print(f'{key}: {value}')
        logger.info(f'{key}: {value}')

    optimizer_gan = torch.optim.Adam(
        list(net.encoder_of_generator_t.parameters()) +
        list(net.decoder_of_generator_t.parameters()) +
        list(net.discriminator_aux_s.parameters()) +
        list(net.discriminator_p_ll.parameters()) +
        list(net.discriminator_t.parameters()) +
        list(net.discriminator_p.parameters()) +
        list(net.decoder.parameters()),
        lr=BASE_LR, betas=(0.5, 0.999)
    )
    optimizer_seg = torch.optim.Adam(
        list(net.encoder.parameters()) +
        list(net.pixel_wise_classifier.parameters()) +
        list(net.pixel_wise_classifier_ll.parameters()),
        lr=BASE_LR
    )

    source_train_dataset = dataset.MyDataset(source_train_dataset_path)
    target_train_dataset = dataset.MyDataset(target_train_dataset_path)
    source_val_dataset = dataset.MyDataset(source_val_dataset_path)
    target_val_dataset = dataset.MyDataset(target_val_dataset_path)
    source_train_dataloader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    target_train_dataloader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    source_val_dataloader = DataLoader(source_val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    target_val_dataloader = DataLoader(target_val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    len_source_train_dataloader = len(source_train_dataloader)
    len_target_train_dataloader = len(target_train_dataloader)
    len_source_val_dataloader = len(source_val_dataloader)
    len_target_val_dataloader = len(target_val_dataloader)

    # encoder = net.encoder
    # torchsummary.summary(encoder, (1, 256, 256))
    # decoder = net.decoder
    # torchsummary.summary(decoder, [(512,32,32),(1,256,256)])
    # pixel_wise_classifier = net.pixel_wise_classifier
    # torchsummary.summary(pixel_wise_classifier, (512,32,32))
    # torchsummary.summary(net, [(1, 256, 256),(1, 256, 256)])

    for epoch in range(MAX_STEP):
        net.train()
        iter_source_train_dataloader = iter(source_train_dataloader)
        iter_target_train_dataloader = iter(target_train_dataloader)
        total_loss_seg = 0.0
        total_cycle_consis_loss = 0.0
        total_lsgan_loss_b = 0.0
        total_d_loss_B = 0.0
        total_lsgan_loss_a = 0.0
        total_d_loss_A = 0.0
        total_lsgan_loss_a_2 = 0.0
        total_lsgan_loss_p = 0.0
        output_imgs_count = 0
        for i in range(min(len_source_train_dataloader, len_target_train_dataloader)):
            print(i)
            optimizer_seg.zero_grad()
            optimizer_gan.zero_grad()
            source_train_data = next(iter_source_train_dataloader)
            target_train_data = next(iter_target_train_dataloader)
            source_train_imgs, source_train_labels = source_train_data[0], source_train_data[1]
            target_train_imgs, _ = target_train_data[0], target_train_data[1]
            source_train_imgs = source_train_imgs[:, None, :, :]
            target_train_imgs = target_train_imgs[:, None, :, :]
            source_train_imgs = source_train_imgs.to(device)
            source_train_labels = source_train_labels.to(device)
            target_train_imgs = target_train_imgs.to(device)
            if source_train_imgs.shape[0] != target_train_imgs.shape[0]:
                break
            outputs = net(source_train_imgs, target_train_imgs)
            losses = compute_losses_and_backward2_and_update(outputs, source_train_imgs, target_train_imgs, source_train_labels)
            with torch.no_grad():
                optimizer_seg.step()
                optimizer_gan.step()
                total_loss_seg += losses['loss_seg']
                total_cycle_consis_loss += losses['cycle_consis_loss']
                total_lsgan_loss_b += losses['lsgan_loss_b']
                total_d_loss_B += losses['d_loss_B']
                total_lsgan_loss_a += losses['lsgan_loss_a']
                total_d_loss_A += losses['d_loss_A']
                total_lsgan_loss_a_2 += losses['lsgan_loss_a_2']
                total_lsgan_loss_p += losses['lsgan_loss_p']
        if epoch % log_interval == 0:
            print(
                'iter = {0:8d}/{1:8d}, total_loss_seg = {2:.3f} total_cycle_consis_loss = {3:.3f} total_lsgan_loss_b = {4:.3f} total_d_loss_B = {5:.3f} total_lsgan_loss_a = {6:.3f} total_d_loss_A = {7:.3f} total_lsgan_loss_a_2 = {8:.3f} total_lsgan_loss_p = {9:.3f}'.format(
                    epoch, MAX_STEP, total_loss_seg, total_cycle_consis_loss, total_lsgan_loss_b, total_d_loss_B,
                    total_lsgan_loss_a, total_d_loss_A, total_lsgan_loss_a_2, total_lsgan_loss_p))
            logger.info(
                'iter = {0:8d}/{1:8d}, total_loss_seg = {2:.3f} total_cycle_consis_loss = {3:.3f} total_lsgan_loss_b = {4:.3f} total_d_loss_B = {5:.3f} total_lsgan_loss_a = {6:.3f} total_d_loss_A = {7:.3f} total_lsgan_loss_a_2 = {8:.3f} total_lsgan_loss_p = {9:.3f}'.format(
                    epoch, MAX_STEP, total_loss_seg, total_cycle_consis_loss, total_lsgan_loss_b, total_d_loss_B,
                    total_lsgan_loss_a, total_d_loss_A, total_lsgan_loss_a_2, total_lsgan_loss_p))
        if epoch % save_interval == 0:
            torch.save(net.state_dict(), f'{checkpoint_path}/model_{epoch}.pth')
        if epoch % evaluation_interval == 0:
            net.eval()
            with torch.no_grad():
                total_dice = 0.0
                total_jaccard = 0.0
                iter_target_val_dataloader = iter(target_val_dataloader)
                iter_source_val_dataloader = iter(source_val_dataloader)
                for i in range(len_target_val_dataloader):
                    target_val_data = next(iter_target_val_dataloader)
                    target_val_imgs, target_val_labels = target_val_data[0], target_val_data[1]
                    target_val_imgs = target_val_imgs[:, None, :, :]
                    target_val_imgs = target_val_imgs.to(device)
                    source_val_data = next(iter_source_val_dataloader)
                    source_val_imgs, source_val_labels = source_val_data[0], source_val_data[1]
                    source_val_imgs = source_val_imgs[:, None, :, :]
                    source_val_imgs = source_val_imgs.to(device)
                    outputs = net(source_val_imgs,target_val_imgs)

                    cycle_images_a = outputs['cycle_images_a']
                    cycle_images_b = outputs['cycle_images_b']
                    fake_images_a = outputs['fake_images_a']
                    fake_images_b = outputs['fake_images_b']
                    pred_mask_b = (outputs['pred_mask_b'] + outputs['pred_mask_b_ll'])/2
                    pred_mask_b = torch.argmax(pred_mask_b, dim=1)
                    pred_mask_b = pred_mask_b.cpu().numpy()
                    pred_mask_fake_b = (outputs['pred_mask_fake_b'] + outputs['pred_mask_fake_b_ll'])/2
                    pred_mask_fake_b = torch.argmax(pred_mask_fake_b, dim=1)
                    pred_mask_fake_b = pred_mask_fake_b.cpu().numpy()

                    for j in range(pred_mask_b.shape[0]):
                        save_tensor_to_png(source_val_imgs[j], f'{output_imgs_path}/source_val_imgs_{output_imgs_count}.png')
                        save_tensor_to_png(target_val_imgs[j], f'{output_imgs_path}/target_val_imgs_{output_imgs_count}.png')
                        save_tensor_to_png(source_val_labels[j], f'{output_imgs_path}/source_val_labels_{output_imgs_count}.png')
                        save_tensor_to_png(pred_mask_b[j], f'{output_imgs_path}/pred_mask_b_{output_imgs_count}.png')
                        save_tensor_to_png(cycle_images_a[j], f'{output_imgs_path}/cycle_images_a_{output_imgs_count}.png')
                        save_tensor_to_png(cycle_images_b[j], f'{output_imgs_path}/cycle_images_b_{output_imgs_count}.png')
                        save_tensor_to_png(fake_images_a[j], f'{output_imgs_path}/fake_images_a_{output_imgs_count}.png')
                        save_tensor_to_png(fake_images_b[j], f'{output_imgs_path}/fake_images_b_{output_imgs_count}.png')
                        save_tensor_to_png(pred_mask_fake_b[j], f'{output_imgs_path}/pred_mask_fake_b_{output_imgs_count}.png')
                        output_imgs_count += 1
                    iou_metric = JaccardIndex(num_classes=NUM_CLS, average='none')
                    iou_per_class = iou_metric(pred_mask_fake_b, source_val_labels)
                    print()





def compute_losses_and_backward2_and_update(outputs, source_train_imgs, target_train_imgs, source_train_labels):
    from torchviz import make_dot

    prob_real_a_is_real = outputs['prob_real_a_is_real']
    prob_real_b_is_real = outputs['prob_real_b_is_real']
    prob_fake_a_is_real = outputs['prob_fake_a_is_real']
    prob_fake_b_is_real = outputs['prob_fake_b_is_real']

    cycle_images_a = outputs['cycle_images_a']
    cycle_images_b = outputs['cycle_images_b']
    fake_images_a = outputs['fake_images_a']
    fake_images_b = outputs['fake_images_b']

    pred_mask_b = outputs['pred_mask_b']
    pred_mask_b_ll = outputs['pred_mask_b_ll']
    pred_mask_fake_b = outputs['pred_mask_fake_b']
    pred_mask_fake_b_ll = outputs['pred_mask_fake_b_ll']

    prob_pred_mask_fake_b_is_real = outputs['prob_pred_mask_fake_b_is_real']
    prob_pred_mask_b_is_real = outputs['prob_pred_mask_b_is_real']
    prob_pred_mask_fake_b_ll_is_real = outputs['prob_pred_mask_fake_b_ll_is_real']
    prob_pred_mask_b_ll_is_real = outputs['prob_pred_mask_b_ll_is_real']

    prob_fake_a_aux_is_real = outputs['prob_fake_a_aux_is_real']
    prob_cycle_a_aux_is_real = outputs['prob_cycle_a_aux_is_real']
    prob_real_a_aux_is_real = outputs['prob_real_a_aux_is_real']
    prob_cycle_a_is_real = outputs['prob_cycle_a_is_real']

    '计算两个L_seg'
    for param in net.parameters():
        param.requires_grad = False
    for param in net.encoder.parameters():
        param.requires_grad = True
    for param in net.pixel_wise_classifier.parameters():
        param.requires_grad = True
    for param in net.pixel_wise_classifier_ll.parameters():
        param.requires_grad = True
    # 为True的params：encoder,pixel_wise_classifier,pixel_wise_classifier_ll
    # 计算x_s2t分割结果与x_s分割标签的损失(celoss和diceloss，celoss:加权交叉熵损失，diceloss对类别不平衡问题有鲁棒性)
    ce_loss_fake_b, dice_loss_fake_b = task_loss(pred_mask_fake_b, source_train_labels, NUM_CLS)
    # 计算x_s2t低层级分割结果与x_s分割标签的损失
    ce_loss_fake_b_ll, dice_loss_fake_b_ll = task_loss(pred_mask_fake_b_ll, source_train_labels, NUM_CLS)
    loss_seg = ce_loss_fake_b + dice_loss_fake_b + ce_loss_fake_b_ll + dice_loss_fake_b_ll
    # 只求encoder和classifier的梯度
    loss_seg.backward(retain_graph=True)


    '计算两个L_cycle'
    # x_s与x_s2t2s的一致性损失(像素差值的平均)
    cycle_consistency_loss_a = LAMBDA_A * cycle_consistency_loss(
        real_images=source_train_imgs, generated_images=cycle_images_a
    )
    # x_t与x_t2s2t的一致性损失(像素差值的平均)
    cycle_consistency_loss_b = LAMBDA_B * cycle_consistency_loss(
        real_images=target_train_imgs, generated_images=cycle_images_b
    )
    cycle_consis_loss = cycle_consistency_loss_a + cycle_consistency_loss_b
    # 只求encoder和classifier的梯度
    for param in net.pixel_wise_classifier.parameters():
        param.requires_grad = False
    for param in net.pixel_wise_classifier_ll.parameters():
        param.requires_grad = False
    for param in net.encoder_of_generator_t.parameters():
        param.requires_grad = True
    for param in net.decoder_of_generator_t.parameters():
        param.requires_grad = True
    for param in net.decoder.parameters():
        param.requires_grad = True
    # 为True的params：encoder_of_generator_t,decoder_of_generator_t,encoder,decoder
    cycle_consis_loss.backward(retain_graph=True)

    """计算L_adv_t梯度"""
    # prob_fake_b_is_real: x_s2t送入判别器D_t的输出
    lsgan_loss_b = lsgan_loss_generator(prob_fake_b_is_real)
    for param in net.encoder.parameters():
        param.requires_grad = False
    for param in net.decoder.parameters():
        param.requires_grad = False
    # 为True的params：encoder_of_generator_t,decoder_of_generator_t
    lsgan_loss_b.backward(retain_graph=True)
    # (prob_real_b_is_real与1平方差 + prob_fake_b_is_real与0平方差)的均值，该损失用来提高D_t的判别能力
    d_loss_B = lsgan_loss_discriminator(
        prob_real_is_real=prob_real_b_is_real,
        prob_fake_is_real=prob_fake_b_is_real
    )
    for param in net.encoder_of_generator_t.parameters():
        param.requires_grad = False
    for param in net.decoder_of_generator_t.parameters():
        param.requires_grad = False
    for param in net.discriminator_t.parameters():
        param.requires_grad = True
    # 为True的params：discriminator_t
    d_loss_B.backward(retain_graph=True)

    """计算L_adv_s梯度"""
    # prob_fake_a_is_real: x_t2s送入判别器D_s的输出
    # lsgan_loss_a: prob_fake_a_is_real与1的平方差均值
    lsgan_loss_a = lsgan_loss_generator(prob_fake_a_is_real)
    lsgan_loss_a += lsgan_loss_generator(prob_fake_a_aux_is_real)
    for param in net.discriminator_t.parameters():
        param.requires_grad = False
    for param in net.encoder.parameters():
        param.requires_grad = True
    for param in net.decoder.parameters():
        param.requires_grad = True
    # 为True的params：encoder,decoder
    lsgan_loss_a.backward(retain_graph=True)
    # (prob_real_a_is_real与1平方差 + prob_fake_a_is_real与0平方差)的均值，该损失用来提高D_s的判别能力
    d_loss_A = lsgan_loss_discriminator(
        prob_real_is_real=prob_real_a_is_real,
        prob_fake_is_real=prob_fake_a_is_real
    )
    d_loss_A += lsgan_loss_discriminator(
        prob_real_is_real=prob_real_a_aux_is_real,
        prob_fake_is_real=prob_fake_a_aux_is_real
    )
    for param in net.encoder.parameters():
        param.requires_grad = False
    for param in net.decoder.parameters():
        param.requires_grad = False
    for param in net.discriminator_aux_s.parameters():
        param.requires_grad = True
    # 为True的params：discriminator_aux_s
    d_loss_A.backward(retain_graph=True)

    """计算L_adv_s_2梯度(x_s2t2s与x_t2s)"""
    for param in net.discriminator_aux_s.parameters():
        param.requires_grad = False
    for param in net.encoder.parameters():
        param.requires_grad = True
    # 为True的params：encoder
    lsgan_loss_a_2 = lsgan_loss_generator(prob_cycle_a_is_real)
    lsgan_loss_a_2 += lsgan_loss_generator(prob_cycle_a_aux_is_real)
    lsgan_loss_a_2 += lsgan_loss_generator(prob_fake_a_is_real)
    lsgan_loss_a_2 += lsgan_loss_generator(prob_fake_a_aux_is_real)
    lsgan_loss_a_2.backward(retain_graph=True)

    """计算L_adv_p梯度"""
    # 首先，该损失鼓励生成的fake_b(x_s2t)尽量接近真实的b，也就是鼓励提升源域到目标域生成器的生成效果；而且提高encoder的从目标域提取出源域特征的能力
    # prob_pred_mask_b_is_real: x_s2t的分割预测结果送入判别器D_p的输出
    lsgan_loss_p = lsgan_loss_generator(prob_pred_mask_fake_b_is_real)
    # prob_pred_mask_b_ll_is_real: x_s2t的分割预测低层级结果送入判别器D_t的输出
    lsgan_loss_p += lsgan_loss_generator(prob_pred_mask_fake_b_ll_is_real)
    # 为True的params：encoder
    lsgan_loss_p.backward(retain_graph=True)
    # (x_s2t预测结果送入D_p与0平方差 + x_t预测结果送入D_p与1平方差)的均值，该损失用来提高D_p的判别能力，
    d_loss_P = lsgan_loss_discriminator(
        prob_real_is_real=prob_pred_mask_b_is_real,
        prob_fake_is_real=prob_pred_mask_fake_b_is_real
    )
    d_loss_P += lsgan_loss_discriminator(
        prob_real_is_real=prob_pred_mask_b_ll_is_real,
        prob_fake_is_real=prob_pred_mask_fake_b_ll_is_real
    )
    for param in net.encoder.parameters():
        param.requires_grad = False
    for param in net.discriminator_p.parameters():
        param.requires_grad = True
    # 为True的params：discriminator_p
    d_loss_P.backward()
    for param in net.parameters():
        param.requires_grad = True
    torch.cuda.empty_cache()
    return {
        'loss_seg': loss_seg,
        'cycle_consis_loss': cycle_consis_loss,
        'lsgan_loss_b': lsgan_loss_b,
        'd_loss_B': d_loss_B,
        'lsgan_loss_a': lsgan_loss_a,
        'd_loss_A': d_loss_A,
        'lsgan_loss_a_2': lsgan_loss_a_2,
        'lsgan_loss_p': lsgan_loss_p,
        'd_loss_P': d_loss_P
    }


"""易于理解的源代码，上边的compute_losses_and_backward2是优化后的代码"""


def compute_losses_and_backward(outputs, source_train_imgs, target_train_imgs, source_train_labels):
    prob_real_a_is_real = outputs['prob_real_a_is_real']
    prob_real_b_is_real = outputs['prob_real_b_is_real']
    prob_fake_a_is_real = outputs['prob_fake_a_is_real']
    prob_fake_b_is_real = outputs['prob_fake_b_is_real']

    cycle_images_a = outputs['cycle_images_a']
    cycle_images_b = outputs['cycle_images_b']
    fake_images_a = outputs['fake_images_a']
    fake_images_b = outputs['fake_images_b']

    pred_mask_b = outputs['pred_mask_b']
    pred_mask_b_ll = outputs['pred_mask_b_ll']
    pred_mask_fake_b = outputs['pred_mask_fake_b']
    pred_mask_fake_b_ll = outputs['pred_mask_fake_b_ll']

    prob_pred_mask_fake_b_is_real = outputs['prob_pred_mask_fake_b_is_real']
    prob_pred_mask_b_is_real = outputs['prob_pred_mask_b_is_real']
    prob_pred_mask_fake_b_ll_is_real = outputs['prob_pred_mask_fake_b_ll_is_real']
    prob_pred_mask_b_ll_is_real = outputs['prob_pred_mask_b_ll_is_real']

    prob_fake_a_aux_is_real = outputs['prob_fake_a_aux_is_real']
    prob_cycle_a_aux_is_real = outputs['prob_cycle_a_aux_is_real']
    prob_real_a_aux_is_real = outputs['prob_real_a_aux_is_real']
    prob_cycle_a_is_real = outputs['prob_cycle_a_is_real']

    '计算两个L_seg'
    # 计算x_s2t分割结果与x_s分割标签的损失(celoss和diceloss，celoss:加权交叉熵损失，diceloss对类别不平衡问题有鲁棒性)
    ce_loss_b, dice_loss_b = task_loss(pred_mask_fake_b, source_train_labels, NUM_CLS)
    # 计算x_s2t低层级分割结果与x_s分割标签的损失
    ce_loss_b_ll, dice_loss_b_ll = task_loss(pred_mask_fake_b_ll, source_train_labels, NUM_CLS)
    loss_seg = ce_loss_b + dice_loss_b + ce_loss_b_ll + dice_loss_b_ll
    # 只求encoder和classifier的梯度
    for param in net.parameters():
        param.requires_grad = False
    for param in net.encoder.parameters():
        param.requires_grad = True
    for param in net.pixel_wise_classifier.parameters():
        param.requires_grad = True
    for param in net.pixel_wise_classifier_ll.parameters():
        param.requires_grad = True
    loss_seg.backward()

    '计算两个L_cycle'
    # x_s与x_s2t2s的一致性损失(像素差值的平均)
    cycle_consistency_loss_a = LAMBDA_A * cycle_consistency_loss(
        real_images=source_train_imgs, generated_images=cycle_images_a
    )
    # x_t与x_t2s2t的一致性损失(像素差值的平均)
    cycle_consistency_loss_b = LAMBDA_B * cycle_consistency_loss(
        real_images=target_train_imgs, generated_images=cycle_images_b
    )
    cycle_consis_loss = cycle_consistency_loss_a + cycle_consistency_loss_b
    # 只求encoder和classifier的梯度
    for param in net.parameters():
        param.requires_grad = False
    for param in net.encoder_of_generator_t.parameters():
        param.requires_grad = True
    for param in net.decoder_of_generator_t.parameters():
        param.requires_grad = True
    for param in net.encoder.parameters():
        param.requires_grad = True
    for param in net.decoder.parameters():
        param.requires_grad = True
    cycle_consis_loss.backward()

    """计算L_adv_t梯度"""
    # prob_fake_b_is_real: x_s2t送入判别器D_t的输出
    lsgan_loss_b = lsgan_loss_generator(prob_fake_b_is_real)
    for param in net.parameters():
        param.requires_grad = False
    for param in net.encoder_of_generator_t.parameters():
        param.requires_grad = True
    for param in net.decoder_of_generator_t.parameters():
        param.requires_grad = True
    lsgan_loss_b.backward()
    # (prob_real_b_is_real与1平方差 + prob_fake_b_is_real与0平方差)的均值，该损失用来提高D_t的判别能力
    d_loss_B = lsgan_loss_discriminator(
        prob_real_is_real=prob_real_b_is_real,
        prob_fake_is_real=prob_fake_b_is_real
    )
    for param in net.parameters():
        param.requires_grad = False
    for param in net.discriminator_t.parameters():
        param.requires_grad = True
    d_loss_B.backward()

    """计算L_adv_s梯度"""
    # prob_fake_a_is_real: x_t2s送入判别器D_s的输出
    # lsgan_loss_a: prob_fake_a_is_real与1的平方差均值
    lsgan_loss_a = lsgan_loss_generator(prob_fake_a_is_real)
    lsgan_loss_a += lsgan_loss_generator(prob_fake_a_aux_is_real)
    for param in net.parameters():
        param.requires_grad = False
    for param in net.encoder.parameters():
        param.requires_grad = True
    for param in net.decoder.parameters():
        param.requires_grad = True
    lsgan_loss_a.backward()
    # (prob_real_a_is_real与1平方差 + prob_fake_a_is_real与0平方差)的均值，该损失用来提高D_s的判别能力
    d_loss_A = lsgan_loss_discriminator(
        prob_real_is_real=prob_real_a_is_real,
        prob_fake_is_real=prob_fake_a_is_real
    )
    d_loss_A += lsgan_loss_discriminator(
        prob_real_is_real=prob_real_a_aux_is_real,
        prob_fake_is_real=prob_fake_a_aux_is_real
    )
    for param in net.parameters():
        param.requires_grad = False
    for param in net.discriminator_aux_s.parameters():
        param.requires_grad = True
    d_loss_A.backward()

    """计算L_adv_s_2梯度(x_s2t2s与x_t2s)"""
    for param in net.parameters():
        param.requires_grad = False
    for param in net.encoder.parameters():
        param.requires_grad = True
    lsgan_loss_a_2 = lsgan_loss_generator(prob_cycle_a_is_real)
    lsgan_loss_a_2 += lsgan_loss_generator(prob_cycle_a_aux_is_real)
    lsgan_loss_a_2 += lsgan_loss_generator(prob_fake_a_is_real)
    lsgan_loss_a_2 += lsgan_loss_generator(prob_fake_a_aux_is_real)
    lsgan_loss_a_2.backward()

    """计算L_adv_p梯度"""
    # 首先，该损失鼓励生成的fake_b(x_s2t)尽量接近真实的b，也就是鼓励提升源域到目标域生成器的生成效果；而且提高encoder的从目标域提取出源域特征的能力
    # prob_pred_mask_b_is_real: x_s2t的分割预测结果送入判别器D_p的输出
    lsgan_loss_p = lsgan_loss_generator(prob_pred_mask_fake_b_is_real)
    # prob_pred_mask_b_ll_is_real: x_s2t的分割预测低层级结果送入判别器D_t的输出
    lsgan_loss_p += lsgan_loss_generator(prob_pred_mask_fake_b_ll_is_real)
    for param in net.parameters():
        param.requires_grad = False
    for param in net.encoder.parameters():
        param.requires_grad = True
    lsgan_loss_p.backward()
    # (x_s2t预测结果送入D_p与0平方差 + x_t预测结果送入D_p与1平方差)的均值，该损失用来提高D_p的判别能力，
    d_loss_P = lsgan_loss_discriminator(
        prob_real_is_real=prob_pred_mask_b_is_real,
        prob_fake_is_real=prob_pred_mask_fake_b_is_real
    )
    d_loss_P += lsgan_loss_discriminator(
        prob_real_is_real=prob_pred_mask_b_ll_is_real,
        prob_fake_is_real=prob_pred_mask_fake_b_ll_is_real
    )
    for param in net.parameters():
        param.requires_grad = False
    for param in net.discriminator_p.parameters():
        param.requires_grad = True
    d_loss_P.backward()


def save_tensor_to_png(tensor, filepath):
    # 将 Tensor 转换为 numpy 数组
    numpy_image = tensor.permute(1, 2, 0).numpy()
    # 将 numpy 数组的值映射到 [0, 255] 范围
    numpy_image = (numpy_image - numpy_image.min()) / (numpy_image.max() - numpy_image.min()) * 255
    numpy_image = numpy_image.astype(np.uint8)
    # 将 numpy 数组转换为 PIL 图像
    pil_image = Image.fromarray(numpy_image)
    # 保存 PIL 图像为 PNG 文件
    pil_image.save(filepath)



if __name__ == '__main__':
    train(config=config)
    # main(config_filename='./config_param.json')
