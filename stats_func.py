import numpy as np
import torch
import torch.nn.functional as F


def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.clip_by_value(tf.div(exponential_map, tensor_sum_exp), -1.0 * 1e15, 1.0 * 1e15,
                            name="pixel_softmax_2d")


def jaccard(conf_matrix):
    num_cls = conf_matrix.shape[0]
    jac = np.zeros(num_cls)
    for ii in range(num_cls):
        pp = np.sum(conf_matrix[:, ii])
        gp = np.sum(conf_matrix[ii, :])
        hit = conf_matrix[ii, ii]
        jac[ii] = hit * 1.0 / (pp + gp - hit)
    return jac


def dice(conf_matrix):
    num_cls = conf_matrix.shape[0]
    dic = np.zeros(num_cls)
    for ii in range(num_cls):
        pp = np.sum(conf_matrix[:, ii])
        gp = np.sum(conf_matrix[ii, :])
        hit = conf_matrix[ii, ii]
        if (pp + gp) == 0:
            dic[ii] = 0
        else:
            dic[ii] = 2.0 * hit / (pp + gp)
    return dic


"""
params:
    compact_pred: argmax之后的预测结果，shape为[batch_size, height, width]
    labels: ground truth，shape为[batch_size, height, width]
    n_class: 类别数
return:
    dice_arr: 各类别的dice值组成的数组
Dice系数:
    Dice系数是一种用于评估两个样本的相似度的指标，其定义如下：
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    其中A为预测结果，B为ground truth
    该指标的取值范围为[0, 1]，值越大表示两个样本越相似
    该指标常用于图像分割任务中
    Dice系数计算用于二分类，多分类问题只需要转变为onehot编码再对每个类求Dice系数即可
"""
def dice_eval(compact_pred, labels, n_class):
    dice_arr = []
    eps = 1e-7
    pred = F.one_hot(compact_pred, num_classes=n_class).permute(0, 3, 1, 2).float()
    labels = F.one_hot(labels, num_classes=n_class).permute(0, 3, 1, 2).float()

    for i in range(n_class):
        inse = torch.sum(pred[:, i, :, :] * labels[:, i, :, :])
        union = torch.sum(pred[:, i, :, :]) + torch.sum(labels[:, i, :, :])
        dice = 2.0 * inse / (union + eps)
        dice_arr.append(dice.item())

    return dice_arr
