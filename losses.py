import tensorflow as tf
import torch
import torch.nn.functional as F

def cycle_consistency_loss(real_images, generated_images):
    """
    Compute the cycle consistency loss.
    """
    return torch.mean(torch.abs(real_images - generated_images))


# 与1平方差的均值
def lsgan_loss_generator(prob_fake_is_real):
    """
    Computes the LS-GAN loss as minimized by the generator.
    """
    return torch.mean(F.mse_loss(prob_fake_is_real, torch.ones(prob_fake_is_real.size()).cuda()))


# prob_real_is_real与1的平方差的均值+prob_fake_is_real与0的平方差的均值 除以2
def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
    """
    Computes the LS-GAN loss as minimized by the discriminator.
    """
    return (torch.mean(F.mse_loss(prob_real_is_real, torch.ones(prob_real_is_real.size()).cuda())) +
            torch.mean(F.mse_loss(prob_fake_is_real, torch.zeros(prob_fake_is_real.size()).cuda()))) * 0.5

"""
加权交叉熵损失，类别在label中占比越少，权重越大
params:
    logits: 网络输出，softmax之前，shape为[batch_size, n_class, height, width]
"""
def _softmax_weighted_loss(logits, gt, n_class):
    """
    Calculate weighted cross-entropy loss.
    """
    softmaxpred = F.softmax(logits, dim=1)
    gt = F.one_hot(gt, num_classes=n_class).permute(0, 3, 1, 2).float()
    raw_loss = torch.tensor(0)
    for i in range(5):
        gti = gt[:, i, :, :]
        predi = softmaxpred[:, i, :, :]
        weighted = 1 - (torch.sum(gti) / torch.sum(gt))
        raw_loss = raw_loss +(-1.0 * weighted * gti * torch.log(torch.clamp(predi, min=0.005, max=1.0)))

    loss = torch.mean(raw_loss)
    return loss


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
    
    改进后的Dice，可微分
"""
def _dice_loss_fun(pred, target, num_classes):
    """
    pred: softmax 输出 (batch_size, num_classes, height, width)
    target: one-hot 编码的标签 (batch_size, num_classes, height, width)
    """
    eps = 1e-7
    target = F.one_hot(target,num_classes=num_classes).permute(0, 3, 1, 2)
    intersection = torch.sum(pred * target, dim=(2, 3))
    union = torch.sum(pred + target, dim=(2, 3))
    dice = 2.0 * intersection / (union + eps)
    return 1 - torch.mean(dice)

"""
# 问题就出在这里，Dice损失求解过程中用到了argmax操作，argmax是不可微分的因此之后的张量的gradfn都为None，损失也就无法反向传播、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、
def _dice_loss_fun(logits_pred, labels, n_class):
    dice_arr = []
    eps = 1e-7
    compact_pred = torch.argmax(logits_pred, dim=1)
    pred = F.one_hot(compact_pred, num_classes=n_class).permute(0, 3, 1, 2).float()
    labels = F.one_hot(labels, num_classes=n_class).permute(0, 3, 1, 2).float()

    for i in range(n_class):
        inse = torch.sum(pred[:, i, :, :] * labels[:, i, :, :])
        union = torch.sum(pred[:, i, :, :]) + torch.sum(labels[:, i, :, :])
        dice = 2.0 * inse / (union + eps)
        dice_arr.append(dice)
    dice = torch.mean(torch.stack(dice_arr))
    return 1-dice
"""



"""
params:
    prediction: 网络输出，softmax之前，shape为[batch_size, n_class, height, width]
    gt: ground truth，shape为[batch_size, height, width]
    n_class: 类别数
"""
def task_loss(prediction, gt, n_class):
    """
    Calculate task loss, which consists of the weighted cross entropy loss and dice loss
    """
    ce_loss = _softmax_weighted_loss(prediction, gt, n_class)
    dice_loss = _dice_loss_fun(prediction, gt,n_class)
    return ce_loss, dice_loss
