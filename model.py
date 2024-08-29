import tensorflow as tf
from torch import nn
import torch.nn.functional as F
import json
import torch

with open('./config_param.json') as config_file:
    config = json.load(config_file)

BATCH_SIZE = int(config['batch_size'])
POOL_SIZE = int(config['pool_size'])

# The height of each image.
IMG_HEIGHT = 256

# The width of each image.
IMG_WIDTH = 256

ngf = 32
ndf = 64

"""
返回:字典
'prob_real_a_is_real': x_s送入判别器D_s得到的输出,
'prob_real_b_is_real': x_t送入判别器D_t得到的输出,
'prob_fake_a_is_real': x_t2s送入判别器D_s得到的输出,
'prob_fake_b_is_real': x_s2t送入判别器D_t得到的输出,
'prob_fake_pool_a_is_real': fake_pool_a送入判别器D_s得到的输出,
'prob_fake_pool_b_is_real': fake_pool_b送入判别器D_t得到的输出,
'cycle_images_a': x_s2t2s,
'cycle_images_b': x_t2s2t,
'fake_images_a': x_t2s,
'fake_images_b': x_s2t,
'pred_mask_a': x_s送入encoder和分割器(pixel-wise classifier)得到的分割结果,
'pred_mask_b': x_t送入encoder和分割器(pixel-wise classifier)得到的分割结果,
'pred_mask_b_ll': x_t送入encoder得到低层级feature和分割器(pixel-wise classifier)得到的分割结果,
'pred_mask_fake_a': x_t2s送入encoder和分割器(pixel-wise classifier)得到的分割结果,
'pred_mask_fake_b': x_s2t送入encoder和分割器(pixel-wise classifier)得到的分割结果,
'pred_mask_fake_b_ll': x_s2t送入encoder得到低层级feature和分割器(pixel-wise classifier)得到的分割结果,
'prob_pred_mask_fake_b_is_real': x_s2t的分割结果送入判别器D_t得到的输出,
'prob_pred_mask_b_is_real': x_t的分割结果送入判别器D_t得到的输出,
'prob_pred_mask_fake_b_ll_is_real': x_s2t的低层级feature分割结果送入判别器D_t得到的输出,
'prob_pred_mask_b_ll_is_real': x_t的低层级feature分割结果送入判别器D_t得到的输出,
'prob_fake_a_aux_is_real': x_t2s送入判别器D_s得到的辅助输出,
'prob_fake_pool_a_aux_is_real': fake_pool_a送入判别器D_s得到的辅助输出,
'prob_cycle_a_aux_is_real': x_s2t2s送入判别器D_s得到的辅助输出,
"""


"""膨胀卷积block"""
class DRNBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=None, use_dropout=False):
        super(DRNBlock, self).__init__()
        self.padding_type = padding_type
        self.norm_layer = norm_layer
        self.use_dropout = use_dropout

        self.conv1 = self.build_conv_layer(dim)
        self.conv2 = self.build_conv_layer(dim, use_relu=False)

    def build_conv_layer(self, dim, use_relu=True):
        conv_block = []
        p = 1  # 调整填充大小为1
        if self.padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(p)]
        elif self.padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(p)]
        elif self.padding_type == 'zero':
            p = 0
        else:
            raise NotImplementedError('padding [%s] is not implemented' % self.padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, dilation=1)]  # 调整卷积层的填充和扩张
        if self.norm_layer is not None:
            conv_block += [self.norm_layer(dim)]
        if use_relu:
            conv_block += [nn.ReLU()]
        if self.use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return F.relu(out + x)


class ResnetBlockDs(nn.Module):
    def __init__(self, dim_in, dim_out, padding_type='reflect', norm_layer=None, keep_rate=0.75):
        super(ResnetBlockDs, self).__init__()
        self.padding_mode = padding_type
        self.keep_rate = keep_rate

        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        if norm_layer is not None:
            self.norm1 = norm_layer(dim_out)
            self.norm2 = norm_layer(dim_out)
            self.norm3 = norm_layer(dim_out)
        else:
            self.norm1 = None
            self.norm2 = None
            self.norm3 = None
    def forward(self, x):
        out_res = self.conv1(x)
        if self.norm1 is not None:
            out_res = self.norm1(out_res)
        out_res = F.relu(out_res)

        out_res = self.conv2(out_res)
        if self.norm2 is not None:
            out_res = self.norm2(out_res)

        x = self.conv3(x)
        if self.norm3 is not None:
            x = self.norm3(x)

        return F.relu(out_res + x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, nef=16, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Encoder, self).__init__()
        self.nef = nef

        self.c1 = nn.Sequential(
            nn.Conv2d(input_nc, nef, kernel_size=7, stride=1, padding=3),
            norm_layer(nef),
            nn.ReLU()
        )
        self.r1 = ResnetBlock(nef, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.r2 = ResnetBlockDs(nef, nef*2, padding_type='reflect', norm_layer=norm_layer)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.r3 = ResnetBlockDs(nef * 2,nef*4, padding_type='reflect', norm_layer=norm_layer)
        self.r4 = ResnetBlock(nef * 4, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.r5 = ResnetBlockDs(nef * 4,nef*8, padding_type='reflect', norm_layer=norm_layer)
        self.r6 = ResnetBlock(nef * 8, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout)

        self.r7 = ResnetBlockDs(nef * 8,nef*16, padding_type='reflect', norm_layer=norm_layer)
        self.r8 = ResnetBlock(nef * 16, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout)

        self.r9 = ResnetBlockDs(nef * 16,nef*32, padding_type='reflect', norm_layer=norm_layer)
        self.r10 = ResnetBlock(nef * 32, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout)

        self.d1 = DRNBlock(nef * 32, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout)
        self.d2 = DRNBlock(nef * 32, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout)

        self.c2 = nn.Sequential(
            nn.Conv2d(nef * 32, nef * 32, kernel_size=3, stride=1, padding=1),
            norm_layer(nef * 32),
            nn.ReLU(False)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(nef * 32, nef * 32, kernel_size=3, stride=1, padding=1),
            norm_layer(nef * 32),
            nn.ReLU(False)
        )


    def forward(self, x):
        o_c1 = self.c1(x)
        o_r1 = self.r1(o_c1)
        out1 = self.pool1(o_r1)

        o_r2 = self.r2(out1)
        out2 = self.pool2(o_r2)

        o_r3 = self.r3(out2)
        o_r4 = self.r4(o_r3)
        out3 = self.pool3(o_r4)

        o_r5 = self.r5(out3)
        o_r6 = self.r6(o_r5)

        o_r7 = self.r7(o_r6)
        o_r8 = self.r8(o_r7)

        o_r9 = self.r9(o_r8)
        o_r10 = self.r10(o_r9)

        o_d1 = self.d1(o_r10)
        o_d2 = self.d2(o_d1)

        o_c2 = self.c2(o_d2)
        o_c3 = self.c3(o_c2)

        return o_c3, o_r10

class Decoder(nn.Module):
    def __init__(self, ngf=16, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Decoder, self).__init__()
        self.ngf = ngf

        self.c1 = nn.Sequential(
            nn.Conv2d(ngf * 32, ngf * 16, kernel_size=3, stride=1, padding=1),
            norm_layer(ngf * 16),
            nn.ReLU()
        )
        self.r1 = ResnetBlockDs(ngf * 16,ngf * 8, padding_type='reflect', norm_layer=norm_layer)
        self.r2 = ResnetBlock(ngf * 8, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout)
        self.r3 = ResnetBlockDs(ngf * 8,ngf*4, padding_type='reflect', norm_layer=norm_layer)
        self.r4 = ResnetBlock(ngf * 4, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout)

        self.c3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 2),
            nn.ReLU()
        )
        self.c4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 2),
            nn.ReLU()
        )
        self.c5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf),
            nn.ReLU()
        )
        self.c6 = nn.Sequential(
            nn.Conv2d(ngf, 1, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x, inputimg, skip=False):
        o_c1 = self.c1(x)
        o_r1 = self.r1(o_c1)
        o_r2 = self.r2(o_r1)
        o_r3 = self.r3(o_r2)
        o_r4 = self.r4(o_r3)

        o_c3 = self.c3(o_r4)
        o_c4 = self.c4(o_c3)
        o_c5 = self.c5(o_c4)
        o_c6 = self.c6(o_c5)

        if skip:
            out_gen = torch.tanh(inputimg + o_c6)
        else:
            out_gen = torch.tanh(o_c6)

        return out_gen

class Segmenter(nn.Module):
    def __init__(self, input_nc, output_nc=5, keep_rate=0.75):
        super(Segmenter, self).__init__()
        layers = []
        current_nc = input_nc

        while current_nc > output_nc:
            next_nc = max(output_nc, current_nc // 2)  # 每次通道数减半，直到达到 output_nc
            layers.append(nn.Conv2d(current_nc, next_nc, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            current_nc = next_nc

        self.conv_layers = nn.Sequential(*layers)
        self.keep_rate = keep_rate

    def forward(self, x):
        x = self.conv_layers(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        return x


class DiscriminatorAux(nn.Module):
    def __init__(self, ndf=64, in_channels=3):
        super(DiscriminatorAux, self).__init__()
        self.ndf = ndf
        self.f = 4
        self.padw = 2

        self.c1 = nn.Conv2d(in_channels, ndf, kernel_size=self.f, stride=2, padding=0)
        self.c2 = nn.Conv2d(ndf, ndf * 2, kernel_size=self.f, stride=2, padding=0)
        self.c3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=self.f, stride=2, padding=0)
        self.c4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=self.f, stride=1, padding=0)
        self.c5 = nn.Conv2d(ndf * 8, 2, kernel_size=self.f, stride=1, padding=0)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.instance_norm = nn.InstanceNorm2d

    def forward(self, x):
        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw), "constant", 0)
        x = self.leaky_relu(self.c1(x))

        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw), "constant", 0)
        x = self.leaky_relu(self.instance_norm(ndf * 2)(self.c2(x)))

        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw), "constant", 0)
        x = self.leaky_relu(self.instance_norm(ndf * 4)(self.c3(x)))

        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw), "constant", 0)
        x = self.leaky_relu(self.instance_norm(ndf * 8)(self.c4(x)))

        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw), "constant", 0)
        x = self.c5(x)

        return x[:, 0:1, :, :], x[:, 1:2, :, :]

class Discriminator(nn.Module):
    def __init__(self, ndf=64,in_channels=3):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.f = 4
        self.padw = 2

        self.c1 = nn.Conv2d(in_channels, ndf, kernel_size=self.f, stride=2, padding=0)
        self.c2 = nn.Conv2d(ndf, ndf * 2, kernel_size=self.f, stride=2, padding=0)
        self.c3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=self.f, stride=2, padding=0)
        self.c4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=self.f, stride=1, padding=0)
        self.c5 = nn.Conv2d(ndf * 8, 1, kernel_size=self.f, stride=1, padding=0)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.instance_norm = nn.InstanceNorm2d

    def forward(self, x):
        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw), "constant", 0)
        x = self.leaky_relu(self.c1(x))

        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw), "constant", 0)
        x = self.leaky_relu(self.instance_norm(ndf * 2)(self.c2(x)))

        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw), "constant", 0)
        x = self.leaky_relu(self.instance_norm(ndf * 4)(self.c3(x)))

        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw), "constant", 0)
        x = self.leaky_relu(self.instance_norm(ndf * 8)(self.c4(x)))

        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw), "constant", 0)
        x = self.c5(x)

        return x


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 最后输出channel2的原因是第二个channel是辅助输出
        self.discriminator_aux_s = DiscriminatorAux(in_channels=1)
        self.discriminator_t = Discriminator(in_channels=1)
        self.discriminator_p = Discriminator(in_channels=5)
        self.discriminator_p_ll = Discriminator(in_channels=5)
        self.encoder = Encoder(input_nc=1, nef=16)
        self.decoder = Decoder(ngf=16)
        # generator_t: x_s -> x_s2t
        self.encoder_of_generator_t = Encoder(input_nc=1, nef=64)
        self.decoder_of_generator_t = Decoder(ngf=64)

        self.pixel_wise_classifier = Segmenter(input_nc=512, output_nc=5)
        self.pixel_wise_classifier_ll = Segmenter(input_nc=512, output_nc=5)

    def forward(self,images_a,images_b):
        prob_real_a_is_real, prob_real_a_aux_is_real = self.discriminator_aux_s(images_a)
        prob_real_b_is_real = self.discriminator_t(images_b)
        oc3,_ = self.encoder_of_generator_t(images_a)
        fake_images_b = self.decoder_of_generator_t(oc3,oc3)
        latent_b, latent_b_ll = self.encoder(images_b)
        fake_images_a = self.decoder(latent_b, images_b)
        pred_mask_b = self.pixel_wise_classifier(latent_b)
        pred_mask_b_ll = self.pixel_wise_classifier_ll(latent_b_ll)
        prob_fake_a_is_real, prob_fake_a_aux_is_real = self.discriminator_aux_s(fake_images_a)
        prob_fake_b_is_real = self.discriminator_t(fake_images_b)
        latent_fake_b, latent_fake_b_ll = self.encoder(fake_images_b)
        oc3_,_ = self.encoder_of_generator_t(fake_images_a)
        cycle_images_b = self.decoder_of_generator_t(oc3_,oc3_)
        cycle_images_a = self.decoder(latent_fake_b, fake_images_b)
        pred_mask_fake_b = self.pixel_wise_classifier(latent_fake_b)
        pred_mask_fake_b_ll = self.pixel_wise_classifier_ll(latent_fake_b_ll)
        prob_cycle_a_is_real, prob_cycle_a_aux_is_real = self.discriminator_aux_s(cycle_images_a)
        prob_pred_mask_fake_b_is_real = self.discriminator_p(pred_mask_fake_b)
        prob_pred_mask_b_is_real = self.discriminator_p(pred_mask_b)
        prob_pred_mask_fake_b_ll_is_real = self.discriminator_p_ll(pred_mask_fake_b_ll)
        prob_pred_mask_b_ll_is_real = self.discriminator_p_ll(pred_mask_b_ll)
        return {
            'prob_real_a_is_real': prob_real_a_is_real,
            'prob_real_b_is_real': prob_real_b_is_real,
            'prob_fake_a_is_real': prob_fake_a_is_real,
            'prob_fake_b_is_real': prob_fake_b_is_real,
            'cycle_images_a': cycle_images_a,
            'cycle_images_b': cycle_images_b,
            'fake_images_a': fake_images_a,
            'fake_images_b': fake_images_b,
            'pred_mask_b': pred_mask_b,
            'pred_mask_b_ll': pred_mask_b_ll,
            'pred_mask_fake_b': pred_mask_fake_b,
            'pred_mask_fake_b_ll': pred_mask_fake_b_ll,
            'prob_pred_mask_fake_b_is_real': prob_pred_mask_fake_b_is_real,
            'prob_pred_mask_b_is_real': prob_pred_mask_b_is_real,
            'prob_pred_mask_fake_b_ll_is_real': prob_pred_mask_fake_b_ll_is_real,
            'prob_pred_mask_b_ll_is_real': prob_pred_mask_b_ll_is_real,
            'prob_fake_a_aux_is_real': prob_fake_a_aux_is_real,
            'prob_cycle_a_aux_is_real': prob_cycle_a_aux_is_real,
            'prob_real_a_aux_is_real': prob_real_a_aux_is_real,
            'prob_cycle_a_is_real': prob_cycle_a_is_real
        }
