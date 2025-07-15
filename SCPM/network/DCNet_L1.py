import torch.nn as nn
from .unet_utils import unetConv2, unetUp
import torch.nn.functional as F
from models.networks_other import init_weights
import torch

# deformable conv import
from models.deform_part import deform_up, deform_down, deform_inconv
import math

class DCNet_L1(nn.Module):

    def __init__(self, feature_scale=8, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(DCNet_L1, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024] 
        filters = [int(x / self.feature_scale) for x in filters] 

        # downsampling
        self.down1 = deform_down(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.down2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.down3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.down4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[2], filters[3], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = deform_up(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):

        conv1 = self.down1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.down2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.down3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        # conv4 = self.down4(maxpool3)
        # maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool3)

        # up4 = self.up_concat4(conv3, center)
        up3 = self.up_concat3(conv3, center)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


def inter_upsample(x, scale_factor=2, mode="bilinear"):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


class CDNet_V1(nn.Module):

    def __init__(self, in_channel=3, out_channel=1, dim=24, feature_scale=4, is_batchnorm=True, is_deconv=False):
        super(CDNet_V1, self).__init__()
        self.out_channel = out_channel
        self.in_channels = in_channel
        self.dim = dim
        self.is_batchnorm = is_batchnorm
        self.is_deconv = is_deconv

        self.feature_scale = feature_scale

        filters = [2*dim, 4*dim, 8*dim, 16*dim, 32*dim] 
        filters = [int(x / self.feature_scale) for x in filters] 

        # downsampling
        self.down1 = deform_down(self.in_channels, filters[0])
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)

        self.down2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.down3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.avgpool3 = nn.AvgPool2d(kernel_size=2)

        self.center = unetConv2(filters[2], filters[3], self.is_batchnorm)

        # upsampling
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = deform_up(filters[1], filters[0], self.is_deconv)


        self.conv1 = nn.Sequential(
            nn.Conv2d(filters[2], filters[0], kernel_size=3, padding=1, bias=True),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1, bias=True),
            nn.ReLU())

        #simam
        self.simam = simam_module()

        # final conv (without any concat)
        self.final = nn.Conv2d(3*filters[0], self.out_channel, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):

        conv1 = self.down1(inputs) #12.256.256
        maxpool1 = self.avgpool1(conv1) # 12.128.128

        conv2 = self.down2(maxpool1) #24.128.128
        maxpool2 = self.avgpool2(conv2) #24.64.64

        conv3 = self.down3(maxpool2) #48.64.64
        maxpool3 = self.avgpool3(conv3) #48.32.32

        # conv4 = self.down4(maxpool3)
        # maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool3) #96.32.32

        # up4 = self.up_concat4(conv3, center)
        up3 = self.up_concat3(conv3, center) #48.64.64
        up2 = self.up_concat2(conv2, up3) #24.128.128
        up1 = self.up_concat1(conv1, up2) #12.256.256

        # #MLCA
        # up3 = self.mlca3(up3)
        # up2 = self.mlca2(up2)
        # up1 = self.mlca1(up1)

        #conv
        up3 = self.conv1(up3)
        up2 = self.conv2(up2)

        # simam
        up3 = self.simam(up3) #12,64,64
        up2 = self.simam(up2) #12,128,128
        up1 = self.simam(up1) #12,256,256

        up3_cat = inter_upsample(up3, scale_factor=4)
        up2_cat = inter_upsample(up2, scale_factor=2)
        up1_cat = torch.cat([up3_cat,up2_cat,up1],dim=1) #36.256.256

        final = self.final(up1_cat)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


# 一种注意力机制
class MLCA(nn.Module):
    def __init__(self, in_size=64, local_size=5, gamma=2, b=1, local_weight=0.5):
        super(MLCA, self).__init__()
 
        # ECA 计算方法
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)  # eca  gamma=2
        k = t if t % 2 else t + 1
 
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
 
        self.local_weight = local_weight
 
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)
 
    def forward(self, x):
        local_arv = self.local_arv_pool(x)
        global_arv = self.global_arv_pool(local_arv)
 
        b, c, m, n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape
 
        # (b,c,local_size,local_size) -> (b,c,local_size*local_size)-> (b,local_size*local_size,c)-> (b,1,local_size*local_size*c)
        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)
 
        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)
 
        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c,
                                                                                                            self.local_size,
                                                                                                            self.local_size)
        # y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)
        y_global_transpose = y_global.view(b, -1).unsqueeze(-1).unsqueeze(-1)  # 代码修正
        # print(y_global_transpose.size())
        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(), [self.local_size, self.local_size])
        # print(att_local.size())
        # print(att_global.size())
        att_all = F.adaptive_avg_pool2d(att_global * (1 - self.local_weight) + (att_local * self.local_weight), [m, n])
        # print(att_all.size())
        x = x * att_all
        return x


# 另一种注意力机制
class simam_module(torch.nn.Module):
    def __init__(self,  e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)












