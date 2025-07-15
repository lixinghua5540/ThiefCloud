import torch.nn as nn
from .unet_utils import unetConv2, unetUp
import torch.nn.functional as F
from models.networks_other import init_weights

# deformable conv import
from models.deform_part import deform_up, deform_down

class DCNet_L123(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(DCNet_L123, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]


        # downsampling
        self.down1 = deform_down(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.down2 = deform_down(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.down3 = deform_down(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.down4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = deform_up(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = deform_up(filters[2], filters[1], self.is_deconv)
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

        conv4 = self.down4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p













