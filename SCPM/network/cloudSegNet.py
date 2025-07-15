import torch.nn as nn
from .unet_utils import unetConv2, cloudSegNetUp
import torch.nn.functional as F
from models.networks_other import init_weights

class cloudSegNet(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=False, in_channels=3, is_batchnorm=True):
        super(cloudSegNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [16, 8, 8, 8, 8]

        # downsampling
        self.down1 = unetConv2(self.in_channels, filters[0],  self.is_batchnorm, n=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.down2 = unetConv2(filters[0], filters[1], self.is_batchnorm, n=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.down3 = unetConv2(filters[1], filters[2], self.is_batchnorm, n=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # upsampling
        self.up_concat3 = cloudSegNetUp(filters[3], filters[2], self.is_deconv, n=1)
        self.up_concat2 = cloudSegNetUp(filters[2], filters[1], self.is_deconv, n=1)
        self.up_concat1 = cloudSegNetUp(filters[1], filters[0], self.is_deconv, n=1)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=5, padding=2)

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

        up3 = self.up_concat3(maxpool3)
        up2 = self.up_concat2(up3)
        up1 = self.up_concat1(up2)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p













