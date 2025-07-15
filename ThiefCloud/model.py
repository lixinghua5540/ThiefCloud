import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from cd_model import DCNet_L1, CDNet_V1
import matplotlib.pyplot as plt
import os





class Downsample(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=to_2tuple(3), padding=1, bias=bias),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=to_2tuple(3), padding=1, bias=bias),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)



class toPixel(nn.Module):
    def __init__(self, dim, up_scale=2, bias=False):
        super(toPixel, self).__init__()       
        self.up = nn.PixelShuffle(up_scale)
        self.qk_pre = nn.Conv2d(int(dim // (up_scale ** 2)), 3, kernel_size=3, padding=1, bias=bias)
        

    def forward(self, x):
        qk = self.qk_pre(self.up(x))
        fake_image = qk

        return fake_image


def frozen(model):
    for p in model.parameters():
        p.requires_grad = False


## FI
class CC(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_mean = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.conv_std = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):

        # mean
        ca_mean = self.avg_pool(x)
        ca_mean = self.conv_mean(ca_mean)

        # std
        m_batchsize, C, height, width = x.size()
        x_dense = x.view(m_batchsize, C, -1)
        ca_std = torch.std(x_dense, dim=2, keepdim=True)
        ca_std = ca_std.view(m_batchsize, C, 1, 1)
        ca_var = self.conv_std(ca_std)

        # Coefficient of Variation
        # # cv1 = ca_std / ca_mean
        # cv = torch.div(ca_std, ca_mean)
        # ram = self.sigmoid(ca_mean + ca_var)

        cc = (ca_mean + ca_var)/2.0
        return cc

#AFE
class LatticeBlock(nn.Module):
    def __init__(self, nFeat, nDiff, nFeat_slice=0):
        super(LatticeBlock, self).__init__()

        self.D3 = nFeat
        self.d = nDiff
        self.s = nFeat_slice

        block_0 = []
        block_0.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=True))
        block_0.append(nn.LeakyReLU(0.05))
        block_0.append(nn.Conv2d(nFeat-nDiff, nFeat-nDiff, kernel_size=3, padding=1, bias=True))
        block_0.append(nn.LeakyReLU(0.05))
        block_0.append(nn.Conv2d(nFeat-nDiff, nFeat, kernel_size=3, padding=1, bias=True))
        block_0.append(nn.LeakyReLU(0.05))
        self.conv_block0 = nn.Sequential(*block_0)

        self.fea_ca1 = CC(nFeat)
        self.x_ca1 = CC(nFeat)

        block_1 = []
        block_1.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=True))
        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat-nDiff, nFeat-nDiff, kernel_size=3, padding=1, bias=True))
        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat-nDiff, nFeat, kernel_size=3, padding=1, bias=True))
        block_1.append(nn.LeakyReLU(0.05))
        self.conv_block1 = nn.Sequential(*block_1)

        self.fea_ca2 = CC(nFeat)
        self.x_ca2 = CC(nFeat)

        self.compress = nn.Conv2d(2 * nFeat, nFeat, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        # x:[12,256,256]
        # analyse unit
        x_feature_shot = self.conv_block0(x) #12,256,256
        fea_ca1 = self.fea_ca1(x_feature_shot) #12,1,1
        x_ca1 = self.x_ca1(x) #12,1,1

        p1z = x + fea_ca1 * x_feature_shot #12,256,256
        q1z = x_feature_shot + x_ca1 * x #12,256,256

        # synthes_unit
        x_feat_long = self.conv_block1(p1z) #12,256,256
        fea_ca2 = self.fea_ca2(q1z) #12,1,1
        p3z = x_feat_long + fea_ca2 * q1z
        x_ca2 = self.x_ca2(x_feat_long)
        q3z = q1z + x_ca2 * x_feat_long

        out = torch.cat((p3z, q3z), 1)
        out = self.compress(out)

        return out

 #

#PGFF
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, nhidden = 96):
        super().__init__()

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        ks = 3

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)



    def forward(self, x, segmap):

        # Part 2. produce scaling and bias conditioned on semantic map
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = x * (1 + gamma) + beta

        return out

    
class ThiefCloud(nn.Module):
    def __init__(self, in_channel=3, out_channel=4, dim=24, depths=(4, 4, 4, 2, 2), ckpt_path='', only_last=False, swich = False):
        super(ThiefCloud, self).__init__()

        self.swich = swich
        self.patch_embed = nn.Conv2d(in_channel, dim, kernel_size=to_2tuple(3), padding=1)

        #Replace the model file which has trained in simulation data
        ckpt_path = "/home/zaq/Dehazing/CDModel/checkpointsCDNet_V1/best_CDNet_V1_SateHaze1k_sim2.pth"


        #cloud detection model SCPM
        self.cd = CDNet_V1() 
        if not self.swich:
            frozen(self.cd)

        checkpoint_cd = torch.load(ckpt_path)
        model_dict_cd = self.cd.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint_cd["model_state"].items() if (k in model_dict_cd)}
        model_dict_cd.update(pretrained_dict)
        self.cd.load_state_dict(model_dict_cd, strict=True)


        self.down1 = Downsample(dim)


        self.down2 = Downsample(int(dim * 2 ** 1))


        self.up1 = Upsample(int(dim * 2 ** 2))


        self.up2 = Upsample(int(dim * 2 ** 1))


        self.conv_cat1 = nn.Sequential(
                  nn.Conv2d(12+dim*2*2, dim*2*2, kernel_size=3,  padding=1),
                  nn.ReLU(),
                )
        
        self.conv_cat2 = nn.Sequential(
            nn.Conv2d(12+dim*2*2, dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv_cat3 = nn.Sequential(
            nn.Conv2d(12+dim*2, dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )


        depth1 = 3
        self.Lblock1 = []
        for _ in range(depth1):
            self.Lblock1.append(LatticeBlock(dim, dim//4))
        self.Lblock1 = nn.Sequential(*self.Lblock1)

        depth2 = 2
        self.Lblock2 = []
        for _ in range(depth2):
            self.Lblock2.append(LatticeBlock(dim*2, dim*2//4))
        self.Lblock2 = nn.Sequential(*self.Lblock2)

        depth3 = 2
        self.Lblock3 = []
        for _ in range(depth3):
            self.Lblock3.append(LatticeBlock(dim*2*2, dim*2*2//4))
        self.Lblock3 = nn.Sequential(*self.Lblock3)

        self.Lblock_cd1 = LatticeBlock(12, 3)
        self.Lblock_cd2 = LatticeBlock(12, 3)
        self.Lblock_cd3 = LatticeBlock(12, 3)



        self.spade3 = SPADE(dim*2*2, dim*2*2, dim*2*2)
        self.spade2 = SPADE(dim*2, dim*2, dim*2)
        self.spade1 = SPADE(dim, dim, dim)

        self.tp1 = toPixel(dim,1)
        self.tp2 = toPixel(dim*2,2)
        self.tp3 = toPixel(dim*2*2,4)


    def forward_features(self, x, name=0):
        if not self.swich:
            with torch.no_grad():
                x_cd, x1_cd, x2_cd, x3_cd = self.cd(x)
        else:
            x_cd, x1_cd, x2_cd, x3_cd = self.cd(x)

        # (1,256,256) (12,256,256) (12,128,128) (12,64,64)

        x1_cd = self.Lblock_cd1(x1_cd)
        x2_cd = self.Lblock_cd2(x2_cd)
        x3_cd = self.Lblock_cd3(x3_cd)

        x1_d = self.patch_embed(x) #24
        x2_d = self.down1(x1_d) #48
        x3_d = self.down2(x2_d) #96

        x1_L = self.Lblock1(x1_d) + x1_d #24
        x2_L = self.Lblock2(x2_d) + x2_d #48
        x3_L = self.Lblock3(x3_d) + x3_d #96

        x3_haze = torch.cat([x3_L, x3_cd], dim=1)
        x3_haze = self.conv_cat1(x3_haze) #96
        x2_u = self.up1(x3_haze) #48

        x2_haze = torch.cat([x2_u, x2_cd, x2_L], dim=1) #48, 12, 48
        x2_haze = self.conv_cat2(x2_haze) #48
        x1_u = self.up2(x2_haze) #24

        x1_haze = torch.cat([x1_u, x1_cd, x1_L], dim=1) #24,12,24
        x1_haze = self.conv_cat3(x1_haze) #24

        x1_clear = self.spade1(x1_d, x1_haze)
        x2_clear = self.spade2(x2_d, x2_haze)
        x3_clear = self.spade3(x3_d, x3_haze)

        x1_clear = self.tp1(x1_clear)
        x2_clear = self.tp2(x2_clear)
        x3_clear = self.tp3(x3_clear)

        return x1_clear, x3_clear, x2_clear

    def forward(self, x, name=0,only_last=False):

        x, fake_image_x4, fake_image_x2 = self.forward_features(x, name)

        if only_last:
            # return x
            return x
        else:
            return x, fake_image_x4, fake_image_x2



if __name__ == '__main__':
    x = torch.randn((1, 3, 512, 512)).cuda()
    net = ThiefCloud().cuda()

    from thop import profile, clever_format
    flops, params = profile(net, (x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
