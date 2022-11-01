from network.encoder import Encoder
from network.utils.theta_encoder import ThetaEncoder
from network.utils.roi_pooling_1d import roi_pooling, roi_pooling_reverse, roi_algin

import torch
import torch.nn as nn
import random


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=groups)
        self.residual_conv = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, groups=groups)
        self.stride = stride
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        if out.shape[1] != residual.shape[1]:  # 若通道数不一致
            residual = self.residual_conv(residual)

        out += residual
        out = self.relu(out)

        return out


class Model_nefnet2(nn.Module):
    '''
    Nef-Net2 implement
    '''

    def __init__(self, theta_encoder_len=1, lead_num=1):
        super(Model_nefnet2, self).__init__()

        self.lead_num = lead_num

        self.W_encoder = Encoder(backbone='resnet34', in_channel=1, use_first_pool=True, lead_num=1,
                                 init_channels=128)
        self.theta_encoder = ThetaEncoder(encoder_len=theta_encoder_len)

        self.mlp1 = nn.Linear((2*theta_encoder_len+1)*4, 128)
        self.mlp2 = nn.Linear((2*theta_encoder_len+1)*4, 256 * 1)

        self.w_feature_extractor = nn.Sequential(
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.w_conv = nn.Sequential(
            BasicBlock(128, 128, 1),
        )

        self.z1_conv = nn.Sequential(
            BasicBlock(64, 128, 1)
        )
        self.z2_conv1 = nn.Sequential(
            BasicBlock(64, 128, 1)
        )
        self.z2_conv2 = nn.Sequential(
            BasicBlock(128 * 7, 128 * 7, 1, groups=7),
            nn.ConvTranspose1d(128 * 7, 128 * 7 // 2,
                               kernel_size=2, stride=2, groups=7),
            BasicBlock(128 * 7 // 2, 128 * 7, 1, groups=7),
        )

        self.single_conv_z1 = nn.Sequential(
            nn.Conv1d(128, 128, 3, 1, 1)
        )
        self.single_conv_z2 = nn.Sequential(
            nn.Conv1d(128, 128, 3, 1, 1)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            DoubleConv(256 * 1, 128),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            DoubleConv(128, 64),
            nn.Conv1d(64, 1, 3, padding=1)
        )

    def forward(self, x, input_thetas, query_theta, rois, rest_theta=None, phase='train'):
        '''
        :param x:   [B, lead_num, 512]
        :param input_thetas:  [B, lead_num, 2]
        :param query_theta:   [B, 2]
        :param rois:    [B, 5 * 2]
        :return:
        '''
        z1_list, z2_list = [], []
        for i in range(self.lead_num):
            x_single = x[:, i:i+1, ...]
            theta_single = input_thetas[:, i:i+1, ...]

            w = self.W_encoder(x_single)  # [B, 128, 128]
            theta_single = self.theta_encoder(theta_single)  # [B, 1, 12]

            encoded_theta = self.mlp1(theta_single)  # [B, 1, 128]
            w = encoded_theta[:, 0][:, :, None] * w  # [B, 128, 128]
            w = self.w_conv(w)  # [B, 128, 128]

            z1, z2 = torch.chunk(w, 2, dim=1)

            z1 = self.z1_conv(z1)   # [B, 128, 128]
            z1 = self.single_conv_z1(z1)  # [B, 128, 128]
            z2 = self.z2_conv1(z2)  # [B, 128, 128]

            z2 = roi_algin(z2, rois, size=16, spatial_scale=128 / 512)    # [B, 128, 7, 16]
            z2 = z2.contiguous().view(z2.size(0), z2.size(1) * z2.size(2), -1)  # [B, 128 * 7, 16]
            z2 = self.z2_conv2(z2).view(z2.size(0), 128, 7, 16*2)   # [B, 128 * lead_num, 7, 32] transconv

            z2 = roi_pooling_reverse(z2, rois, spatial_scale=128 / 512)     # [B, 128 * lead_num, 128]
            z2 = self.single_conv_z2(z2)  # [B, 128 * lead_num, 128]

            z1_list.append(z1)
            z2_list.append(z2)

        # get mean of z1 z2
        z1_mean = torch.mean(torch.stack(z1_list, dim=0), dim=0)  # [B, 128, 128]
        z2_mean = torch.mean(torch.stack(z2_list, dim=0), dim=0)  # [B, 128, 128]

        latent_all = torch.cat([z1_mean, z2_mean], dim=1)  # [B, 256 * 1, 128]

        if phase == 'gen':  # get z2 latent before roi reverse
            return z1_mean, z2_mean

        # change z1 to shuffle choice
        lead_choice_z1 = random.randint(0, self.lead_num-1)
        shuffle_z1 = z1_list[lead_choice_z1]
        lead_choice_z2 = random.randint(0, self.lead_num-1)
        shuffle_z2 = z2_list[lead_choice_z2]

        shuffle_patient_all = torch.cat([shuffle_z1, z2_mean], dim=1)
        shuffle_lead_all = torch.cat([z1_mean, shuffle_z2], dim=1)

        # get pred, patient shuffle, lead shuffle
        query_theta = self.theta_encoder(query_theta)  # [B, 12]
        query_theta = self.mlp2(query_theta.view(query_theta.size(0), -1))  # [B, 256]

        query_w = query_theta[:, :, None] * latent_all  # [B, 256, 128]
        out = self.decoder(query_w)  # [B, 1, 512]
        out = torch.sigmoid(out / 3)

        query_w_shuffle_p = query_theta[:, :, None] * shuffle_patient_all
        query_w_shuffle_p = self.decoder(query_w_shuffle_p)
        query_w_shuffle_p = torch.sigmoid(query_w_shuffle_p / 3)

        query_w_shuffle_l = query_theta[:, :, None] * shuffle_lead_all
        query_w_shuffle_l = self.decoder(query_w_shuffle_l)
        query_w_shuffle_l = torch.sigmoid(query_w_shuffle_l / 3)

        if phase == 'train':
            return out, query_w_shuffle_p, query_w_shuffle_l

        elif phase in ['val', 'test']:
            rest_theta = self.theta_encoder(rest_theta)
            rest_theta = self.mlp2(rest_theta)  # [B, 256]
            rest_out_all = []
            for i in range(rest_theta.shape[1]):
                rest_w = rest_theta[:, i, :, None] * latent_all  # [B, 256, 128]
                rest_out = self.decoder(rest_w)  # [B, 1, 512]
                rest_out = torch.sigmoid(rest_out / 3)
                rest_out_all.append(rest_out)
            rest_out = torch.cat(rest_out_all, dim=1)

            return out, query_w_shuffle_p, query_w_shuffle_l, rest_out
        else:
            raise KeyError("please type correct phase")

    def gen_ecg(self, z1, z2, query_theta, rois):
        self.eval()
        z2 = roi_pooling_reverse(z2, rois, spatial_scale=128 / 512)     # [B, 128 * lead_num, 128]

        # get mean of z1 z2
        z1_list = torch.chunk(z1, self.lead_num, dim=1)
        z2_list = torch.chunk(z2, self.lead_num, dim=1)
        z1_mean = torch.mean(torch.stack(z1_list, dim=0), dim=0)  # [B, 64, 128]
        z2_mean = torch.mean(torch.stack(z2_list, dim=0), dim=0)  # [B, 64, 128]

        latent_all = torch.cat([z1_mean, z2_mean], dim=1)
        query_theta = self.theta_encoder(query_theta)  # [B, 12]
        query_theta = self.mlp2(query_theta)  # [B, 256]

        rest_out_all = []
        for i in range(query_theta.shape[1]):
            rest_w = query_theta[:, i, :, None] * latent_all  # [B, 256, 128]
            rest_out = self.decoder(rest_w)  # [B, 1, 512]
            rest_out = torch.sigmoid(rest_out / 3)
            rest_out_all.append(rest_out)
        rest_out = torch.cat(rest_out_all, dim=1)

        return rest_out

