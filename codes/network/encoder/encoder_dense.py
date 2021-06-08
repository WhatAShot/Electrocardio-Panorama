import torch
from torch import nn
from network.encoder import resnet_1d


class Encoder_dense(nn.Module):
    def __init__(self, backbone='resnet34', in_channel=1, use_first_pool=True, lead_num=1, init_channels=64):
        """
        add dense block
        :param backbone Backbone network.
        :param num_layers number of resnet layers to use, 1-5
        :param upsample_interp Interpolation to use for upscaling latent code
        :param in_channel encoder in channel
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        """
        super().__init__()
        self.lead_num = lead_num

        # define backbone
        model = getattr(resnet_1d, backbone)(in_channel=in_channel, lead_num=lead_num, init_channels=init_channels)
        self.conv1 = model.conv1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1

        self.use_first_pool = use_first_pool

        self.conv_1 = nn.Conv1d(128*2*self.lead_num, 128*self.lead_num, 1, groups=self.lead_num)

    def forward(self, x):
        '''
        :param x:   [B, num_lead, 512]
        :return:    [B, out_channels, 128]
        '''
        # ori_size = x.size(-1)

        x = self.conv1(x)
        x = self.relu(x)
        if self.use_first_pool:
            x = self.maxpool(x)

        dense_res = x
        x = self.layer1(x)

        # dense connect block
        x_list = torch.chunk(x, self.lead_num, dim=1)
        dense_res_list = torch.chunk(dense_res, self.lead_num, dim=1)
        feat_list = []
        for i in range(self.lead_num):
            feat_list.append(x_list[i])
            feat_list.append(dense_res_list[i])
        x = self.conv_1(torch.cat(feat_list, dim=1))

        return x


if __name__ == '__main__':
    x = torch.randn(8, 1, 512)
    rois = []
    for i in range(8):
        roi, _ = torch.randint(0, 512, [10]).sort()
        roi = roi.view(5, 2)
        rois.append(roi)
    rois = torch.stack(rois)
    net = Encoder('resnet34', in_channel=1, use_first_pool=True, lead_num=1, init_channels=128)
    y1 = net(x)
    print(y1.shape)
