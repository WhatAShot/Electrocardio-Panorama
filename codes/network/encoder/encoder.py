import torch
from torch import nn
from network.encoder import resnet_1d


class Encoder(nn.Module):
    def __init__(self, backbone='resnet34', in_channel=1, use_first_pool=True, lead_num=1, init_channels=64):
        """
        :param backbone Backbone network.
        :param num_layers number of resnet layers to use, 1-5
        :param upsample_interp Interpolation to use for upscaling latent code
        :param in_channel encoder in channel
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        """
        super().__init__()

        # define backbone
        model = getattr(resnet_1d, backbone)(in_channel=in_channel, lead_num=lead_num, init_channels=init_channels)
        self.conv1 = model.conv1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1

        self.use_first_pool = use_first_pool

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
        x = self.layer1(x)
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
