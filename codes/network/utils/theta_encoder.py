import torch
import torch.nn as nn
import numpy as np


class ThetaEncoder(nn.Module):

    def __init__(self, encoder_len):
        super(ThetaEncoder, self).__init__()
        self.encoder_len = encoder_len
        self.omega = 1

    def forward(self, theta):
        '''
        :param theta: [B, lead_num, 2]
        :return: [B, lead_num, 12]
        '''
        b, lead_num = theta.size(0), theta.size(1)
        sum_theta = theta[..., 0:1] + theta[..., 1:2]
        sub_theta = theta[..., 0:1] - theta[..., 1:2]
        before_encode = torch.cat([theta, sum_theta, sub_theta], dim=-1)
        out_all = [before_encode]
        
        sin = torch.sin(before_encode * self.omega)
        cos = torch.cos(before_encode * self.omega)
        out_all += [sin, cos]

        after_encode = torch.stack(out_all, dim=-1).view(b, lead_num, -1)
        return after_encode


if __name__ == '__main__':
    x = torch.tensor([[[np.pi / 2, np.pi / 3], [np.pi / 4, np.pi / 6]]])
    print(x.shape)
    net = ThetaEncoder(1)
    y = net(x)
    print(y.shape)
