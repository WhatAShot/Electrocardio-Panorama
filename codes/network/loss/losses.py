import torch
import torch.nn as nn


class OurLoss1(torch.nn.Module):
    def __init__(self):
        super(OurLoss1_v2, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, input0, input1):
        """
        :param input0: [B, 1, 512]
        :param input1: [B, 1, 512]
        :param target: [B, 1, 512]
        :return:
        """
        input0 = input0.detach()
        return self.l1_loss(input0, input1)


def losswrapper(predict, predict_shuffle_p, predict_shuffle_l, target, cfg, rest_out=None, rest_view=None,
                loss1_gt=None, loss2_gt=None):
    loss_f_1 = OurLoss1().cuda()

    # reg_loss choose
    if cfg.SOLVER.reg_loss == 'l2_loss':
        loss_f_2 = nn.MSELoss(reduction='mean').cuda()
    elif cfg.SOLVER.reg_loss == 'l1_loss':
        loss_f_2 = nn.L1Loss(reduction='mean').cuda()
    else:
        raise NotImplemented

    # loss1 loss2 which gt
    loss1_gt = predict if loss1_gt is None else loss1_gt
    loss2_gt = predict if loss2_gt is None else loss2_gt

    loss1 = loss_f_1(loss1_gt, predict_shuffle_p) if 1 in cfg.SOLVER.loss_using else 0.
    loss2 = loss_f_1(loss2_gt, predict_shuffle_l) if 2 in cfg.SOLVER.loss_using else 0.
    loss3 = loss_f_2(predict, target) if 3 in cfg.SOLVER.loss_using else 0.

    if rest_out is not None and rest_view is not None:
        loss_unsperv = loss_f_2(rest_out, rest_view)

    factor = cfg.SOLVER.loss_factor
    loss = loss1 * factor[0] + loss2 * factor[1] + loss3 * factor[2]

    if rest_out is not None and rest_view is not None:  # val
        return loss, loss1*factor[0], loss2 * factor[1], loss3 * factor[2], loss_unsperv
    else:  # train
        return loss, loss1*factor[0], loss2 * factor[1], loss3 * factor[2]


class MSELead(nn.Module):
    def __init__(self):
        super(MSELead, self).__init__()
        self.loss_func = nn.MSELoss()

    def forward(self, input, target):
        loss_list = []
        # print('mean:', torch.mean(target, dim=(0, 2)))
        # print('std:', torch.std(target, dim=(0, 2)))
        for i in range(input.size(1)):
            loss_list.append(self.loss_func(input[:, i], target[:, i]))
        return torch.mean(torch.stack(loss_list))


