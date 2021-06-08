import torch
import torch.nn.functional as F


def roi_pooling(input, rois, size=8, spatial_scale=1.0):  # pytorch version use for loop !!!
    '''
    :param input: [B, channels, length]
    :param rois:  [B, 5, 2]
    :param size:  target pooling size
    :param spatial_scale:   input_length / ori_length
    :return: [B, channels, 5, size]
    '''
    assert rois.dim() == 3
    assert rois.size(-1) == 2
    ori_roi = rois
    output = []
    rois = rois.data.float()
    batch_size, num_rois = rois.size(0), rois.size(1)
    rois.mul_(spatial_scale)
    rois = rois.long()
    # print(rois)
    for i in range(batch_size):
        extract_rois = []
        for j in range(num_rois):
            roi = rois[i, j]
            im = input.narrow(0, i, 1)[..., roi[0]:roi[1] + 1]
            if im.shape[-1] != 0:
                extract_rois.append(F.adaptive_max_pool1d(im, size))
            else:
                print(rois[i, j])
                print(ori_roi[i, j])
                extract_rois.append(torch.empty(0).to(input.device))
        output.append(torch.cat(extract_rois))
    output = torch.stack(output)
    output = output.transpose(1, 2)
    return output


def roi_algin(input, rois, size=8, spatial_scale=1.0):  # pytorch version use for loop !!!
    '''
    :param input: [B, channels, length]
    :param rois:  [B, 5, 2]
    :param size:  target pooling size
    :param spatial_scale:   input_length / ori_length
    :return: [B, channels, 5, size]
    '''
    assert rois.dim() == 3
    assert rois.size(-1) == 2
    ori_roi = rois
    output = []
    rois = rois.data.float()
    batch_size, num_rois, length = rois.size(0), rois.size(1), input.size(2)
    rois.mul_(spatial_scale)
    rois = rois.mul_(2/length).add_(-1)  # projection to (-1, 1)
    gridX = []
    for i in range(batch_size):
        gridX_patient = []
        for j in range(num_rois):
            gridX_single = torch.linspace(rois[i, j, 0], rois[i, j, 1], steps=size)
            gridX_patient.append(gridX_single)
        gridX_patient = torch.stack(gridX_patient, dim=0)
        gridX.append(gridX_patient)

    gridX = torch.stack(gridX, dim=0)
    gridY = torch.zeros_like(gridX)
    grid = torch.stack([gridX, gridY], dim=3).type(gridX.type()).to(input.device)

    output = F.grid_sample(input.unsqueeze(-1), grid, align_corners=False)

    return output


def roi_pooling_reverse(input, rois, spatial_scale=1.):
    '''

    :param input: [B, channels, 5, size]
    :param rois:    [B, 5, 2]
    :param spatial_scale:  downsample ratio
    :return:    [B, channels, len]
    '''
    assert rois.dim() == 3
    assert rois.size(-1) == 2
    rois = rois.data.float()
    batch_size, num_rois = rois.size(0), rois.size(1)
    rois.mul_(spatial_scale)
    rois = rois.long()
    output = []
    for i in range(batch_size):
        reverse_rois = []
        for j in range(num_rois):
            roi = rois[i, j]
            im = input.narrow(0, i, 1)[..., j, :]
            roi_len = int(roi[1] - roi[0])
            if roi_len != 0:
                reverse_rois.append(F.interpolate(im, roi_len, mode='linear', align_corners=False))
            else:
                reverse_rois.append(torch.empty(0).to(input.device))
        output.append(torch.cat(reverse_rois, dim=-1))
    output = torch.cat(output, dim=0)
    return output


if __name__ == '__main__':
    x = torch.randn(2, 2, 16)
    roi = torch.LongTensor([[[1, 12], [2, 14]]])
    print(roi.shape)
    print(x, roi)
    output = roi_pooling(x, rois=roi)
    print(output.shape)
