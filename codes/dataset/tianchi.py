import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import wfdb
import numpy as np
import json
import random


class EcgTianChiDataset(Dataset):

    def __init__(self, cfg, phase, transform=None):
        self.dataset = None
        self.label_name = None
        self.cfg = cfg
        self.transform = transform
        self._read_data(cfg, phase)

    def _read_data(self, cfg, phase):
        all_set = pd.read_csv(cfg.DATA.train_label_path)

        self.label_name = all_set.columns.values[3:]
        self.data_root = cfg.DATA.train_data_root

        train_set, test_set = train_test_split(all_set, shuffle=True, test_size=0.2, random_state=cfg.seed)

        if phase == 'train':
            self.dataset = train_set
        elif phase == 'test':
            self.dataset = test_set

        self.label = self.dataset.iloc[:, 3:].values.astype(np.int)

    def __getitem__(self, index):
        file_path = os.path.join(self.data_root, self.dataset.iloc[index, 0])
        source_data = np.load(file_path).astype(np.float)
        label = self.label[index]
        if self.transform is not None:
            source_data = self.transform(source_data)
        return source_data, label

    def __len__(self):
        return len(self.dataset)


class EcgTianChiInterval(Dataset):

    def __init__(self, cfg, phase, transform=None):
        self.dataset = None
        self.label_name = None
        self.label_dir = None
        self.cfg = cfg
        self.phase = phase
        self.transform = transform
        self.theta = np.array([[np.pi / 2, np.pi / 2],  # I
                               [np.pi * 5 / 6, np.pi / 2],  # II
                               [np.pi / 2, -np.pi / 18],  # v1
                               [np.pi / 2, np.pi / 18],  # v2
                               [np.pi * (19/36), np.pi / 12],  # v3
                               [np.pi * (11/20), np.pi / 6],  # v4
                               [np.pi * (16/30), np.pi / 3],  # v5
                               [np.pi * (16/30), np.pi / 2],  # v6
                               [np.pi * (5/6), -np.pi / 2],  # III
                               [np.pi * (1/3), -np.pi / 2],  # aVR
                               [np.pi * (1/3), np.pi / 2],  # aVL
                               [np.pi * 1, np.pi / 2],  # aVF
                               ])
        self._read_data(cfg, phase)

    def _read_data(self, cfg, phase):
        label_path = cfg.DATA.train_label_path if phase == 'train' else cfg.DATA.test_label_path
        with open(label_path) as f:
            self.dataset = f.read().splitlines()
        # all_set = pd.read_csv(cfg.DATA.train_label_path)
        self.data_root = cfg.DATA.train_data_root
        self.label_dir = cfg.DATA.train_label_root

    def angle_jitter(self, angle, jitter_factor):
        jitter_angle = jitter_factor / 180 * np.pi
        jitter = np.random.normal(scale=jitter_angle, size=angle.shape)
        angle = angle + jitter

        return angle

    def __getitem__(self, index):
        file_path = os.path.join(self.data_root, self.dataset[index].replace('.json', '.npy'))
        source_data = np.load(file_path).astype(np.float)

        # add III, aVR, aVL, aVF. III = II - I, aVR = -0.5(I + II), aVL = I - 0.5II, aVF = II - 0.5I
        III = source_data[1:2, :] - source_data[0:1, :]
        aVR = - 0.5 * (source_data[0:1, :] + source_data[1:2, :])
        aVL = source_data[0:1, :] - 0.5 * source_data[1:2, :]
        aVF = source_data[1:2, :] - 0.5 * source_data[0:1, :]
        source_data = np.concatenate([source_data, III, aVR, aVL, aVF], axis=0)

        with open(os.path.join(self.label_dir, self.dataset[index])) as f:
            label = json.loads(f.read())
        random_index = random.sample(range(len(label['P on']) - 1), k=1)[0]

        p_on, p_off, r_on, r_off, t_on, t_off = label['P on'][random_index], label['P off'][random_index], \
                                                label['R on'][random_index], label['R off'][random_index], \
                                                label['T on'][random_index], label['T off'][random_index],
        end_point = label['P on'][random_index + 1] if random_index + 1 < len(label['P on']) else source_data.shape[-1]
        rois_list = np.array(
            [[p_on, p_off], [p_off, r_on], [r_on, r_off], [r_off, t_on], [t_on, t_off], [t_off, end_point],
             [end_point, 512 + p_on]])
        rois_list -= p_on
        source_data = source_data[:, p_on:end_point]

        # normalized
        max_, min_ = np.max(source_data), np.min(source_data)
        source_data = (source_data - min_) / (max_ - min_)

        # get noise
        noise_region = source_data[:, (rois_list[5][0] + rois_list[5][1]) // 2: rois_list[5][1]]
        noise_std = np.std(noise_region, axis=1)
        noise = np.random.normal(loc=0, scale=noise_std, size=(source_data.shape[-1], 12))

        # angle jitter
        theta_ = self.theta
        if self.cfg.MODEL.jitter_factor > 0 and self.phase == 'train':
            theta_ = self.angle_jitter(theta_, self.cfg.MODEL.jitter_factor)

        # 改成从8个监督导联中取input和ouput的监督，剩下4个导联不作监督
        supervision_lead_lamb, supervision_lead_chest = [2, 4, 6, 7], [0, 1, 8, 9]
        supervision_lead = supervision_lead_lamb + supervision_lead_chest
        unsupervision_lead = [x for x in range(0, 12) if x not in supervision_lead]
        if self.cfg.DATA.lead_num == 3:
            random_num_in_lamb = random.randint(1, 2)
            if self.cfg.DATA.train_data_mode == 'input_fix':
                if self.cfg.DATA.super_mode == '_354':
                    select_index = [1, 2, 6]
                elif self.cfg.DATA.super_mode == '_336':
                    select_index = [1, 2, 6]
                    supervision_lead = [2, 4, 6] + [0, 1, 8]
                    unsupervision_lead = [x for x in range(0, 12) if x not in supervision_lead]
                elif self.cfg.DATA.super_mode == '_390':
                    select_index = [0, 1, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index]
                    unsupervision_lead = []
                elif self.cfg.DATA.super_mode == '_3120':
                    select_index = [0, 1, 8]
                    supervision_lead = list(range(12))
                    unsupervision_lead = []
                elif self.cfg.DATA.super_mode == '_372':
                    select_index = [1, 2, 6]
                    unsupervision_lead = [4, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv3v4_372':
                    select_index = [1, 4, 5]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv1v6_372':
                    select_index = [1, 2, 7]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVLv5v6_372':
                    select_index = [10, 6, 7]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVLv1v4_372':
                    select_index = [10, 2, 5]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVRv3v4_372':
                    select_index = [9, 4, 5]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'Iv3v4_372':
                    select_index = [0, 4, 5]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv3v4_372':
                    select_index = [8, 4, 5]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVRv1v6_372':
                    select_index = [9, 2, 7]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'Iv1v6_372':
                    select_index = [0, 2, 7]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv1v6_372':
                    select_index = [8, 2, 7]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVLv3v4_372':
                    select_index = [4, 5, 10]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'Iv1v4_372':
                    select_index = [0, 2, 5]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv1v4_372':
                    select_index = [1, 2, 5]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv1v4_372':
                    select_index = [8, 2, 5]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVRv1v4_372':
                    select_index = [9, 2, 5]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'Iv5v6_372':
                    select_index = [0, 6, 7]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv5v6_372':
                    select_index = [1, 6, 7]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv5v6_372':
                    select_index = [8, 6, 7]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVRv5v6_372':
                    select_index = [9, 6, 7]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv5v6_v1aVF_372':
                    select_index = [8, 6, 7]
                    unsupervision_lead = [2, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv5v6_v3aVF_372':
                    select_index = [8, 6, 7]
                    unsupervision_lead = [4, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv5v6_v4aVF_372':
                    select_index = [8, 6, 7]
                    unsupervision_lead = [5, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv5v6_v2I_372':
                    select_index = [8, 6, 7]
                    unsupervision_lead = [3, 0]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv5v6_v2II_372':
                    select_index = [8, 6, 7]
                    unsupervision_lead = [3, 1]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv5v6_v2aVL_372':
                    select_index = [8, 6, 7]
                    unsupervision_lead = [3, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv5v6_v2aVR_372':
                    select_index = [8, 6, 7]
                    unsupervision_lead = [3, 9]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv3v4_v1aVF_372':
                    select_index = [8, 4, 5]
                    unsupervision_lead = [2, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv3v4_v5aVF_372':
                    select_index = [8, 4, 5]
                    unsupervision_lead = [6, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv3v4_v6aVF_372':
                    select_index = [8, 4, 5]
                    unsupervision_lead = [7, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv3v4_v2I_372':
                    select_index = [8, 4, 5]
                    unsupervision_lead = [3, 0]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv3v4_v2II_372':
                    select_index = [8, 4, 5]
                    unsupervision_lead = [3, 1]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv3v4_v2aVL_372':
                    select_index = [8, 4, 5]
                    unsupervision_lead = [3, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv3v4_v2aVR_372':
                    select_index = [8, 4, 5]
                    unsupervision_lead = [3, 9]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv1v6_v2aVF_372':
                    select_index = [8, 2, 7]
                    unsupervision_lead = [3, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv1v6_v3aVF_372':
                    select_index = [8, 2, 7]
                    unsupervision_lead = [4, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv1v6_v4aVF_372':
                    select_index = [8, 2, 7]
                    unsupervision_lead = [5, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv1v6_v2I_372':
                    select_index = [8, 2, 7]
                    unsupervision_lead = [3, 0]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv1v6_v2II_372':
                    select_index = [8, 2, 7]
                    unsupervision_lead = [3, 1]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv1v6_v2aVL_372':
                    select_index = [8, 2, 7]
                    unsupervision_lead = [3, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIIv1v6_v2aVR_372':
                    select_index = [8, 2, 7]
                    unsupervision_lead = [3, 9]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

                elif self.cfg.DATA.super_mode == 'aVFv2v5_v3I_372':
                    select_index = [11, 3, 6]
                    unsupervision_lead = [4, 0]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv2v5_v3II_372':
                    select_index = [11, 3, 6]
                    unsupervision_lead = [4, 1]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv2v5_v3III_372':
                    select_index = [11, 3, 6]
                    unsupervision_lead = [4, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv2v5_v3aVL_372':
                    select_index = [11, 3, 6]
                    unsupervision_lead = [4, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv2v5_v3aVR_372':
                    select_index = [11, 3, 6]
                    unsupervision_lead = [4, 9]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv2v5_v1II_372':
                    select_index = [11, 3, 6]
                    unsupervision_lead = [2, 1]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv2v5_v4II_372':
                    select_index = [11, 3, 6]
                    unsupervision_lead = [5, 1]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv2v5_v6II_372':
                    select_index = [11, 3, 6]
                    unsupervision_lead = [7, 1]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

                elif self.cfg.DATA.super_mode == 'IIv2v5_v3I_372':
                    select_index = [1, 3, 6]
                    unsupervision_lead = [4, 0]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv2v5_v3III_372':
                    select_index = [1, 3, 6]
                    unsupervision_lead = [4, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv2v5_v3aVL_372':
                    select_index = [1, 3, 6]
                    unsupervision_lead = [4, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv2v5_v3aVR_372':
                    select_index = [1, 3, 6]
                    unsupervision_lead = [4, 9]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv2v5_v3aVF_372':
                    select_index = [1, 3, 6]
                    unsupervision_lead = [4, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv2v5_v1III_372':
                    select_index = [1, 3, 6]
                    unsupervision_lead = [2, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv2v5_v4III_372':
                    select_index = [1, 3, 6]
                    unsupervision_lead = [5, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv2v5_v4I_372':
                    select_index = [1, 3, 6]
                    unsupervision_lead = [5, 0]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv2v5_v6III_372':
                    select_index = [1, 3, 6]
                    unsupervision_lead = [7, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

                elif self.cfg.DATA.super_mode == 'aVRv2v5_v3I_372':
                    select_index = [9, 3, 6]
                    unsupervision_lead = [4, 0]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVRv2v5_v3II_372':
                    select_index = [9, 3, 6]
                    unsupervision_lead = [4, 1]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVRv2v5_v3III_372':
                    select_index = [9, 3, 6]
                    unsupervision_lead = [4, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVRv2v5_v3aVL_372':
                    select_index = [9, 3, 6]
                    unsupervision_lead = [4, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVRv2v5_v3aVF_372':
                    select_index = [9, 3, 6]
                    unsupervision_lead = [4, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVRv2v5_v1III_372':
                    select_index = [9, 3, 6]
                    unsupervision_lead = [2, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVRv2v5_v4III_372':
                    select_index = [9, 3, 6]
                    unsupervision_lead = [5, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVRv2v5_v6III_372':
                    select_index = [9, 3, 6]
                    unsupervision_lead = [7, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

                elif self.cfg.DATA.super_mode == 'IIv1v3_v4I_372':
                    select_index = [1, 2, 4]
                    unsupervision_lead = [5, 0]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv1v3_v4III_372':
                    select_index = [1, 2, 4]
                    unsupervision_lead = [5, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv1v3_v4aVL_372':
                    select_index = [1, 2, 4]
                    unsupervision_lead = [5, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv1v3_v4aVR_372':
                    select_index = [1, 2, 4]
                    unsupervision_lead = [5, 9]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv1v3_v4aVF_372':
                    select_index = [1, 2, 4]
                    unsupervision_lead = [5, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv1v3_v2III_372':
                    select_index = [1, 2, 4]
                    unsupervision_lead = [3, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv1v3_v5III_372':
                    select_index = [1, 2, 4]
                    unsupervision_lead = [6, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv1v3_v6III_372':
                    select_index = [1, 2, 4]
                    unsupervision_lead = [7, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'IIv1v3_v6aVL_372':
                    select_index = [1, 2, 4]
                    unsupervision_lead = [7, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

                elif self.cfg.DATA.super_mode == 'aVFv1v2_v6I_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [7, 0]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v6II_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [7, 1]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v6III_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [7, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v6aVL_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [7, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v6aVR_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [7, 9]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v3aVL_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [4, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v4aVL_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [5, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v5aVL_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [6, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v6aVL_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [7, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v3I_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [4, 0]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v3II_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [4, 1]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v3III_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [4, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v3aVR_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [4, 9]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v4III_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [5, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv1v2_v5III_372':
                    select_index = [11, 2, 3]
                    unsupervision_lead = [6, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

                elif self.cfg.DATA.super_mode == 'aVFv5v6_v3I_372':
                    select_index = [11, 6, 7]
                    unsupervision_lead = [4, 0]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv5v6_v3II_372':
                    select_index = [11, 6, 7]
                    unsupervision_lead = [4, 1]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv5v6_v3III_372':
                    select_index = [11, 6, 7]
                    unsupervision_lead = [4, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv5v6_v3aVL_372':
                    select_index = [11, 6, 7]
                    unsupervision_lead = [4, 10]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv5v6_v3aVR_372':
                    select_index = [11, 6, 7]
                    unsupervision_lead = [4, 9]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv5v6_v1III_372':
                    select_index = [11, 6, 7]
                    unsupervision_lead = [2, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv5v6_v2III_372':
                    select_index = [11, 6, 7]
                    unsupervision_lead = [3, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'aVFv5v6_v4III_372':
                    select_index = [11, 6, 7]
                    unsupervision_lead = [5, 8]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]






                elif self.cfg.DATA.super_mode == '_381':
                    select_index = [1, 2, 6]
                    unsupervision_lead = [4]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'val_381':
                    select_index = [0, 1, 8]
                    unsupervision_lead = [4]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == 'val_372':
                    select_index = [0, 1, 8]
                    unsupervision_lead = [4, 11]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
                elif self.cfg.DATA.super_mode == '_fps':
                    select_index = [0, 1, 8]
                    supervision_lead = [7]
                    unsupervision_lead = []
                else:
                    select_index = random.sample([2, 6], random_num_in_lamb) + random.sample([0, 8],
                                                                                             3 - random_num_in_lamb)
            else:
                select_index = random.sample(supervision_lead_lamb, random_num_in_lamb) + random.sample(
                    supervision_lead_chest, 3 - random_num_in_lamb)
        elif self.cfg.DATA.lead_num == 12 and self.cfg.DATA.super_mode == '_12120':
            select_index = list(range(12))
            supervision_lead = list(range(12))
            unsupervision_lead = []
        elif self.cfg.DATA.lead_num == 9:
            supervision_lead = [0, 1, 3]
            select_index = [x for x in range(0, 12) if x not in supervision_lead]
            unsupervision_lead = []
        elif self.cfg.DATA.lead_num == 8 and self.cfg.DATA.super_mode == '_8120':
            select_index = list(range(8))
            supervision_lead = list(range(12))
            unsupervision_lead = []
        elif self.cfg.DATA.lead_num == 7:
            select_index = [2, 6, 7, 0, 1, 8, 9]  # v3导联作为监督不输入，即去掉idx 4
        elif self.cfg.DATA.lead_num == 4:
            select_index = [2, 6, 0, 8]
            if self.cfg.DATA.super_mode == '_480':
                supervision_lead = [x for x in range(0, 12) if x not in select_index]
                unsupervision_lead = []
            elif self.cfg.DATA.super_mode == '_462':
                unsupervision_lead = [4, 11]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
        elif self.cfg.DATA.lead_num == 5:
            if self.cfg.DATA.super_mode == '_552':
                select_index = [2, 6, 0, 8, 10]
                unsupervision_lead = [4, 11]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == '_561':
                select_index = [2, 6, 0, 8, 10]
                unsupervision_lead = [4]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == '_cr_561':
                select_index = [2, 6, 0, 8, 10]
                unsupervision_lead = [11]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

            elif self.cfg.DATA.super_mode == 'IIaVLv1v3v5_v6I_552':
                select_index = [1, 10, 2, 4, 6]
                unsupervision_lead = [7, 0]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv1v3v5_v6III_552':
                select_index = [1, 10, 2, 4, 6]
                unsupervision_lead = [7, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv1v3v5_v6aVR_552':
                select_index = [1, 10, 2, 4, 6]
                unsupervision_lead = [7, 9]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv1v3v5_v6aVF_552':
                select_index = [1, 10, 2, 4, 6]
                unsupervision_lead = [7, 11]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv1v3v5_v2III_552':
                select_index = [1, 10, 2, 4, 6]
                unsupervision_lead = [3, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv1v3v5_v4III_552':
                select_index = [1, 10, 2, 4, 6]
                unsupervision_lead = [5, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

            elif self.cfg.DATA.super_mode == 'IIaVLv1v2v3_v6I_552':
                select_index = [1, 10, 2, 3, 4]
                unsupervision_lead = [7, 0]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv1v2v3_v6III_552':
                select_index = [1, 10, 2, 3, 4]
                unsupervision_lead = [7, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv1v2v3_v6aVR_552':
                select_index = [1, 10, 2, 3, 4]
                unsupervision_lead = [7, 9]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv1v2v3_v6aVF_552':
                select_index = [1, 10, 2, 3, 4]
                unsupervision_lead = [7, 11]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv1v2v3_v5III_552':
                select_index = [1, 10, 2, 3, 4]
                unsupervision_lead = [6, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv1v2v3_v4III_552':
                select_index = [1, 10, 2, 3, 4]
                unsupervision_lead = [5, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

            elif self.cfg.DATA.super_mode == 'IIaVFv1v2v3_v4I_552':
                select_index = [1, 11, 2, 3, 4]
                unsupervision_lead = [5, 0]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVFv1v2v3_v4III_552':
                select_index = [1, 11, 2, 3, 4]
                unsupervision_lead = [5, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVFv1v2v3_v4aVR_552':
                select_index = [1, 11, 2, 3, 4]
                unsupervision_lead = [5, 9]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVFv1v2v3_v4aVL_552':
                select_index = [1, 11, 2, 3, 4]
                unsupervision_lead = [5, 10]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVFv1v2v3_v5III_552':
                select_index = [1, 11, 2, 3, 4]
                unsupervision_lead = [6, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVFv1v2v3_v6III_552':
                select_index = [1, 11, 2, 3, 4]
                unsupervision_lead = [7, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

            elif self.cfg.DATA.super_mode == 'aVLaVFv4v5v6_v3I_552':
                select_index = [10, 11, 5, 6, 7]
                unsupervision_lead = [4, 0]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'aVLaVFv4v5v6_v3II_552':
                select_index = [10, 11, 5, 6, 7]
                unsupervision_lead = [4, 1]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'aVLaVFv4v5v6_v3III_552':
                select_index = [10, 11, 5, 6, 7]
                unsupervision_lead = [4, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'aVLaVFv4v5v6_v3aVR_552':
                select_index = [10, 11, 5, 6, 7]
                unsupervision_lead = [4, 9]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'aVLaVFv4v5v6_v1III_552':
                select_index = [10, 11, 5, 6, 7]
                unsupervision_lead = [2, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'aVLaVFv4v5v6_v2III_552':
                select_index = [10, 11, 5, 6, 7]
                unsupervision_lead = [3, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

            elif self.cfg.DATA.super_mode == 'IIaVLv2v4v5_v3I_552':
                select_index = [1, 10, 3, 5, 6]
                unsupervision_lead = [4, 0]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv2v4v5_v3III_552':
                select_index = [1, 10, 3, 5, 6]
                unsupervision_lead = [4, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv2v4v5_v3aVR_552':
                select_index = [1, 10, 3, 5, 6]
                unsupervision_lead = [4, 9]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv2v4v5_v3aVF_552':
                select_index = [1, 10, 3, 5, 6]
                unsupervision_lead = [4, 11]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv2v4v5_v1III_552':
                select_index = [1, 10, 3, 5, 6]
                unsupervision_lead = [2, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'IIaVLv2v4v5_v6III_552':
                select_index = [1, 10, 3, 5, 6]
                unsupervision_lead = [7, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]





            elif self.cfg.DATA.super_mode == '_570':
                select_index = [2, 6, 0, 8, 10]
                unsupervision_lead = []
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
        elif self.cfg.DATA.lead_num == 2:
            select_index = [1, 6]
            if self.cfg.DATA.super_mode == '_228':
                supervision_lead = [1, 6, 9, 3]
                unsupervision_lead = [x for x in range(0, 12) if x not in supervision_lead]
            elif self.cfg.DATA.super_mode == '_2100':
                supervision_lead = [x for x in range(0, 12) if x not in select_index]
                unsupervision_lead = []
        elif self.cfg.DATA.lead_num == 1:
            select_index = [1]  # II lead as input, other 7 leads as supervision
            if self.cfg.DATA.super_mode == '_1110':
                supervision_lead = [x for x in range(0, 12) if x not in select_index]
                unsupervision_lead = []
            elif self.cfg.DATA.super_mode == '_1101':
                unsupervision_lead = [4]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == '_192':
                unsupervision_lead = [4, 11]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'val_192':
                unsupervision_lead = [4, 3]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

            elif self.cfg.DATA.super_mode == 'II_v3I_192':
                select_index = [1]
                unsupervision_lead = [4, 0]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'II_v3III_192':
                select_index = [1]
                unsupervision_lead = [4, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'II_v3aVL_192':
                select_index = [1]
                unsupervision_lead = [4, 10]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'II_v3aVR_192':
                select_index = [1]
                unsupervision_lead = [4, 9]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'II_v3aVF_192':
                select_index = [1]
                unsupervision_lead = [4, 11]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'II_v1III_192':
                select_index = [1]
                unsupervision_lead = [2, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'II_v2III_192':
                select_index = [1]
                unsupervision_lead = [3, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'II_v4III_192':
                select_index = [1]
                unsupervision_lead = [5, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'II_v5III_192':
                select_index = [1]
                unsupervision_lead = [6, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'II_v6III_192':
                select_index = [1]
                unsupervision_lead = [7, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

            elif self.cfg.DATA.super_mode == 'v3_v5I_192':
                select_index = [4]
                unsupervision_lead = [6, 0]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v3_v5II_192':
                select_index = [4]
                unsupervision_lead = [6, 1]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v3_v5III_192':
                select_index = [4]
                unsupervision_lead = [6, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v3_v5aVL_192':
                select_index = [4]
                unsupervision_lead = [6, 10]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v3_v5aVR_192':
                select_index = [4]
                unsupervision_lead = [6, 9]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v3_v5aVF_192':
                select_index = [4]
                unsupervision_lead = [6, 11]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v3_v1III_192':
                select_index = [4]
                unsupervision_lead = [2, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v3_v2III_192':
                select_index = [4]
                unsupervision_lead = [3, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v3_v4III_192':
                select_index = [4]
                unsupervision_lead = [5, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v3_v6III_192':
                select_index = [4]
                unsupervision_lead = [7, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

            elif self.cfg.DATA.super_mode == 'v2_v3I_192':
                select_index = [3]
                unsupervision_lead = [4, 0]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v2_v3II_192':
                select_index = [3]
                unsupervision_lead = [4, 1]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v2_v3III_192':
                select_index = [3]
                unsupervision_lead = [4, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v2_v3aVL_192':
                select_index = [3]
                unsupervision_lead = [4, 10]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v2_v3aVR_192':
                select_index = [3]
                unsupervision_lead = [4, 9]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v2_v3aVF_192':
                select_index = [3]
                unsupervision_lead = [4, 11]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v2_v1III_192':
                select_index = [3]
                unsupervision_lead = [2, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v2_v4III_192':
                select_index = [3]
                unsupervision_lead = [5, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v2_v5III_192':
                select_index = [3]
                unsupervision_lead = [6, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v2_v6III_192':
                select_index = [3]
                unsupervision_lead = [7, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]

            elif self.cfg.DATA.super_mode == 'v5_v3I_192':
                select_index = [6]
                unsupervision_lead = [4, 0]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v5_v3II_192':
                select_index = [6]
                unsupervision_lead = [4, 1]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v5_v3III_192':
                select_index = [6]
                unsupervision_lead = [4, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v5_v3aVL_192':
                select_index = [6]
                unsupervision_lead = [4, 10]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v5_v3aVR_192':
                select_index = [6]
                unsupervision_lead = [4, 9]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v5_v3aVF_192':
                select_index = [6]
                unsupervision_lead = [4, 11]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v5_v1III_192':
                select_index = [6]
                unsupervision_lead = [2, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v5_v2III_192':
                select_index = [6]
                unsupervision_lead = [3, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v5_v4III_192':
                select_index = [6]
                unsupervision_lead = [5, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
            elif self.cfg.DATA.super_mode == 'v5_v6III_192':
                select_index = [6]
                unsupervision_lead = [7, 8]
                supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]




            elif self.cfg.DATA.super_mode == '_mit':
                mode_choice = random.choice([2, 3])  # 1 means 2->2, 2 means 5->2, 3 means 2->5
                if mode_choice == 2:
                    select_index, supervision_lead = [6], [1]
                elif mode_choice == 3:
                    select_index, supervision_lead = [1], [6]
                unsupervision_lead = []
        else:
            raise KeyError("WORANG lead num: {}".format(self.cfg.DATA.lead_num))
        # if self.cfg.DATA.lead_num == 6:
        #     select_index = [0, 2, 4, 6, 8, 10]  # 正常采样，不抽样
        # elif self.cfg.DATA.lead_num == 1:
        #     select_index = [1]
        # else:
        #     raise KeyError("WORANG lead num")
        rest_index = [x for x in supervision_lead if x not in select_index] if self.cfg.DATA.super_mode not in ['_12120', '_3120', '_8120'] else supervision_lead
        target_index = random.sample(rest_index, 1)[0]
        target_view, target_theta, target_noise = source_data[target_index], theta_[target_index], noise[:, target_index]
        rest_index += unsupervision_lead  # 把非监督的导联加在rest的最后
        rest_view, rest_theta = source_data[rest_index], theta_[rest_index]
        input_theta = theta_[select_index]
        unsuper_view, unsuper_theta = source_data[unsupervision_lead], theta_[unsupervision_lead]

        data = source_data[select_index, ...]
        data = np.pad(data, ((0, 0), (0, 512 - data.shape[-1])), mode='constant') if data.shape[-1] < 512 else data[:,
                                                                                                               :512]
        target_view = np.pad(target_view, (0, 512 - target_view.shape[-1]), mode='constant') if target_view.shape[
                                                                                                    -1] < 512 else target_view[
                                                                                                                   :512]
        source_data = np.pad(source_data, ((0, 0), (0, 512 - source_data.shape[-1])), mode='constant') if \
            source_data.shape[-1] < 512 else source_data[:, :512]
        rest_view = np.pad(rest_view, ((0, 0), (0, 512 - rest_view.shape[-1])), mode='constant') if \
            rest_view.shape[-1] < 512 else rest_view[:, :512]
        unsuper_view = np.pad(unsuper_view, ((0, 0), (0, 512 - unsuper_view.shape[-1])), mode='constant') if \
            unsuper_view.shape[-1] < 512 else unsuper_view[:, :512]
        target_noise = np.pad(target_noise, (0, 512 - target_noise.shape[-1]), mode='constant')
        meta = {
            'data': data.astype(np.float32),
            'rois': np.array(rois_list).astype(np.int),
            'input_theta': np.array(input_theta).astype(np.float32),
            'target_view': np.array(target_view).astype(np.float32),
            'target_theta': np.array(target_theta).astype(np.float32),
            'id': self.dataset[index],
            'ori_data': source_data,
            'rest_view': rest_view,
            'rest_theta': np.array(rest_theta).astype(np.float32),
            # 'unsuper_view': unsuper_view,
            # 'unsuper_theta': np.array(unsuper_theta).astype(np.float32),
            'noise': target_noise.astype(np.float32),
            'unsupervision_lead_name': unsupervision_lead,
        }
        return meta

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from config import cfg

    # cfg.DATA.train_label_path = '/data/yhy/project/ecg_generation/dataset/valid_jsons.txt'
    # cfg.DATA.train_data_root = '/data/share/ecg_data/npy_data/tianchi_train_round1'
    # cfg.DATA.train_label_root = '/data/yhy/project/ecg_generation/output/tianchi_interval'

    dataset = EcgTianChiInterval(cfg, 'train', transform=None)
    print(len(dataset))
    dl = DataLoader(dataset, batch_size=32, shuffle=True)
    for index, meta in enumerate(dl):
        print(meta['data'].shape, meta['rois'].shape, meta['input_theta'].shape, meta['target_view'].shape,
              meta['target_theta'].shape, meta['ori_data'].shape)
