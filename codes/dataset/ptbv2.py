import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import wfdb
import numpy as np
import json
import random
import pickle
from tqdm import tqdm
import logging


class PTBV2(Dataset):
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
                               [np.pi * (19 / 36), np.pi / 12],  # v3
                               [np.pi * (11 / 20), np.pi / 6],  # v4
                               [np.pi * (16 / 30), np.pi / 3],  # v5
                               [np.pi * (16 / 30), np.pi / 2],  # v6
                               [np.pi * (5 / 6), -np.pi / 2],  # III
                               [np.pi * (1 / 3), -np.pi / 2],  # aVR
                               [np.pi * (1 / 3), np.pi / 2],  # aVL
                               [np.pi * 1, np.pi / 2],  # aVF
                               ])
        self._read_data(cfg, phase)

    def _read_data(self, cfg, phase):
        pkl_path = cfg.DATA.train_pkl_path if phase == 'train' else cfg.DATA.test_pkl_path
        label_path = cfg.DATA.train_label_path if phase == 'train' else cfg.DATA.test_label_path
        self.dataset = HeartBeatList(label_path, cfg.DATA.train_data_root, pkl_path).heart_beats

    def __getitem__(self, index):
        source_data, rois_list = self.dataset[index].data, self.dataset[index].rois_list

        source_data = np.concatenate([source_data[0:2], source_data[6:], source_data[2:6]], axis=0)

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
                if self.cfg.DATA.super_mode == 'IIv2v5_v4I_372':
                    select_index = [1, 3, 6]
                    unsupervision_lead = [5, 0]
                    supervision_lead = [x for x in range(0, 12) if x not in select_index + unsupervision_lead]
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
        else:
            raise KeyError("WORANG lead num: {}".format(self.cfg.DATA.lead_num))
        rest_index = [x for x in supervision_lead if x not in select_index] if self.cfg.DATA.super_mode not in ['_12120', '_3120', '_8120'] else supervision_lead
        target_index = random.sample(rest_index, 1)[0]
        target_view, target_theta, target_noise = source_data[target_index], theta_[target_index], noise[:, target_index]
        rest_index += unsupervision_lead  # unsupervision_lead in the end of rest
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
        target_noise = np.pad(target_noise, (0, 512 - target_noise.shape[-1]), mode='constant')
        meta = {
            'data': data.astype(np.float32),
            'rois': np.array(rois_list).astype(np.int),
            'input_theta': np.array(input_theta).astype(np.float32),
            'target_view': np.array(target_view).astype(np.float32),
            'target_theta': np.array(target_theta).astype(np.float32),
            'ori_data': source_data,
            'rest_view': rest_view,
            'rest_theta': np.array(rest_theta).astype(np.float32),
            'noise': target_noise.astype(np.float32),
            'unsupervision_lead_name': unsupervision_lead,
        }
        return meta

    def angle_jitter(self, angle, jitter_factor):
        jitter_angle = jitter_factor / 180 * np.pi
        jitter = np.random.normal(scale=jitter_angle, size=angle.shape)
        angle = angle + jitter

        return angle

    def __len__(self):
        return len(self.dataset)


class HeartBeatList(object):
    def __init__(self, txt_path, data_root, pkl_path):
        self.heart_beats = []
        if os.path.exists(pkl_path):
            self.load_from_pkl(pkl_path)
        else:
            self.read_data(txt_path, data_root)
            self.save_pkl(pkl_path)

    def read_data(self, txt_path, data_root):
        with open(txt_path) as f:
            dataset = f.read().splitlines()
        for patient_name in tqdm(dataset):
            patient_dir = os.path.join(data_root, patient_name)
            file_name_list = [x for x in os.listdir(patient_dir) if x.endswith('.json')]
            for file_name in file_name_list:
                file_path = os.path.join(patient_dir, file_name.replace('.json', '.npy'))
                source_data = np.load(file_path).astype(np.float)
                with open(os.path.join(patient_dir, file_name)) as f:
                    label = json.loads(f.read())
                for index in range(len(label['P on']) - 1):
                    p_on, p_off, r_on, r_off, t_on, t_off = label['P on'][index], label['P off'][index], \
                                                            label['R on'][index], label['R off'][index], \
                                                            label['T on'][index], label['T off'][index]
                    end_point = label['P on'][index + 1]
                    rois_list = np.array(
                        [[p_on, p_off], [p_off, r_on], [r_on, r_off], [r_off, t_on], [t_on, t_off], [t_off, end_point],
                         [end_point, 512 + p_on]])
                    rois_list -= p_on

                    split_data = source_data[:, p_on:end_point]
                    hb = HeartBeat(split_data, rois_list)
                    self.heart_beats.append(hb)

    def load_from_pkl(self, pkl_path):
        logging.info("Loading tianchi hearbeats from pickle...")
        with open(pkl_path, 'rb') as handle:
            self.heart_beats = pickle.load(handle)
            logging.info("Loaded successfully...")

    def save_pkl(self, pkl_path):
        logging.info('Saving tianchi hearbeats from pickle...')
        with open(pkl_path, 'wb') as output:
            pickle.dump(self.heart_beats, output, pickle.HIGHEST_PROTOCOL)


class HeartBeat(object):
    def __init__(self, data, rois_list):
        self.data = data
        self.rois_list = rois_list


if __name__ == '__main__':
    train_txt_path = '/data/yhy/project/ecg_generation/dataset/ptb_train.txt'
    test_txt_path = '/data/yhy/project/ecg_generation/dataset/ptb_test.txt'
    data_root = '/data/share/ecg_data/ptb-diag_preprocess'
    train_pkl_path = '/data/share/ecg_data/ptb_pkl_data/train_ptb.pkl'
    test_pkl_path = '/data/share/ecg_data/ptb_pkl_data/test_ptb.pkl'
    ds_train = HeartBeatList(train_txt_path, data_root, train_pkl_path)
    ds_test = HeartBeatList(test_txt_path, data_root, test_pkl_path)
    #
    from config import cfg
    from config import cfg
    import argparse

    parser = argparse.ArgumentParser(description='ecg generation')
    parser.add_argument(
        '--config-file',
        default="",
        metavar="FILE",
        help="path to config file",
        type=str
    )

    args = parser.parse_args()
    if args.config_file != '':
        cfg.merge_from_file(args.config_file)

    dataset = PTBV2(cfg, 'train', transform=None)

    dl = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    for index, meta in enumerate(dl):
        print(meta['data'].shape, meta['rois'].shape, meta['input_theta'].shape, meta['target_view'].shape,
              meta['target_theta'].shape, meta['ori_data'].shape)
