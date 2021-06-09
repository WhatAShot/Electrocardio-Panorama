from .tianchi import EcgTianChiInterval
from .ptbv2 import PTBV2, HeartBeat


def build_dataset(cfg, phase):
    if cfg.DATA.dataset == 'tianchi':
        return EcgTianChiInterval(cfg, phase)
    elif cfg.DATA.dataset == 'ptbv2':
        cfg.DATA.train_pkl_path = '/data/share/ecg_data/ptb_pkl_data/train_ptb.pkl'
        cfg.DATA.test_pkl_path = '/data/share/ecg_data/ptb_pkl_data/test_ptb.pkl'
        cfg.DATA.train_label_path = '/data/yhy/project/ecg_generation/dataset/ptb_train.txt'
        cfg.DATA.test_label_path = '/data/yhy/project/ecg_generation/dataset/ptb_test.txt'
        cfg.DATA.train_data_root = '/data/share/ecg_data/ptb-diag_preprocess'
        return PTBV2(cfg, phase)
    else:
        raise NotImplemented("{} is not support".format(cfg.DATA.dataset))

