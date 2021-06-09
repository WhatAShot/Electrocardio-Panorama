from dataset import build_dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from solver import Solver
from utils import seed_torch
import os
import torch


def main(cfg):
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_torch(seed=cfg.seed)

    output_dir = os.path.join(cfg.output_dir, cfg.desc)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')

    if cfg.DATA.weighted_sample:
        train_dl = DataLoader(train_dataset, batch_size=32,
                              sampler=WeightedRandomSampler(train_dataset.get_label_weight(),
                                                            num_samples=5000), num_workers=0,
                              drop_last=True)
    else:
        train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=32, num_workers=8, drop_last=True)

    solver = Solver(cfg)

    solver.train(train_dl, test_dl)


if __name__ == '__main__':
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

    main(cfg)
