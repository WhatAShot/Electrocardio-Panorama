from dataset import *
from torch.utils.data import DataLoader
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

    test_dataset = build_dataset(cfg, phase='test')

    test_dl = DataLoader(test_dataset, batch_size=32, num_workers=8, drop_last=True)

    solver = Solver(cfg, use_tensorboardx=False)
    with torch.no_grad():
        solver.val(test_dl, epoch=args.epoch)


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
    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--ds', default='tianchi', type=str)

    args = parser.parse_args()
    if args.config_file != '':
        cfg.merge_from_file(args.config_file)

    cfg.desc = args.config_file.split('/')[-1].replace('.yml', '')
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.desc)

    main(cfg)
