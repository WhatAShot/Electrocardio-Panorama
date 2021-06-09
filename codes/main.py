from train_net import main
import argparse
from config import cfg
import os
import torch
import setproctitle

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='ecg generation')
parser.add_argument(
    '--config-file',
    default="",
    metavar="FILE",
    help="path to config file",
    type=str
)
parser.add_argument('--a', default="", metavar="FILE", help="path to config file", type=str)

args = parser.parse_args()

if args.config_file != '':
    cfg.merge_from_file(args.config_file)
print('Using config: ', cfg)
cfg.desc = args.config_file.split('/')[-1].replace('.yml', '')
cfg.output_dir = os.path.join(cfg.output_dir, cfg.desc)

setproctitle.setproctitle(cfg.desc)

main(cfg)
