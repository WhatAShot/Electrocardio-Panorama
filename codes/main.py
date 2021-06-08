# import fitlog
from train_net import main
import argparse
from config import cfg
from utils import seed_torch
import os
import torch
import setproctitle
from dataset import *

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

# fitlog.commit(__file__, fit_msg=cfg.desc)  # auto commit your codes
# fitlog.set_log_dir("logs/")
# fitlog.add_hyper_in_file(__file__)  # record your hyperparameters

setproctitle.setproctitle(cfg.desc)
# seed_torch(seed=cfg.seed)

# try:
#     main(cfg)
# except KeyboardInterrupt:
#     fitlog.finish()

main(cfg)

# fitlog.finish()

# a = input()
