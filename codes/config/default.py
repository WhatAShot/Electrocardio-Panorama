from yacs.config import CfgNode as Node
import os

cfg = Node()
cfg.seed = 123
cfg.fit_msg = 'None'
cfg.output_dir = '{your folder}'
cfg.latent_save_dir = '{your folder}'
cfg.desc = 'model_v2_tianchi'

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
cfg.DATA = Node()
cfg.DATA.dataset = 'tianchi'
cfg.DATA.train_label_path = 'data/tianchi/tianchi_train_jsons.txt'
cfg.DATA.test_label_path = 'data/tianchi/tianchi_test_jsons.txt'
cfg.DATA.train_data_root = 'data/tianchi/npy_data/tianchi_train_round1'
cfg.DATA.train_label_root = 'data/tianchi/tianchi_interval'
cfg.DATA.train_pkl_path = 'data/tianchi/npy_data/pkl_data/train_heartbeats.pkl'
cfg.DATA.test_pkl_path = 'data/tianchi/npy_data/pkl_data/test_heartbeats.pkl'
cfg.DATA.noise_std = [4.37258895, 4.73799667, 5.00643047, 6.7582663, 6.57354042, 6.31023917, 6.05944371, 7.05612394]
cfg.DATA.lead_num = 1
cfg.DATA.noise = False
cfg.DATA.train_data_mode = 'normal'
cfg.DATA.super_mode = "normal"
cfg.DATA.weighted_sample = False


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
cfg.MODEL = Node()
cfg.MODEL.model = 'modelv2'
cfg.MODEL.resume = ''
cfg.MODEL.loss = 'v1'
cfg.MODEL.jitter_factor = 0.0
cfg.MODEL.theta_L = 1


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
cfg.SOLVER = Node()
cfg.SOLVER.optim = 'sgd'
cfg.SOLVER.scheduler = 'steplr'
cfg.SOLVER.lr_step = [150, 350]
cfg.SOLVER.lr = 1e-3
cfg.SOLVER.epochs = 500

cfg.SOLVER.OurLoss1_version = 'v2'
cfg.SOLVER.reg_loss = 'l1_loss'
cfg.SOLVER.loss_using = [1, 2, 3]
cfg.SOLVER.part_loss_no_grad = False
cfg.SOLVER.loss_factor = [1, 1, 1]

