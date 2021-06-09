from network import build_model, build_loss
from solver.optim_scheduler import get_optimizer, get_lr_scheduler
from utils.mertic import SSIM, PSNR
from utils import CheckPointer
from time import time, sleep
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tensorboardX
import torchprof


class Solver:
    def __init__(self, cfg, use_tensorboardx=True):
        self.cfg = cfg
        self.output_dir = os.path.join(cfg.output_dir, cfg.desc)
        self.desc = cfg.desc
        self.model = build_model(cfg).float()
        self.loss = build_loss(cfg)
        self._init_model_device()
        if self.desc != 'debug' and use_tensorboardx:
            self.summary_writer = tensorboardX.SummaryWriter(logdir=os.path.join(cfg.output_dir, 'tf_logs'))
        else:
            self.summary_writer = None

    def _init_model_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            if torch.cuda.device_count() > 1:
                print('Using', torch.cuda.device_count(), 'GPUs')
                self.model = nn.DataParallel(self.model)
            else:
                print('Using Single GPU')
        else:
            self.device = torch.device('cpu')
            print('cuda is not available, using cpu')
        self.model.to(self.device)

    def write_tensorboardx(self, scalars, names, epoch):
        for i in range(len(scalars)):
            self.summary_writer.add_scalar(names[i], scalars[i], global_step=epoch)

    def train(self, dl_train, dl_test):
        optimizer = get_optimizer(self.cfg, self.model.parameters())
        scheduler = get_lr_scheduler(self.cfg, optimizer)
        checkpointer = CheckPointer(self.model, optimizer, scheduler, self.output_dir)
        extra_checkpoint_data = checkpointer.load(self.cfg.MODEL.resume)
        max_epochs = self.cfg.SOLVER.epochs

        start_epoch = extra_checkpoint_data['epoch'] if 'epoch' in extra_checkpoint_data.keys() else 0
        best_test_loss = extra_checkpoint_data[
            'best_test_loss'] if 'best_test_loss' in extra_checkpoint_data.keys() else 999.
        best_test_psnr_gen = extra_checkpoint_data[
            'best_test_psnr_gen'] if 'best_test_psnr_gen' in extra_checkpoint_data.keys() else 0.
        print('the latest best_test_psnr_gen is {:06f}'.format(best_test_psnr_gen))

        start_time = time()
        save_arguments = {}

        for epoch in range(start_epoch, max_epochs):
            print('---------------------------------{}---{}-------------------------------------'.format(self.cfg.desc,
                                                                                                         epoch))
            train_losses, train_gt_views, train_predict_views, train_input_views, _, _ = self.run_one_epoch(dl_train,
                                                                                                         phase='train',
                                                                                                         optim=optimizer)
            scheduler.step()
            test_losses, test_gt_views, test_predict_views, test_input_views, mertics_all, _, mertics_gen_singlelead = self.run_one_epoch(dl_test,
                                                                                                               phase='test')
            train_loss_all = np.mean(train_losses, axis=0)[0]
            test_loss_all = np.mean(test_losses, axis=0)[0]
            train_loss1, train_loss2, train_loss3 = np.mean(train_losses, axis=0)[1], np.mean(train_losses, axis=0)[2], \
                                                    np.mean(train_losses, axis=0)[3]
            test_loss1, test_loss2, test_loss3, test_loss_unsuperv = np.mean(test_losses, axis=0)[1], \
                                                                     np.mean(test_losses, axis=0)[2], \
                                                                     np.mean(test_losses, axis=0)[3], \
                                                                     np.mean(test_losses, axis=0)[4]
            psnr_gen, psnr_reg, ssim_gen, ssim_reg = np.mean(mertics_all, axis=0)[0], np.mean(mertics_all, axis=0)[1], \
                                                     np.mean(mertics_all, axis=0)[2], np.mean(mertics_all, axis=0)[3]

            if self.desc != 'debug':
                scalars = [train_loss_all, test_loss_all, train_loss1, test_loss1,
                           train_loss2, test_loss2, train_loss3, test_loss3, test_loss_unsuperv,
                           psnr_gen, psnr_reg, ssim_gen, ssim_reg]
                names = ['train_loss_all', 'test_loss_all', 'train_loss_1', 'test_loss_1',
                         'train_loss_2', 'test_loss_2', 'train_3', 'test_3', 'test_unsuperv',
                         'psnr_gen', 'psnr_reg', 'ssim_gen', 'ssim_reg']

                # add scalar for single gen lead
                if len(mertics_gen_singlelead) != 0:
                    mertics_gen_singlelead = np.array(mertics_gen_singlelead)
                    for i in range(mertics_gen_singlelead.shape[1]):
                        names.append('psnr_reg_lead_{}'.format(i))
                        scalars.append(np.mean(mertics_gen_singlelead[:, i, 0]))
                        names.append('ssim_reg_lead_{}'.format(i))
                        scalars.append(np.mean(mertics_gen_singlelead[:, i, 1]))
                        print('psnr_reg_lead_{}'.format(i), np.mean(mertics_gen_singlelead[:, i, 0]),
                              'ssim_reg_lead_{}'.format(i), np.mean(mertics_gen_singlelead[:, i, 1]))
                self.write_tensorboardx(scalars, names, epoch)

            print('Epoch {}: train_loss: {}, test_loss: {}'.format(epoch, train_loss_all, test_loss_all))
            print('psnr_gen: {}, psnr_reg: {}, ssim_gen:{}, ssim_reg:{}'.format(psnr_gen, psnr_reg, ssim_gen, ssim_reg))

            # save pth every epoch
            save_arguments['psnr_gen'] = psnr_gen
            save_arguments['psnr_reg'] = psnr_reg
            save_arguments['epoch'] = epoch
            checkpointer.save('epoch_{}'.format(epoch), **save_arguments)

            if psnr_gen > best_test_psnr_gen:

                best_test_psnr_gen = psnr_gen
                save_arguments['best_test_psnr_gen'] = best_test_psnr_gen
                save_arguments['epoch'] = epoch
                checkpointer.save('best_valid', **save_arguments)

    def val(self, dl_test, epoch=-1):
        self.model.eval()
        optimizer = get_optimizer(self.cfg, self.model.parameters())
        scheduler = get_lr_scheduler(self.cfg, optimizer)
        checkpointer = CheckPointer(self.model, optimizer, scheduler, self.output_dir)
        if epoch == -1:
            extra_checkpoint_data = checkpointer.load(best_valid=True)
        else:
            extra_checkpoint_data = checkpointer.load(os.path.join(self.output_dir, 'epoch_{}.pkl'.format(epoch)))
        start_epoch = extra_checkpoint_data['epoch'] if 'epoch' in extra_checkpoint_data.keys() else 0
        best_test_psnr_gen = extra_checkpoint_data[
            'best_test_psnr_gen'] if 'best_test_psnr_gen' in extra_checkpoint_data.keys() else 0.
        print('the latest best_test_psnr_gen is {:06f} of epoch {}'.format(best_test_psnr_gen, start_epoch))

        test_losses, test_gt_views, test_predict_views, test_input_views, mertics_all, rois_all, mertics_gen_singlelead \
            = self.run_one_epoch(dl_test, phase='test')
        psnr_gen, psnr_reg, ssim_gen, ssim_reg = np.mean(mertics_all, axis=0)[0], np.mean(mertics_all, axis=0)[1], \
                                                 np.mean(mertics_all, axis=0)[2], np.mean(mertics_all, axis=0)[3]

        print('psnr_gen:{}, psnr_reg:{}, ssim_gen:{}, ssim_reg:{}'.format(psnr_gen, psnr_reg, ssim_gen, ssim_reg))

        # print('======saving======')
        # np.savez(os.path.join(self.output_dir, 'pred_gt_input_rois.npz'),
        #          gt=test_gt_views, pred=test_predict_views, input=test_input_views, rois=rois_all)
        # print('painting...')
        # self.paint(test_gt_views[::100], test_predict_views[::100], test_input_views[::100], start_epoch, 'inference')

    def run_one_epoch(self, dl, phase, optim=None):
        if phase == 'train':
            self.model.train()
        elif phase == 'test':
            self.model.eval()
        else:
            raise ValueError('phase param not found.')
        losses = []
        gt_views = []
        predict_views = []
        ori_data_views = []
        input_views = []
        rest_views = []
        mertics_all = []
        mertics_gen_singlelead = []
        rois_all = []
        nums = 0
        times_all = []
        for meta in tqdm(dl):
            source_data, rois, input_theta, target_view, target_theta, ori_data, noise = meta['data'].to(self.device), \
                                                                                         meta['rois'].to(self.device), \
                                                                                         meta['input_theta'].to(
                                                                                             self.device), \
                                                                                         meta['target_view'].unsqueeze(
                                                                                             1).to(self.device), \
                                                                                         meta['target_theta'].to(
                                                                                             self.device), \
                                                                                         meta['ori_data'], \
                                                                                         meta['noise'].unsqueeze(1).to(
                                                                                             self.device)
            rest_view, rest_theta = meta['rest_view'].to(self.device), meta['rest_theta'].to(self.device)
            unsuper_L_name = meta['unsupervision_lead_name']
            # with torchprof.Profile(self.model, use_cuda=True, profile_memory=True) as prof:
                # time_start = time()
            result = self.model(source_data, input_theta, target_theta, rois, rest_theta=rest_theta, phase=phase)
                # sleep(1)
            #     time_end = time()
            # trace, event_lists_dict = prof.raw()
            # # print(trace[2])
            # print(event_lists_dict[trace[2].path][0])
            # print(time_end - time_start)
            # times_all.append(time_end - time_start)
            # continue

            if phase == 'train':
                if self.cfg.MODEL.model in ['modeld7_3', 'modeld7_4', 'modeld7_5']:
                    out, query_w_shuffle_p, query_w_shuffle_l, loss1_gt, loss2_gt = result
                else:
                    out, query_w_shuffle_p, query_w_shuffle_l = result
                    loss1_gt, loss2_gt = None, None
                predict_views += [x for x in out.squeeze().cpu().detach().numpy()]
            else:
                out, query_w_shuffle_p, query_w_shuffle_l, rest_out = result
                predict_views += [x for x in rest_out.squeeze().cpu().detach().numpy()]
            # if self.cfg.DATA.noise:
            #     out = out + noise
            if phase == 'train':
                if self.cfg.DATA.noise:
                    out = out + noise
                loss, loss1, loss2, loss3 = self.loss(out, query_w_shuffle_p, query_w_shuffle_l, target_view,
                                                      self.cfg, loss1_gt=loss1_gt, loss2_gt=loss2_gt)
                losses.append([loss.item(), loss1.item(), loss2.item(), loss3.item()])
            else:
                loss, loss1, loss2, loss3, loss_unsperv = self.loss(out, query_w_shuffle_p, query_w_shuffle_l,
                                                                    target_view, self.cfg, rest_out[:, -4:, :],
                                                                    rest_view[:, -4:, :])
                losses.append([loss.item(), loss1.item(), loss2.item(), loss3.item(), loss_unsperv.item()])
                ecg_point_len = rest_out.shape[-1]

                gen_num = 6 if self.cfg.DATA.lead_num == 336 else 4
                if self.cfg.DATA.super_mode != 'normal':
                    gen_num = eval(self.cfg.DATA.super_mode[-1])
                if self.cfg.DATA.dataset == 'mit' or self.cfg.DATA.super_mode[-1] == '0' \
                        or self.cfg.DATA.super_mode == '_mit':
                    psnr_gen = PSNR(rest_out.contiguous().cpu().detach().numpy(),
                                    rest_view.contiguous().cpu().detach().numpy(),)
                    ssim_gen = SSIM(rest_out.contiguous().cpu().detach().numpy(),
                                    rest_view.contiguous().cpu().detach().numpy(),)
                    psnr_reg, ssim_reg = psnr_gen, ssim_gen
                else:
                    psnr_gen = PSNR(rest_out[:, -gen_num:, :].contiguous().cpu().detach().numpy(),
                                    rest_view[:, -gen_num:, :].contiguous().cpu().detach().numpy(),
                                    rois.cpu().detach().numpy())
                    psnr_reg = PSNR(rest_out[:, :-gen_num, :].contiguous().cpu().detach().numpy(),
                                    rest_view[:, :-gen_num, :].contiguous().cpu().detach().numpy(),
                                    rois.cpu().detach().numpy())
                    ssim_gen = SSIM(rest_out[:, -gen_num:, :].contiguous().cpu().detach().numpy(),
                                    rest_view[:, -gen_num:, :].contiguous().cpu().detach().numpy(),
                                    rois.cpu().detach().numpy())
                    ssim_reg = SSIM(rest_out[:, :-gen_num, :].contiguous().cpu().detach().numpy(),
                                    rest_view[:, :-gen_num, :].contiguous().cpu().detach().numpy(),
                                    rois.cpu().detach().numpy())

                    # cal psnr and ssim for single target gen lead
                    unsuper_out, unsuper_view = rest_out[:, -gen_num:, :].contiguous().cpu().detach().numpy(), \
                                                rest_view[:, -gen_num:, :].contiguous().cpu().detach().numpy(),
                    mertics_gen_singlelead_iter = []
                    for i in range(gen_num):
                        psnr_single = PSNR(unsuper_out[:, i:i+1, :], unsuper_view[:, i:i+1, :], rois.cpu().detach().numpy())
                        ssim_single = SSIM(unsuper_out[:, i:i+1, :], unsuper_view[:, i:i+1, :], rois.cpu().detach().numpy())
                        mertics_gen_singlelead_iter.append([psnr_single, ssim_single])
                        # TODO: complete the single lead gen psnr calation
                    mertics_gen_singlelead.append(mertics_gen_singlelead_iter)
                mertics_all.append([psnr_gen, psnr_reg, ssim_gen, ssim_reg])

            if phase == 'train':
                loss.backward()
                optim.step()
                optim.zero_grad()
            ori_data_views += [x for x in ori_data.cpu().detach().numpy()]
            gt_views += [x for x in target_view.squeeze().cpu().detach().numpy()]
            # predict_views += [x for x in result.squeeze().cpu().detach().numpy()]
            input_views += [x for x in source_data.cpu().detach().numpy()]
            rest_views += [x for x in rest_view.cpu().detach().numpy()]
            rois_all += [x for x in rois.cpu().detach().numpy()]

            # break
        # print('all mean time is {}'.format(np.mean(times_all)))
        if phase == 'train':
            return losses, gt_views, predict_views, input_views, mertics_all, rois_all
        else:
            return losses, rest_views, predict_views, input_views, mertics_all, rois_all, mertics_gen_singlelead

    def paint(self, target, pred, input_data=None, epoch=None, flag='train', dataset_name='tianchi'):
        # print(input_data[0].shape)
        import matplotlib
        matplotlib.rcParams.update({'figure.max_open_warning': 0})
        matplotlib.use('Agg')

        lead_all_num = 2 if dataset_name == 'mit' or self.cfg.DATA.super_mode == '_mit' else 12
        # target = [x[None, :] for x in target] if dataset_name == 'mit' else target
        pred = [x[None, :] for x in pred] if dataset_name == 'mit' or self.cfg.DATA.super_mode == '_mit' else pred
        total_plot_num = (lead_all_num - self.cfg.DATA.lead_num) + self.cfg.DATA.lead_num  # 本来前面括号有*2的
        for i in range(len(target)):
            plt.figure(figsize=(32, 3 * total_plot_num))
            if flag == 'train':
                plt.subplot(total_plot_num, 1, 1)
                plt.plot(target[i])
                # plt.subplot(total_plot_num, 1, 2)
                plt.plot(pred[i], color='orange')
                for j in range(self.cfg.DATA.lead_num):
                    plt.subplot(total_plot_num, 1, j + 2)
                    plt.plot(input_data[i][j])
            else:
                for j in range(lead_all_num - self.cfg.DATA.lead_num):
                    plt.subplot(total_plot_num, 1, 1 + j)
                    plt.plot(target[i][j, :])
                    # plt.subplot(total_plot_num, 1, 2+j*2)
                    plt.plot(pred[i][j, :], color='orange')
                for j in range(self.cfg.DATA.lead_num):
                    plt.subplot(total_plot_num, 1, j + (lead_all_num - self.cfg.DATA.lead_num) + 1)
                    plt.plot(input_data[i][j])
            os.makedirs(os.path.join(self.output_dir, '{}_{}'.format(epoch, flag)), exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, '{}_{}'.format(epoch, flag), str(i) + '.png'),
                        format='png')
            plt.close()

    def paint_for_other_method(self, target, pred, input_data=None, epoch=None, flag='train'):
        '''
        :param target: [B, 7, 512]
        :param pred: [B, 7, 512]
        :param input_data: [B, 1, 512]
        :param epoch:
        :param flag:
        :return:
        '''
        import matplotlib
        matplotlib.rcParams.update({'figure.max_open_warning': 0})
        matplotlib.use('Agg')
        total_plot_num = target.shape[1]
        for i in range(len(target)):
            plt.figure(figsize=(32, 3 * total_plot_num))
            for ind, (t, p) in enumerate(zip(target[i], pred[i])):
                plt.subplot(total_plot_num, 2, 2 * ind + 1)
                plt.plot(t)
                plt.subplot(total_plot_num, 2, 2 * ind + 2)
                plt.plot(p)
            os.makedirs(os.path.join(self.output_dir, '{}_{}'.format(epoch, flag)), exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, '{}_{}'.format(epoch, flag), str(i) + '.png'),
                        format='png')
            plt.close()

    def paint_for_mit(self, target, pred, input_data=None, epoch=None, flag='train'):
        '''
        :param target: [B, 7, 512]
        :param pred: [B, 7, 512]
        :param input_data: [B, 1, 512]
        :param epoch:
        :param flag:
        :return:
        '''
        import matplotlib
        matplotlib.rcParams.update({'figure.max_open_warning': 0})
        matplotlib.use('Agg')
        total_plot_num = target.shape[1]
        for i in range(len(target)):
            plt.figure(figsize=(32, 3 * total_plot_num))
            for ind, (t, p) in enumerate(zip(target[i], pred[i])):
                plt.subplot(total_plot_num, 2, 2 * ind + 1)
                plt.plot(t)
                plt.subplot(total_plot_num, 2, 2 * ind + 2)
                plt.plot(p)
            os.makedirs(os.path.join(self.output_dir, '{}_{}'.format(epoch, flag)), exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, '{}_{}'.format(epoch, flag), str(i) + '.png'),
                        format='png')
            plt.close()

    def train_for_other_method(self, dl_train, dl_test):
        optimizer = get_optimizer(self.cfg, self.model.parameters())
        scheduler = get_lr_scheduler(self.cfg, optimizer)
        checkpointer = CheckPointer(self.model, optimizer, scheduler, self.output_dir)
        extra_checkpoint_data = checkpointer.load(self.cfg.MODEL.resume)
        max_epochs = self.cfg.SOLVER.epochs
        start_epoch = extra_checkpoint_data['epoch'] if 'epoch' in extra_checkpoint_data.keys() else 0
        best_test_loss = extra_checkpoint_data[
            'best_test_loss'] if 'best_test_loss' in extra_checkpoint_data.keys() else 999.
        best_test_psnr_gen = extra_checkpoint_data[
            'best_test_psnr_gen'] if 'best_test_psnr_gen' in extra_checkpoint_data.keys() else 0.
        print('the latest best_test_psnr_gen is {:06f}'.format(best_test_psnr_gen))

        start_time = time()
        save_arguments = {}

        for epoch in range(start_epoch, max_epochs):
            print('---------------------------------{}---{}-------------------------------------'.format(self.cfg.desc,
                                                                                                         epoch))
            train_metrics, train_gt_views, train_pred_views, train_input_views = self.run_one_epochs_for_other_method(
                dl_train, 'train',
                optim=optimizer)
            print('Train metrics: ', train_metrics)
            # fitlog.add_metric(train_metrics, step=epoch)
            # fitlog.add_loss(train_metrics['loss'], step=epoch, name='train')
            test_metrics, test_gt_views, test_pred_views, test_input_views = self.run_one_epochs_for_other_method(
                dl_test, phase='test',
                optim=optimizer)
            print('Test metrics: ', test_metrics)
            # fitlog.add_metric(test_metrics, step=epoch)
            # fitlog.add_loss(test_metrics['loss'], step=epoch, name='test')
            if test_metrics['psnr'] > best_test_psnr_gen:
                # fitlog.add_best_metric(test_metrics)
                best_test_psnr_gen = test_metrics['psnr']
                save_arguments['best_test_psnr_gen'] = best_test_psnr_gen
                save_arguments['epoch'] = epoch
                checkpointer.save('best_valid', **save_arguments)
                self.paint_for_other_method(train_gt_views[::2000], train_pred_views[::2000], train_input_views[::2000],
                                            epoch, 'train')
                self.paint_for_other_method(test_gt_views[::500], test_pred_views[::500], test_input_views[::500],
                                            epoch, 'test')
            elif epoch % 10 == 0:
                self.paint_for_other_method(train_gt_views[::2000], train_pred_views[::2000], train_input_views[::2000],
                                            epoch, 'train')
                self.paint_for_other_method(test_gt_views[::500], test_pred_views[::500], test_input_views[::500],
                                            epoch, 'test')

    def run_one_epochs_for_other_method(self, dl, phase, optim=None):
        if phase == 'train':
            self.model.train()
        elif phase == 'test':
            self.model.eval()
        else:
            raise ValueError('phase param not found.')
        losses = []
        gt_views = []
        predict_views = []
        input_views = []
        rois_list = []
        for index, meta in tqdm(enumerate(dl)):
            source_data, rois, input_theta, target_view, target_theta, ori_data, noise = meta['data'].to(self.device), \
                                                                                         meta['rois'].to(self.device), \
                                                                                         meta['input_theta'].to(
                                                                                             self.device), \
                                                                                         meta['target_view'].unsqueeze(
                                                                                             1).to(self.device), \
                                                                                         meta['target_theta'].to(
                                                                                             self.device), \
                                                                                         meta['ori_data'], \
                                                                                         meta['noise'].unsqueeze(1).to(
                                                                                             self.device)
            rest_view = meta['rest_view'].to(self.device)
            result = self.model(source_data)
            # result [B, 12-lead_num, 512], rest_view [B, 12 - lead_num, 512]
            loss = self.loss(result.float(), rest_view.float())
            if phase == 'train':
                loss.backward()
                optim.step()
                optim.zero_grad()
            predict_views += [x for x in result.cpu().detach().numpy()]
            gt_views += [x for x in rest_view.cpu().detach().numpy()]
            input_views += [x for x in source_data.cpu().detach().numpy()]
            rois_list += [x for x in rois.cpu().detach().numpy()]
            losses.append(loss.item())
        ssim = SSIM(np.array(predict_views), np.array(gt_views), np.array(rois_list))
        psnr = PSNR(np.array(predict_views), np.array(gt_views), np.array(rois_list))
        metrics = {
            'ssim': ssim,
            'psnr': psnr,
            'loss': np.mean(losses)
        }
        return metrics, np.array(gt_views), np.array(predict_views), np.array(input_views)
