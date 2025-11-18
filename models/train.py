import time
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.transforms import DataAugmentation
from models.fid_classifier import ClassifierForFID
from utils.log import AverageMeter, AverageMeterTorch
from utils.util import padding_traj, compute_all_metrics
from torchvision import transforms
import os
from scipy.spatial.distance import pdist, squareform
from utils.metrics import APD, APDE, ADE, FDE, MMADE, MMFDE, CMD_helper, FID_helper, CMD_pose, FID_pose
import pandas as pd
import torch.nn.functional as F

class Trainer:
    def __init__(self,
                 model,
                 dataset,
                 cfg,
                 logger,
                 tb_logger):
        super().__init__()
        self.generator_val = None
        self.val_losses = None
        self.t_s = None
        self.train_losses = None
        self.criterion = None
        self.lr_scheduler = None
        self.optimizer = None
        self.generator_train = None
        self.model = model
        self.dataset = dataset
        self.cfg = cfg
        print('Preparing datasets...')
        transform = transforms.Compose([DataAugmentation(cfg.rota_prob)])
        self.train_dataset = dataset('train', cfg.t_his, cfg.t_pred, augmentation=cfg.augmentation,stride=cfg.stride, transform=transform)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=cfg.batch_size, shuffle=True,num_workers=0, pin_memory=True)
        # NOTE: test partition: can be seen only once
        self.eval_dataset = dataset('test', cfg.t_his, cfg.t_pred, augmentation=0, stride=1, transform=None)
        self.eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=16, shuffle=False, num_workers=0,pin_memory=True)
        # multimodal GT
        print('Calculating mmGT...')
        self.multimodal_traj = self.get_multimodal_gt()
        self.logger = logger
        self.tb_logger = tb_logger
        self.lrs = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.milestone,gamma=self.cfg.gamma)
        self.criterion = nn.MSELoss()
        mmapd_path = os.path.join('auxiliar/datasets/', cfg.dataset, 'mmapd_GT.csv')
        self.mmapd = self.get_mmapd(mmapd_path)
        if self.cfg.dataset == 'h36m':
            print('Loading FID classifier...')
            self.classifier_for_fid = self.get_classifier()
        else:
            print('No FID classifier available...')
            self.classifier_for_fid = None

    def get_classifier(self):
        classifier_for_fid = ClassifierForFID(input_size=48, hidden_size=128, hidden_layer=2,
                                              output_size=15, use_noise=None, device=self.cfg.device,
                                              dtype=self.cfg.dtype).to(self.cfg.device)
        classifier_path = os.path.join("./auxiliar", "h36m_classifier.pth")
        classifier_state = torch.load(classifier_path, map_location=self.cfg.device)
        classifier_for_fid.load_state_dict(classifier_state["model"])
        classifier_for_fid.eval()
        return classifier_for_fid

    def train(self):
        best_val_loss = np.inf
        best_ade = np.inf
        for epoch in range(self.cfg.start_epoch, self.cfg.num_epoch + 1):
            self.model.train()
            t_s = time.time()
            train_losses = AverageMeter()
            self.logger.info(f"Starting training epoch {epoch}:")
            for traj_np, extra in tqdm(self.train_dataloader, desc="Training Progress", unit="batch"):
                noise, predicted_noise = self.model(traj_np,self.cfg)
                if self.cfg.loss == "mse":
                    loss = F.mse_loss(predicted_noise,noise,reduction = 'none')
                elif self.cfg.loss == "mae":
                    loss = F.l1_loss(predicted_noise, noise, reduction='none')
                loss = loss.mean(-1).mean(-1).min(dim = -1)[0].mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.update(loss.item())
            self.tb_logger.add_scalar('Loss/train', train_losses.avg, epoch)
            self.lr_scheduler.step()
            self.lrs.append(self.optimizer.param_groups[0]['lr'])
            self.logger.info('====> Epoch: {} Time: {:.2f} Train Loss: {} lr: {:.5f}'.format(epoch,
                                                                                             time.time() - t_s,
                                                                                             train_losses.avg,
                                                                                             self.lrs[-1]))
            val_loss = self.eval(epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"best_val_loss_ckpt_{epoch}.pt"))
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"best_model.pt"))
            if epoch % self.cfg.metrics_epoch == 0:
                meter = self.evaluation()
                ade = meter["ADE"]
                self.tb_logger.add_scalar('APD', meter["APD"], epoch)
                self.tb_logger.add_scalar('ADE', meter["ADE"], epoch)
                self.tb_logger.add_scalar('FDE', meter["FDE"], epoch)
                self.tb_logger.add_scalar('MMADE', meter["MMADE"], epoch)
                self.tb_logger.add_scalar('MMFDE', meter["MMFDE"], epoch)
                if ade < best_ade:
                    best_ade = ade
                    torch.save(self.model.state_dict(),os.path.join(self.cfg.model_path, f"best_ade_ckpt_{epoch}.pt"))
                    torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"best_ade.pt"))

            if self.cfg.save_model_interval > 0 and epoch % self.cfg.save_model_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{epoch}.pt"))


    def eval(self,epoch):
        self.model.eval()
        t_s = time.time()
        val_losses = AverageMeter()
        self.logger.info(f"Starting val epoch {epoch}:")
        for traj_np, extra in self.eval_dataloader:
            with torch.no_grad():
                noise, predicted_noise = self.model(traj_np,self.cfg)
                if self.cfg.loss == "mse":
                    loss = F.mse_loss(predicted_noise,noise, reduction='none')
                elif self.cfg.loss == "mae":
                    loss = F.l1_loss(predicted_noise, noise, reduction='none')
                loss = loss.mean(-1).mean(-1).min(dim=-1)[0].mean()
                loss = loss.mean()
                val_losses.update(loss.item())
        self.tb_logger.add_scalar('Loss/val', val_losses.avg, epoch)
        self.logger.info('====> Epoch: {} Time: {:.2f} Val Loss: {}'.format(epoch, time.time() - t_s, val_losses.avg))
        return val_losses.avg

    def get_multimodal_gt(self):
        """
        return list of tensors of shape [[num_similar, t_pred, NC]]
        """
        all_data = []
        for i, (data, extra) in enumerate(self.eval_dataloader):    # [batch_size, t_all, num_joints, dim]
            data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
            all_data.append(data)
        all_data = np.concatenate(all_data, axis=0)
        all_start_pose = all_data[:,self.cfg.t_his-1,:]
        pd = squareform(pdist(all_start_pose))
        traj_gt_arr = []
        for i in range(pd.shape[0]):
            ind = np.nonzero(pd[i] < self.cfg.multimodal_threshold)
            traj_gt_arr.append(torch.from_numpy(all_data[ind][:, self.cfg.t_his:, :]).to(torch.float32))
        return traj_gt_arr

    def get_prediction(self, data, sample_num):
        """
        data: [batch_size, total_len, num_joints=17, 3]
        act:  [batch_size]
        sample_num: how many samples to generate for one data entry
        """
        with torch.no_grad():
            traj = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1).to(self.cfg.device).to(
                self.cfg.dtype)  # [b, t_total, 16x3]
            past_pad = padding_traj(traj, self.cfg.idx_pad)
            past_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], past_pad)
            if self.cfg.sample == "ddim":
                predict_y = self.model.ddim_sample_num(past_dct, self.cfg.mod_test, traj[:, :self.cfg.t_his],
                                                           self.cfg.dct_m_all,
                                                           self.cfg.idct_m_all,
                                                           self.cfg.t_his, sample_num)
            elif self.cfg.sample == "onestep":
                predict_y = self.model.onestep_sample_num(past_dct, traj[:, :self.cfg.t_his],
                                                                 self.cfg.dct_m_all,
                                                                 self.cfg.idct_m_all,
                                                                 self.cfg.t_his, sample_num,self.cfg.use_condition)
            else:
                print("sample is not implemented")
                exit(-1)
            traj = predict_y.permute(1, 0, 2, 3).contiguous()
        return traj[:,:,self.cfg.t_his:]

    def get_mmapd(self, mmapd_path):
        df = pd.read_csv(mmapd_path)
        mmapds = torch.as_tensor(list(df["gt_APD"]))
        return mmapds

    @torch.no_grad()
    def compute_stats(self):
        """
        return: dic [stat_name, stat_val] NOTE: val.avg is standard
        """
        self.model.eval()
        def get_gt(data, input_n):
            gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
            return gt[:, input_n:, :]
        # all quantitative results in paper
        stats_func = {'APD': APD, 'APDE': APDE, 'ADE': ADE, 'FDE': FDE, 'MMADE': MMADE, 'MMFDE': MMFDE, 'FID': None,'CMD': None}
        stats_names = list(stats_func.keys())
        stats_meter = {x: AverageMeterTorch() for x in stats_names}
        histogram_data = []
        all_pred_activations = []  # for FID. We need to compute the activations of the predictions
        all_gt_activations = []  # for FID. We need to compute the activations of the GT
        all_pred_classes = []
        all_gt_classes = []
        all_obs_classes = []
        counter = 0
        for i, (data, extra) in tqdm(enumerate(self.eval_dataloader)):
            gt = get_gt(data, self.cfg.t_his).to(self.cfg.device).to(self.cfg.dtype)  # [batch_size, t_pred, NC]
            pred = self.get_prediction(data, sample_num = 50)
            gt_multi = self.multimodal_traj[counter:counter + gt.shape[0]]
            gt_multi = [t.to(self.cfg.device) for t in gt_multi]
            gt_apd = self.mmapd[counter:counter + gt.shape[0]].to(self.cfg.device)
            for stats in stats_names:
                if stats not in ('APDE', 'FID', 'CMD'):
                    val = stats_func[stats](pred, gt, gt_multi)
                    stats_meter[stats].update(val)
            # calculate APDE
            apde = stats_func['APDE'](stats_meter['APD'].raw_val, gt_apd)
            stats_meter['APDE'].update(apde)
            counter += data.shape[0]
            CMD_helper(pred, extra, histogram_data, all_obs_classes)
            if self.cfg.dataset == 'h36m':
                FID_helper(pred, gt, self.classifier_for_fid, all_pred_activations, all_gt_activations,
                           all_pred_classes, all_gt_classes)
        cmd_val = CMD_pose(self.eval_dataset, histogram_data, all_obs_classes)
        stats_meter['CMD'].direct_set_avg(cmd_val)
        if self.cfg.dataset == 'h36m':
            fid_val = FID_pose(all_gt_activations, all_pred_activations)
            stats_meter['FID'].direct_set_avg(fid_val)
        return stats_meter

    def evaluation(self):
        """NOTE: can be only called once"""
        stats_dic = self.compute_stats()
        for stats in stats_dic:
            str_stats = f'{stats:<6}: ' + f'({stats_dic[stats].avg:.4f})'
            self.logger.info(str_stats)
        return {x: y.avg for x, y in stats_dic.items()}

class DistillTrainer:
    def __init__(self,
                 model,
                 teacher_model,
                 dataset,
                 cfg,
                 logger,
                 tb_logger):
        super().__init__()
        self.generator_val = None
        self.val_losses = None
        self.t_s = None
        self.train_losses = None
        self.criterion = None
        self.lr_scheduler = None
        self.optimizer = None
        self.generator_train = None
        self.model = model
        self.teacher_model = teacher_model
        self.teacher_model.requires_grad_(False)

        self.dataset = dataset
        self.cfg = cfg
        print('Preparing datasets...')
        transform = transforms.Compose([DataAugmentation(cfg.rota_prob)])
        self.train_dataset = dataset('train', cfg.t_his, cfg.t_pred, augmentation=cfg.augmentation,stride=cfg.stride, transform=transform)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=cfg.batch_size, shuffle=True,num_workers=0, pin_memory=True)
        # NOTE: test partition: can be seen only once
        self.eval_dataset = dataset('test', cfg.t_his, cfg.t_pred, augmentation=0, stride=1, transform=None)
        self.eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=16, shuffle=False, num_workers=0,pin_memory=True)
        # multimodal GT
        print('Calculating mmGT...')
        self.multimodal_traj = self.get_multimodal_gt()
        self.logger = logger
        self.tb_logger = tb_logger
        self.lrs = []
        for param in self.model.denoise_motion_transformer.gcn.parameters():
            param.requires_grad = False
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cfg.lr)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.milestone,gamma=self.cfg.gamma)
        self.criterion = nn.MSELoss()
        mmapd_path = os.path.join('auxiliar/datasets/', cfg.dataset, 'mmapd_GT.csv')
        self.mmapd = self.get_mmapd(mmapd_path)
        if self.cfg.dataset == 'h36m':
            print('Loading FID classifier...')
            self.classifier_for_fid = self.get_classifier()
        else:
            print('No FID classifier available...')
            self.classifier_for_fid = None

    def get_classifier(self):
        classifier_for_fid = ClassifierForFID(input_size=48, hidden_size=128, hidden_layer=2,
                                              output_size=15, use_noise=None, device=self.cfg.device,
                                              dtype=self.cfg.dtype).to(self.cfg.device)
        classifier_path = os.path.join("./auxiliar", "h36m_classifier.pth")
        classifier_state = torch.load(classifier_path, map_location=self.cfg.device)
        classifier_for_fid.load_state_dict(classifier_state["model"])
        classifier_for_fid.eval()
        return classifier_for_fid

    def train(self):
        best_val_loss = np.inf
        best_ade = np.inf
        for epoch in range(self.cfg.start_epoch, self.cfg.num_epoch + 1):
            self.model.train()
            t_s = time.time()
            train_losses = AverageMeter()
            self.logger.info(f"Starting training epoch {epoch}:")
            for traj_np, extra in tqdm(self.train_dataloader, desc="Training Progress", unit="batch"):
                noise, predicted_noise = self.model.distill_train(self.teacher_model,traj_np,self.cfg)
                if self.cfg.loss == "mse":
                    loss = F.mse_loss(predicted_noise,noise)
                elif self.cfg.loss == "mae":
                    loss = F.l1_loss(predicted_noise, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.update(loss.item())
            self.tb_logger.add_scalar('Loss/train', train_losses.avg, epoch)
            self.lr_scheduler.step()
            self.lrs.append(self.optimizer.param_groups[0]['lr'])
            self.logger.info('====> Epoch: {} Time: {:.2f} Train Loss: {} lr: {:.5f}'.format(epoch,
                                                                                             time.time() - t_s,
                                                                                             train_losses.avg,
                                                                                             self.lrs[-1]))
            val_loss = self.eval(epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"best_val_loss_ckpt_{epoch}.pt"))
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"best_model.pt"))
            if epoch % self.cfg.metrics_epoch == 0:
                meter = self.evaluation()
                ade = meter["ADE"]
                self.tb_logger.add_scalar('APD', meter["APD"], epoch)
                self.tb_logger.add_scalar('ADE', meter["ADE"], epoch)
                self.tb_logger.add_scalar('FDE', meter["FDE"], epoch)
                self.tb_logger.add_scalar('MMADE', meter["MMADE"], epoch)
                self.tb_logger.add_scalar('MMFDE', meter["MMFDE"], epoch)
                if ade < best_ade:
                    best_ade = ade
                    torch.save(self.model.state_dict(),os.path.join(self.cfg.model_path, f"best_ade_ckpt_{epoch}.pt"))
                    torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"best_ade.pt"))
            if self.cfg.save_model_interval > 0 and epoch % self.cfg.save_model_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{epoch}.pt"))


    def eval(self,epoch):
        self.model.eval()
        t_s = time.time()
        val_losses = AverageMeter()
        self.logger.info(f"Starting val epoch {epoch}:")
        for traj_np, extra in self.eval_dataloader:
            with torch.no_grad():
                noise, predicted_noise = self.model.distill_train(self.teacher_model,traj_np,self.cfg)
                if self.cfg.loss == "mse":
                    loss = F.mse_loss(predicted_noise,noise)
                elif self.cfg.loss == "mae":
                    loss = F.l1_loss(predicted_noise, noise)
                val_losses.update(loss.item())
        self.tb_logger.add_scalar('Loss/val', val_losses.avg, epoch)
        self.logger.info('====> Epoch: {} Time: {:.2f} Val Loss: {}'.format(epoch, time.time() - t_s, val_losses.avg))
        return val_losses.avg

    def get_multimodal_gt(self):
        """
        return list of tensors of shape [[num_similar, t_pred, NC]]
        """
        all_data = []
        for i, (data, extra) in enumerate(self.eval_dataloader):    # [batch_size, t_all, num_joints, dim]
            data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
            all_data.append(data)
        all_data = np.concatenate(all_data, axis=0)
        all_start_pose = all_data[:,self.cfg.t_his-1,:]
        pd = squareform(pdist(all_start_pose))
        traj_gt_arr = []
        for i in range(pd.shape[0]):
            ind = np.nonzero(pd[i] < self.cfg.multimodal_threshold)
            traj_gt_arr.append(torch.from_numpy(all_data[ind][:, self.cfg.t_his:, :]).to(torch.float32))
        return traj_gt_arr

    def get_prediction(self, data, sample_num):
        """
        data: [batch_size, total_len, num_joints=17, 3]
        act:  [batch_size]
        sample_num: how many samples to generate for one data entry
        """
        with torch.no_grad():
            traj = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1).to(self.cfg.device).to(
                self.cfg.dtype)  # [b, t_total, 16x3]
            past_pad = padding_traj(traj, self.cfg.idx_pad)
            past_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], past_pad)
            if self.cfg.sample == "ddim":
                predict_y = self.model.ddim_sample_num(past_dct, self.cfg.mod_test, traj[:, :self.cfg.t_his],
                                                           self.cfg.dct_m_all,
                                                           self.cfg.idct_m_all,
                                                           self.cfg.t_his, sample_num)
            elif self.cfg.sample == "onestep":
                predict_y = self.model.onestep_sample_num(past_dct, traj[:, :self.cfg.t_his],
                                                                 self.cfg.dct_m_all,
                                                                 self.cfg.idct_m_all,
                                                                 self.cfg.t_his, sample_num,self.cfg.use_condition)
            traj = predict_y.permute(1, 0, 2, 3).contiguous()
        return traj[:,:,self.cfg.t_his:]

    def get_mmapd(self, mmapd_path):
        df = pd.read_csv(mmapd_path)
        mmapds = torch.as_tensor(list(df["gt_APD"]))
        return mmapds

    @torch.no_grad()
    def compute_stats(self):
        """
        return: dic [stat_name, stat_val] NOTE: val.avg is standard
        """
        self.model.eval()
        def get_gt(data, input_n):
            gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
            return gt[:, input_n:, :]
        # all quantitative results in paper
        stats_func = {'APD': APD, 'APDE': APDE, 'ADE': ADE, 'FDE': FDE, 'MMADE': MMADE, 'MMFDE': MMFDE, 'FID': None,'CMD': None}
        stats_names = list(stats_func.keys())
        stats_meter = {x: AverageMeterTorch() for x in stats_names}
        histogram_data = []
        all_pred_activations = []  # for FID. We need to compute the activations of the predictions
        all_gt_activations = []  # for FID. We need to compute the activations of the GT
        all_pred_classes = []
        all_gt_classes = []
        all_obs_classes = []
        counter = 0
        for i, (data, extra) in tqdm(enumerate(self.eval_dataloader)):
            gt = get_gt(data, self.cfg.t_his).to(self.cfg.device).to(self.cfg.dtype)  # [batch_size, t_pred, NC]
            pred = self.get_prediction(data, sample_num = 50)
            gt_multi = self.multimodal_traj[counter:counter + gt.shape[0]]
            gt_multi = [t.to(self.cfg.device) for t in gt_multi]
            gt_apd = self.mmapd[counter:counter + gt.shape[0]].to(self.cfg.device)
            for stats in stats_names:
                if stats not in ('APDE', 'FID', 'CMD'):
                    val = stats_func[stats](pred, gt, gt_multi)
                    stats_meter[stats].update(val)
            # calculate APDE
            apde = stats_func['APDE'](stats_meter['APD'].raw_val, gt_apd)
            stats_meter['APDE'].update(apde)
            counter += data.shape[0]
            CMD_helper(pred, extra, histogram_data, all_obs_classes)
            if self.cfg.dataset == 'h36m':
                FID_helper(pred, gt, self.classifier_for_fid, all_pred_activations, all_gt_activations,
                           all_pred_classes, all_gt_classes)
        cmd_val = CMD_pose(self.eval_dataset, histogram_data, all_obs_classes)
        stats_meter['CMD'].direct_set_avg(cmd_val)
        if self.cfg.dataset == 'h36m':
            fid_val = FID_pose(all_gt_activations, all_pred_activations)
            stats_meter['FID'].direct_set_avg(fid_val)
        return stats_meter

    def evaluation(self):
        """NOTE: can be only called once"""
        stats_dic = self.compute_stats()
        for stats in stats_dic:
            str_stats = f'{stats:<6}: ' + f'({stats_dic[stats].avg:.4f})'
            self.logger.info(str_stats)
        return {x: y.avg for x, y in stats_dic.items()}

class npTrainer:
    def __init__(self,
                 np_model,
                 model,
                 dataset,
                 cfg,
                 logger,
                 tb_logger):
        super().__init__()
        self.generator_val = None
        self.val_losses = None
        self.t_s = None
        self.train_losses = None
        self.criterion = None
        self.lr_scheduler = None
        self.optimizer = None
        self.generator_train = None
        self.np_model = np_model
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.cfg = cfg
        print('Preparing datasets...')
        transform = transforms.Compose([DataAugmentation(cfg.rota_prob)])
        self.train_dataset = dataset('train', cfg.t_his, cfg.t_pred, augmentation=cfg.augmentation,stride=cfg.stride, transform=transform)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=cfg.batch_size, shuffle=True,num_workers=0, pin_memory=True)
        # NOTE: test partition: can be seen only once
        self.eval_dataset = dataset('test', cfg.t_his, cfg.t_pred, augmentation=0, stride=1, transform = None)
        self.eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=16, shuffle=False, num_workers=0,pin_memory=True)
        # multimodal GT
        print('Calculating mmGT...')
        self.multimodal_traj = self.get_multimodal_gt()
        self.logger = logger
        self.tb_logger = tb_logger
        self.lrs = []
        self.optimizer = optim.Adam(self.np_model.parameters(), lr=self.cfg.lr)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.milestone,gamma=self.cfg.gamma)
        self.criterion = nn.MSELoss()
        mmapd_path = os.path.join('auxiliar/datasets/', cfg.dataset, 'mmapd_GT.csv')
        self.mmapd = self.get_mmapd(mmapd_path)
        if self.cfg.dataset == 'h36m':
            print('Loading FID classifier...')
            self.classifier_for_fid = self.get_classifier()
        else:
            print('No FID classifier available...')
            self.classifier_for_fid = None

    def get_classifier(self):
        classifier_for_fid = ClassifierForFID(input_size=48, hidden_size=128, hidden_layer=2,
                                              output_size=15, use_noise=None, device=self.cfg.device,
                                              dtype=self.cfg.dtype).to(self.cfg.device)
        classifier_path = os.path.join("./auxiliar", "h36m_classifier.pth")
        classifier_state = torch.load(classifier_path, map_location=self.cfg.device)
        classifier_for_fid.load_state_dict(classifier_state["model"])
        classifier_for_fid.eval()
        return classifier_for_fid

    def train(self):
        best_val_loss = np.inf
        best_ade = np.inf
        for epoch in range(self.cfg.start_epoch, self.cfg.num_epoch + 1):
            self.np_model.train()
            t_s = time.time()
            recon_losses = AverageMeter()
            train_losses = AverageMeter()
            self.logger.info(f"Starting training epoch {epoch}:")
            for traj_np, extra in tqdm(self.train_dataloader, desc="Training Progress", unit="batch"):
                ## 16,50,125,48
                traj, predicted_traj, noise = self.model.np_train(self.np_model,traj_np,self.cfg)
                if self.cfg.loss == "mse":
                    recon_loss = F.mse_loss(predicted_traj,traj,reduction = 'none')
                elif self.cfg.loss == "mae":
                    recon_loss = F.l1_loss(predicted_traj, traj, reduction='none')
                recon_loss = recon_loss.mean(-1).mean(-1).min(dim = -1)[0].mean()
                loss = recon_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                recon_losses.update(recon_loss.item())
                train_losses.update(loss.item())
            self.tb_logger.add_scalar('Loss/train', train_losses.avg, epoch)
            self.lr_scheduler.step()
            self.lrs.append(self.optimizer.param_groups[0]['lr'])
            self.logger.info('====> Epoch: {} Time: {:.2f} Train Loss: {} Recon Loss: {} lr: {:.5f}'.format(epoch,
                                                                                             time.time() - t_s,
                                                                                             train_losses.avg,
                                                                                             recon_losses.avg,
                                                                                             self.lrs[-1]))
            val_loss = self.eval(epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.np_model.state_dict(),
                           os.path.join(self.cfg.model_path, f"best_val_loss_ckpt_{epoch}.pt"))
                torch.save(self.np_model.state_dict(), os.path.join(self.cfg.model_path, f"best_model.pt"))
            if epoch % self.cfg.metrics_epoch == 0:
                meter = self.evaluation()
                ade = meter["ADE"]
                self.tb_logger.add_scalar('APD', meter["APD"], epoch)
                self.tb_logger.add_scalar('ADE', meter["ADE"], epoch)
                self.tb_logger.add_scalar('FDE', meter["FDE"], epoch)
                self.tb_logger.add_scalar('MMADE', meter["MMADE"], epoch)
                self.tb_logger.add_scalar('MMFDE', meter["MMFDE"], epoch)
                if ade < best_ade:
                    best_ade = ade
                    torch.save(self.np_model.state_dict(),
                               os.path.join(self.cfg.model_path, f"best_ade_ckpt_{epoch}.pt"))
                    torch.save(self.np_model.state_dict(), os.path.join(self.cfg.model_path, f"best_ade.pt"))

            if self.cfg.save_model_interval > 0 and epoch % self.cfg.save_model_interval == 0:
                torch.save(self.np_model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{epoch}.pt"))


    def eval(self,epoch):
        self.np_model.eval()
        t_s = time.time()
        recon_losses = AverageMeter()
        val_losses = AverageMeter()
        self.logger.info(f"Starting val epoch {epoch}:")
        for traj_np, extra in self.eval_dataloader:
            with torch.no_grad():
                traj, predicted_traj, noise = self.model.np_train(self.np_model, traj_np, self.cfg)
                if self.cfg.loss == "mse":
                    recon_loss = F.mse_loss(predicted_traj, traj, reduction='none')
                elif self.cfg.loss == "mae":
                    recon_loss = F.l1_loss(predicted_traj, traj, reduction='none')
                recon_loss = recon_loss.mean(-1).mean(-1).min(dim=-1)[0].mean()
                recon_losses.update(recon_loss.item())
                loss = recon_loss
                val_losses.update(loss.item())
        self.tb_logger.add_scalar('Loss/val', val_losses.avg, epoch)
        self.logger.info('====> Epoch: {} Time: {:.2f} Val Loss: {} Recon Loss: {}'.format(epoch, time.time() - t_s, val_losses.avg, recon_losses.avg))
        return val_losses.avg

    def get_multimodal_gt(self):
        """
        return list of tensors of shape [[num_similar, t_pred, NC]]
        """
        all_data = []
        for i, (data, extra) in enumerate(self.eval_dataloader):    # [batch_size, t_all, num_joints, dim]
            data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
            all_data.append(data)
        all_data = np.concatenate(all_data, axis=0)
        all_start_pose = all_data[:,self.cfg.t_his-1,:]
        pd = squareform(pdist(all_start_pose))
        traj_gt_arr = []
        for i in range(pd.shape[0]):
            ind = np.nonzero(pd[i] < self.cfg.multimodal_threshold)
            traj_gt_arr.append(torch.from_numpy(all_data[ind][:, self.cfg.t_his:, :]).to(torch.float32))
        return traj_gt_arr

    def get_prediction(self, data, sample_num):
        """
        data: [batch_size, total_len, num_joints=17, 3]
        act:  [batch_size]
        sample_num: how many samples to generate for one data entry
        """
        with torch.no_grad():
            traj = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1).to(self.cfg.device).to(
                self.cfg.dtype)  # [b, t_total, 16x3]
            past_pad = padding_traj(traj, self.cfg.idx_pad)
            past_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], past_pad)
            if self.cfg.sample == "ddim":
                predict_y = self.model.ddim_sample_num(past_dct, self.cfg.mod_test, traj[:, :self.cfg.t_his],
                                                           self.cfg.dct_m_all,
                                                           self.cfg.idct_m_all,
                                                           self.cfg.t_his, sample_num)
            elif self.cfg.sample == "onestep":
                predict_y = self.model.onestep_sample_num(past_dct, traj[:, :self.cfg.t_his],
                                                                 self.cfg.dct_m_all,
                                                                 self.cfg.idct_m_all,
                                                                 self.cfg.t_his, sample_num,self.cfg.use_condition)
            elif self.cfg.sample == "np":
                predict_y = self.model.np_sample_num(self.np_model,past_dct, traj[:, :self.cfg.t_his],
                                                          self.cfg.dct_m_all,
                                                          self.cfg.idct_m_all,
                                                          self.cfg.t_his,self.cfg.use_condition)
            traj = predict_y.permute(1, 0, 2, 3).contiguous()
        return traj[:,:,self.cfg.t_his:]

    def get_mmapd(self, mmapd_path):
        df = pd.read_csv(mmapd_path)
        mmapds = torch.as_tensor(list(df["gt_APD"]))
        return mmapds

    @torch.no_grad()
    def compute_stats(self):
        """
        return: dic [stat_name, stat_val] NOTE: val.avg is standard
        """
        self.model.eval()
        self.np_model.eval()
        def get_gt(data, input_n):
            gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
            return gt[:, input_n:, :]
        # all quantitative results in paper
        stats_func = {'APD': APD, 'APDE': APDE, 'ADE': ADE, 'FDE': FDE, 'MMADE': MMADE, 'MMFDE': MMFDE, 'FID': None,'CMD': None}
        stats_names = list(stats_func.keys())
        stats_meter = {x: AverageMeterTorch() for x in stats_names}
        histogram_data = []
        all_pred_activations = []  # for FID. We need to compute the activations of the predictions
        all_gt_activations = []  # for FID. We need to compute the activations of the GT
        all_pred_classes = []
        all_gt_classes = []
        all_obs_classes = []
        counter = 0
        for i, (data, extra) in tqdm(enumerate(self.eval_dataloader)):
            gt = get_gt(data, self.cfg.t_his).to(self.cfg.device).to(self.cfg.dtype)  # [batch_size, t_pred, NC]
            pred = self.get_prediction(data, sample_num = 50)
            gt_multi = self.multimodal_traj[counter:counter + gt.shape[0]]
            gt_multi = [t.to(self.cfg.device) for t in gt_multi]
            gt_apd = self.mmapd[counter:counter + gt.shape[0]].to(self.cfg.device)
            for stats in stats_names:
                if stats not in ('APDE', 'FID', 'CMD'):
                    val = stats_func[stats](pred, gt, gt_multi)
                    stats_meter[stats].update(val)
            # calculate APDE
            apde = stats_func['APDE'](stats_meter['APD'].raw_val, gt_apd)
            stats_meter['APDE'].update(apde)
            counter += data.shape[0]
            CMD_helper(pred, extra, histogram_data, all_obs_classes)
            if self.cfg.dataset == 'h36m':
                FID_helper(pred, gt, self.classifier_for_fid, all_pred_activations, all_gt_activations,
                           all_pred_classes, all_gt_classes)
        cmd_val = CMD_pose(self.eval_dataset, histogram_data, all_obs_classes)
        stats_meter['CMD'].direct_set_avg(cmd_val)
        if self.cfg.dataset == 'h36m':
            fid_val = FID_pose(all_gt_activations, all_pred_activations)
            stats_meter['FID'].direct_set_avg(fid_val)
        return stats_meter

    def evaluation(self):
        """NOTE: can be only called once"""
        stats_dic = self.compute_stats()
        for stats in stats_dic:
            str_stats = f'{stats:<6}: ' + f'({stats_dic[stats].avg:.4f})'
            self.logger.info(str_stats)
        return {x: y.avg for x, y in stats_dic.items()}

