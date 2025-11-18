import torch
from torch import nn
import numpy as np
from models.motion_transformer import LatentDenoisingTransformer
from models.var import diffusion_var_schedulr
from utils.util import padding_traj

class SPARD(nn.Module):
    def __init__(self,input_feats,gcn_input, n_pre,latent_dim = 512,ff_size = 1024,num_layers = 8,num_heads = 8,dropout = 0.2,activation = "gelu",
                 noise_steps=1000, etas_min = 0.04, etas_max = 0.99, scheduler = 'exponential',kappa = 1.0,power = 0.3, device = "cuda:7",sample_steps = 10,objective = "x0",
                 gcn_linear_size = 256, gcn_dropout = 0.5, gcn_layers = 12,div_k = 2):
        super().__init__()
        ## encoder
        self.device = device
        self.n_pre = n_pre
        self.refine_n_pre = gcn_input
        self.denoise_motion_transformer = LatentDenoisingTransformer(input_feats = input_feats,gcn_input = gcn_input,n_pre = n_pre,latent_dim = latent_dim,ff_size = ff_size,num_layers=num_layers,num_heads = num_heads,
                                                                              dropout = dropout,activation = activation,gcn_linear_size= gcn_linear_size,gcn_dropout = gcn_dropout,gcn_layers = gcn_layers)
        self.var_sche = diffusion_var_schedulr(noise_steps = noise_steps,sample_steps = sample_steps,etas_min = etas_min, etas_max = etas_max, scheduler = scheduler, kappa = kappa,power = power,device = device,objective = objective)
        self.div_k = div_k

    def forward(self,traj_np,cfg):
        with torch.no_grad():
            traj = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1).to(cfg.device).to(cfg.dtype)  # [b, t_total, 16x3]
            past_pad = padding_traj(traj, cfg.idx_pad)
            y_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj)
            past_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], past_pad)
        batchsize = past_dct.size(0)
        t = torch.randint(0, self.var_sche.noise_steps, (batchsize,), device = self.device).long()
        t = t.unsqueeze(1).repeat(1,self.div_k).reshape(-1)
        past_dct = past_dct.unsqueeze(1).repeat(1, self.div_k, 1, 1).reshape(batchsize * self.div_k,past_dct.size(1),past_dct.size(2))
        y_dct = y_dct.unsqueeze(1).repeat(1,self.div_k, 1, 1).reshape(batchsize * self.div_k,y_dct.size(1),y_dct.size(2))
        past_motion = traj[:,:cfg.t_his]
        past_motion = past_motion.unsqueeze(1).repeat(1, self.div_k, 1, 1).reshape(batchsize * self.div_k, past_motion.size(1), past_motion.size(2))
        y_noise_dct = self.var_sche.noise_motion(past_dct,y_dct,t)
        if np.random.random() > cfg.mod_train:
            past_dct = None
        predicted_noise = self.denoise_motion_transformer(y_noise_dct,t,past_dct,past_motion,cfg.dct_m_all,cfg.idct_m_all,cfg.t_his)
        predicted_noise = predicted_noise.reshape(batchsize,self.div_k,predicted_noise.size(1),predicted_noise.size(2))
        traj = traj.unsqueeze(1).repeat(1, self.div_k, 1, 1)
        return traj,predicted_noise

    def distill_train(self, teacher_model, traj_np, cfg):
        with torch.no_grad():
            traj = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1).to(cfg.device).to(cfg.dtype)
            past_pad = padding_traj(traj, cfg.idx_pad)
            past_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], past_pad)
        past_motion = traj[:, :cfg.t_his]
        y_T = self.var_sche.prior_sample(past_dct)
        y_t = y_T.clone()
        with torch.no_grad():
            for i in range(len(self.var_sche.sample_steps)):
                y_t = teacher_model.condition_ddim_p_sample(past = past_dct,y_t = y_t, step=i, random=cfg.mod_test, past_motion=past_motion,
                                                                     dct_m_all = cfg.dct_m_all, idct_m_all = cfg.idct_m_all, t_his = cfg.t_his)
        if not cfg.use_condition:
            past_dct = None
        predicted_y_0 = self.denoise_motion_transformer(y_T, torch.full((y_t.shape[0],), self.var_sche.sample_steps[0],device=self.device, dtype=torch.long),
                                                        past_dct, past_motion, cfg.dct_m_all,cfg.idct_m_all,cfg.t_his)
        return y_t, predicted_y_0

    def np_train(self, np_model, traj_np, cfg):
        with torch.no_grad():
            traj = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1).to(cfg.device).to(cfg.dtype)  # [b, t_total, 16x3]
            past_pad = padding_traj(traj, cfg.idx_pad)
            y_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj)
            past_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], past_pad)
        batchsize = past_dct.size(0)
        noise = np_model(past_dct)
        k = noise.size(1)
        past_dct = past_dct.unsqueeze(1).repeat(1, k, 1, 1).reshape(batchsize * k, past_dct.size(1),past_dct.size(2))
        y_dct = y_dct.unsqueeze(1).repeat(1, k, 1, 1).reshape(batchsize * k, y_dct.size(1),y_dct.size(2))
        noise = noise.reshape(batchsize * k,noise.size(2), noise.size(3))
        t = torch.randint(0, self.var_sche.noise_steps, (batchsize,), device=self.device).long()
        t = torch.full((batchsize,), self.var_sche.sample_steps[0], device=self.device, dtype=torch.long)
        t = t.unsqueeze(1).repeat(1, k).reshape(-1)
        y_noise_dct = self.var_sche.noise_motion(past_dct,y_dct, t, noise)
        past_motion = traj[:, :cfg.t_his]
        past_motion = past_motion.unsqueeze(1).repeat(1, k, 1, 1).reshape(batchsize * k,past_motion.size(1),past_motion.size(2))
        # if np.random.random() > cfg.mod_train:
        #     past_dct = None
        predicted_traj = self.denoise_motion_transformer(y_noise_dct,t,past_dct,past_motion,cfg.dct_m_all,cfg.idct_m_all,cfg.t_his)
        predicted_traj = predicted_traj.reshape(batchsize, k, predicted_traj.size(1),predicted_traj.size(2))
        traj = traj.unsqueeze(1).repeat(1, k, 1, 1)
        noise = noise.reshape(batchsize,k,noise.size(1), noise.size(2))
        return traj,predicted_traj, noise

    def ddim_p_sample(self,past,y_t,step, random,past_motion,dct_m_all, idct_m_all, t_his):
        t = torch.full((y_t.shape[0],), self.var_sche.sample_steps[step], device=self.device, dtype=torch.long)
        condition_past = past.detach().clone()
        condition_predicted_x0 = self.denoise_motion_transformer(y_t, t, condition_past, past_motion, dct_m_all,idct_m_all, t_his)
        no_condition_predicted_x0 = self.denoise_motion_transformer(y_t, t, None, past_motion, dct_m_all,idct_m_all, t_his)
        x0_recon = no_condition_predicted_x0 + random * (condition_predicted_x0 - no_condition_predicted_x0)
        if self.var_sche.sample_steps[step] == 0:
            return x0_recon
        else:
            traj = x0_recon
            traj[:, :t_his, :] = past_motion
            x0_recon = torch.matmul(dct_m_all[:self.n_pre], traj)
            n_t = torch.full((y_t.shape[0],), self.var_sche.sample_steps[step + 1], device=y_t.device,dtype=torch.long)
            k,m,j = self.var_sche.ddim_q_posterior(y_t,t,n_t)
            y_next = x0_recon * k + m * y_t + j * past
            return y_next

    def ddim_p_sample_loop(self, past_dct, random, past_motion,dct_m_all,idct_m_all, t_his,noise = None):
        y_t = self.var_sche.prior_sample(past_dct,noise)
        with torch.no_grad():
            for i in range(len(self.var_sche.sample_steps)):
                y_t = self.ddim_p_sample(past=past_dct, y_t=y_t, step=i, random=random, past_motion = past_motion,
                                                   dct_m_all = dct_m_all,idct_m_all = idct_m_all,t_his=t_his)
            return y_t

    def ddim_sample_num(self,past_dct, random,past_motion,dct_m_all, idct_m_all,t_his,num):
        batchsize = past_dct.size(0)
        past_dct = past_dct.unsqueeze(0).repeat(num, *[1] * past_dct.dim())
        past_dct = past_dct.reshape(num * batchsize, past_dct.size(2), past_dct.size(3))
        past_motion = past_motion.unsqueeze(0).repeat(num, *[1] * past_motion.dim())
        past_motion = past_motion.reshape(num * batchsize, past_motion.size(2), past_motion.size(3))
        predict = self.ddim_p_sample_loop(past_dct, random, past_motion, dct_m_all, idct_m_all, t_his)
        predict = predict.reshape(num, batchsize, predict.size(1), predict.size(2))
        return predict

    def onestep_sample(self, past_dct, past_motion, dct, idct, t_his,use_condition = True):
        y_t = self.var_sche.prior_sample(past_dct)
        condition_past = past_dct.detach().clone()
        with torch.no_grad():
            if not use_condition:
                condition_past = None
            x0_recon = self.denoise_motion_transformer(y_t, torch.full((y_t.shape[0],), self.var_sche.sample_steps[0], device=self.device, dtype=torch.long), condition_past, past_motion, dct, idct, t_his)
            return x0_recon

    def onestep_sample_num(self, past_dct, past_motion, dct, idct, t_his, num,use_condition = True):
        batchsize = past_dct.size(0)
        past_dct = past_dct.unsqueeze(0).repeat(num, *[1] * past_dct.dim())
        past_dct = past_dct.reshape(num * batchsize, past_dct.size(2), past_dct.size(3))
        past_motion = past_motion.unsqueeze(0).repeat(num, *[1] * past_motion.dim())
        past_motion = past_motion.reshape(num * batchsize, past_motion.size(2), past_motion.size(3))
        predict = self.onestep_sample(past_dct, past_motion, dct, idct,t_his,use_condition)
        predict = predict.reshape(num, batchsize, predict.size(1), predict.size(2))
        return predict

    def np_sample(self, np_model, past_dct, past_motion, dct, idct, t_his,use_condition = True):
        batchsize = past_dct.size(0)
        noise = np_model(past_dct)
        k = noise.size(1)
        noise = noise.permute(1,0,2,3).contiguous().reshape(k * batchsize, noise.size(2), noise.size(3))
        past_dct = past_dct.unsqueeze(0).repeat(k, *[1] * past_dct.dim())
        past_dct = past_dct.reshape(k * batchsize, past_dct.size(2), past_dct.size(3))
        past_motion = past_motion.unsqueeze(0).repeat(k, *[1] * past_motion.dim())
        past_motion = past_motion.reshape(k * batchsize, past_motion.size(2), past_motion.size(3))
        y_t = self.var_sche.prior_sample(past_dct,noise)
        condition_past = past_dct.detach().clone()
        with torch.no_grad():
            if not use_condition:
                condition_past = None
            x0_recon = self.denoise_motion_transformer(y_t, torch.full((y_t.shape[0],), self.var_sche.sample_steps[0], device=self.device, dtype=torch.long), condition_past, past_motion, dct, idct, t_his)
            x0_recon = x0_recon.reshape(k, batchsize, x0_recon.size(1), x0_recon.size(2))
            return x0_recon

    def np_sample_num(self,np_model, past_dct, past_motion, dct, idct, t_his, use_condition = True):
        predict = self.np_sample(np_model,past_dct, past_motion, dct, idct,t_his,use_condition)
        return predict

