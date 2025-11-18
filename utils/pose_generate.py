from torch import tensor
import numpy as np
import torch

from utils.util import padding_traj

def pose_generator(data_set, model, refine_model, cfg,mode = "pred",num = 10):
    """
    stack k rows examples in one gif

    The logic of 'draw_order_indicator' is to cheat the render_animation(),
    because this render function only identify the first two as context and gt, which is a bit tricky to modify.
    """
    while True:
        poses = {}
        action = ""
        subject = ""
        fr_start = ""
        if mode == 'switch':
            data, actionlist = data_set.sample_all_action()
        elif mode == 'pred':
            # data,subject, action,  fr_start = data_set.sample()

            # data, subject, action,  fr_start = data_set.chose_sample(subject = "Validate/S1",action = "Walking 1 chunk4", fr_start = 192)
            # data, subject, action, fr_start = data_set.chose_sample(subject="Validate/S2", action="Jog 1 chunk0",fr_start=153)
            # data, subject, action, fr_start = data_set.chose_sample(subject="Validate/S3", action="Box 1 chunk0",
            #                                                        fr_start=42)
            # data, subject, action, fr_start = data_set.chose_sample(subject="S9",action="Eating", fr_start=1274)
            # data, subject, action, fr_start = data_set.chose_sample(subject="S9",action="SittingDown", fr_start=1186)
            # data, subject, action, fr_start = data_set.chose_sample(subject="S11",action="WalkDog 1", fr_start=802)

            # data, subject, action, fr_start = data_set.chose_sample(subject="S11", action="Walking",fr_start = 190)
            data, subject, action, fr_start = data_set.chose_sample(subject="S9",action = "Posing 1",fr_start = 559)
            if "/" in subject:
                subject = subject.split("/")[1]
        # gt
        gt = data[0].copy()
        gt[:, :1, :] = 0
        data[:, :, :1, :] = 0
        if mode == 'switch':
            index = 0
            poses = {}
            traj = data[..., 1:, :].reshape([data.shape[0], cfg.t_his + cfg.t_pred, -1])
            n = traj.shape[0]
            traj = tensor(traj, device=cfg.device, dtype=cfg.dtype)
            past_pad = padding_traj(traj, cfg.idx_pad)
            past_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], past_pad)
            traj_switch = traj[:, (cfg.t_pred - cfg.t_his):(cfg.t_pred + cfg.t_his), :]
            print(actionlist)
            traj_switch = traj_switch[index].unsqueeze(0).repeat(n, 1, 1)
            with torch.no_grad():
                sampled_motion = model.switch_sample(past_dct, cfg.mod_test, cfg.dct_m_all[:cfg.n_pre],
                                                       cfg.idct_m_all[:, :cfg.n_pre], cfg.t_his, cfg.t_pred,traj_switch,switch_start = cfg.t_pred - cfg.t_his,switch_end = cfg.t_pred + cfg.t_his)
                sampled_motion = sampled_motion.squeeze(1).permute(0, 2, 1).contiguous()
                sampled_motion = refine_model(sampled_motion)
                sampled_motion = sampled_motion.permute(0, 2, 1).contiguous()
            traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
            traj_est = traj_est.cpu().numpy()
            traj_est = traj_est.reshape(traj_est.shape[0], traj_est.shape[1], -1, 3)
            traj_est = np.concatenate((np.tile(np.zeros((1, cfg.t_his + cfg.t_pred, 1, 3)), (traj_est.shape[0], 1, 1, 1)), traj_est), axis=2)
            traj_est[..., :1, :] = 0
            traj_est[:, :cfg.t_his] = data[:, :cfg.t_his]
            # traj_est[:, (cfg.t_pred - cfg.t_his):(cfg.t_pred + cfg.t_his)] = data[index,(cfg.t_pred - cfg.t_his):(cfg.t_pred + cfg.t_his)]
            for j in range(traj_est.shape[0]):
                poses[f'humanbbd_{j}'] = traj_est[j]
            pose_id = f"end_{actionlist[index]}"
            yield poses, pose_id
        else:
            # poses['context'] = gt
            # poses['gt'] = gt
            gt = np.expand_dims(gt, axis=0)
            traj_np = gt[..., 1:, :].reshape([gt.shape[0], cfg.t_his + cfg.t_pred, -1])
            traj = tensor(traj_np, device=cfg.device, dtype=cfg.dtype)
            past_pad = padding_traj(traj, cfg.idx_pad)
            past_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], past_pad)
            with torch.no_grad():
                sampled_motion = model.idct_sample_num(past_dct, cfg.mod_test, cfg.dct_m_all[:cfg.n_pre],cfg.idct_m_all[:, :cfg.n_pre], cfg.t_his,num = num)
                # sampled_motion = model.sample_num(past_dct, cfg.mod_test, num=num)
                # sampled_motion = sampled_motion.squeeze(1)
                sampled_motion = sampled_motion.squeeze(1).permute(0, 2, 1).contiguous()
                sampled_motion = refine_model(sampled_motion)
                sampled_motion = sampled_motion.permute(0, 2, 1).contiguous()
            traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
            traj_est = traj_est.cpu().numpy()
            traj_est = traj_est.reshape(traj_est.shape[0], traj_est.shape[1], -1, 3)
            traj_est = np.concatenate((np.tile(np.zeros((1, cfg.t_his + cfg.t_pred, 1, 3)), (traj_est.shape[0], 1, 1, 1)), traj_est), axis=2)
            traj_est[..., :1, :] = 0
            traj_est[:,:cfg.t_his] = gt[:,:cfg.t_his]
            ##绘画噪声分布
            # for j in range(traj_est.shape[0]):
            #     traj_est[j] = np.random.randn(*traj_est[j].shape)
            #     poses[f'humanbdb_{j}'] = traj_est[j]

            for j in range(traj_est.shape[0]):
                poses[f'humanbbd_{j}'] = traj_est[j]
            pose_id = f"{subject}_{action}_f{fr_start}"
            print(pose_id)
            yield poses, pose_id

def chose_switch_pose_generator(data_set, model, refine_model, cfg,subject = "S9", action1 = "Eating",fr_start1 = -1,action2 = "SittingDown",fr_start2 = -1,):
    """
    stack k rows examples in one gif

    The logic of 'draw_order_indicator' is to cheat the render_animation(),
    because this render function only identify the first two as context and gt, which is a bit tricky to modify.
    """
    while True:
        start_data, subject, start_action, start_fr_start = data_set.chose_sample(subject=subject,action=action1,fr_start = fr_start1)
        end_data, subject, end_action, end_fr_start = data_set.chose_sample(subject=subject,action=action2, fr_start= fr_start2)
        start_data[:, :, :1, :] = 0
        poses = {}
        traj = start_data[..., 1:, :].reshape([start_data.shape[0], cfg.t_his + cfg.t_pred, -1])
        traj = tensor(traj, device=cfg.device, dtype=cfg.dtype)
        past_pad = padding_traj(traj, cfg.idx_pad)
        past_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], past_pad)
        end_traj = end_data[..., 1:, :].reshape([end_data.shape[0], cfg.t_his + cfg.t_pred, -1])
        end_traj = tensor(end_traj, device=cfg.device, dtype=cfg.dtype)
        traj_switch = end_traj[:, cfg.t_pred:(cfg.t_pred + cfg.t_his), :]
        with torch.no_grad():
            sampled_motion = model.switch_sample(past_dct, cfg.mod_test, cfg.dct_m_all[:cfg.n_pre],
                                                   cfg.idct_m_all[:, :cfg.n_pre], cfg.t_his, cfg.t_pred,traj_switch,switch_start = cfg.t_pred, switch_end = cfg.t_pred + cfg.t_his)
            sampled_motion = sampled_motion.squeeze(1).permute(0, 2, 1).contiguous()
            sampled_motion = refine_model(sampled_motion)
            sampled_motion = sampled_motion.permute(0, 2, 1).contiguous()
        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est.reshape(traj_est.shape[0], traj_est.shape[1], -1, 3)
        traj_est = np.concatenate((np.tile(np.zeros((1, cfg.t_his + cfg.t_pred, 1, 3)), (traj_est.shape[0], 1, 1, 1)), traj_est), axis=2)
        traj_est[..., :1, :] = 0
        traj_est[:, :cfg.t_his] = start_data[:, :cfg.t_his]
        # traj_est[:, (cfg.t_pred - cfg.t_his):(cfg.t_pred + cfg.t_his)] = data[index,(cfg.t_pred - cfg.t_his):(cfg.t_pred + cfg.t_his)]
        for j in range(traj_est.shape[0]):
            poses[f'humanbbd_{j}'] = traj_est[j]
        pose_id = f"{start_action}_f{start_fr_start}_{end_action}_f{end_fr_start}"
        yield poses, pose_id

def ground_truth_generator(data_set,subject="Validate/S3",action="Box 1 chunk0",fr_start = 0):
    """
    stack k rows examples in one gif

    The logic of 'draw_order_indicator' is to cheat the render_animation(),
    because this render function only identify the first two as context and gt, which is a bit tricky to modify.
    """
    while True:
        poses = {}
        data, subject, action, fr_start = data_set.chose_sample(subject=subject,action=action,fr_start = fr_start)
        if "/" in subject:
            subject = subject.split("/")[1]
        # gt
        gt = data[0].copy()
        gt[:, :1, :] = 0
        data[:, :, :1, :] = 0
        poses['context'] = gt
        poses['gt'] = gt
        pose_id = f"{subject}_{action}_f{fr_start}"
        yield poses, pose_id

def refine_visual_pose_generator(data_set, model, refine_model, cfg, num = 10,chosenum = None):
    while True:
        poses = {}
        action = ""
        subject = ""
        fr_start = ""
        data,subject, action,  fr_start = data_set.sample()
        # S11_Discussion 1_f2056
        data, subject, action, fr_start = data_set.chose_sample(subject="S11", action="Discussion 1",fr_start=2056)
        if "/" in subject:
            subject = subject.split("/")[1]
        # gt
        gt = data[0].copy()
        gt[:, :1, :] = 0
        data[:, :, :1, :] = 0
        poses['gt'] = gt
        gt = np.expand_dims(gt, axis=0)
        traj_np = gt[..., 1:, :].reshape([gt.shape[0], cfg.t_his + cfg.t_pred, -1])
        traj = tensor(traj_np, device=cfg.device, dtype=cfg.dtype)
        past_pad = padding_traj(traj, cfg.idx_pad)
        past_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], past_pad)
        with torch.no_grad():
            sampled_motion = model.idct_sample_num(past_dct, cfg.mod_test, cfg.dct_m_all[:cfg.n_pre],cfg.idct_m_all[:, :cfg.n_pre], cfg.t_his,num = num)
            # sampled_motion = model.sample_num(past_dct, cfg.mod_test, num=num)
            sampled_motion = sampled_motion.squeeze(1)
            refine_sampled_motion =  refine_model(sampled_motion.permute(0, 2, 1).contiguous())
            refine_sampled_motion = refine_sampled_motion.permute(0, 2, 1).contiguous()
        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est.reshape(traj_est.shape[0], traj_est.shape[1], -1, 3)
        traj_est = np.concatenate((np.tile(np.zeros((1, cfg.t_his + cfg.t_pred, 1, 3)), (traj_est.shape[0], 1, 1, 1)), traj_est), axis=2)
        traj_est[..., :1, :] = 0
        traj_est[:,:cfg.t_his] = gt[:,:cfg.t_his]
        refine_traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], refine_sampled_motion)
        refine_traj_est = refine_traj_est.cpu().numpy()
        refine_traj_est = refine_traj_est.reshape(refine_traj_est.shape[0], refine_traj_est.shape[1], -1, 3)
        refine_traj_est= np.concatenate(
            (np.tile(np.zeros((1, cfg.t_his + cfg.t_pred, 1, 3)), (refine_traj_est.shape[0], 1, 1, 1)), refine_traj_est), axis=2)
        refine_traj_est[..., :1, :] = 0
        refine_traj_est[:, :cfg.t_his] = gt[:, :cfg.t_his]
        diff = refine_traj_est - gt
        diff = diff.reshape(diff.shape[0],diff.shape[1],-1)
        dist = torch.linalg.norm(torch.tensor(diff), dim=2)
        ade, index = dist.mean(dim=1).min(dim=0)
        result = np.array([traj_est[index],refine_traj_est[index]])
        for j in range(result.shape[0]):
            poses[f'humanbbd_{j}'] = result[j]
        pose_id = f"{subject}_{action}_f{fr_start}"
        print(pose_id)
        yield poses, pose_id