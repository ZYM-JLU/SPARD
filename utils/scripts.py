import torch
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import pdist, squareform

def display_exp_setting(logger, cfg):
    """
    log the current experiment settings.
    """
    logger.info('=' * 80)
    log_dict = cfg.__dict__.copy()
    for key in list(log_dict):
        if 'dir' in key or 'path' in key or 'dct' in key:
            del log_dict[key]
    del log_dict['zero_index']
    del log_dict['idx_pad']
    logger.info(log_dict)
    logger.info('=' * 80)

def get_multimodal_gt_full(logger, dataset_multi_test, args, cfg):
    """
    calculate the multi-modal data
    """
    logger.info('preparing full evaluation dataset...')
    data_group = []
    acts = []
    num_samples = 0
    data_gen_multi_test = dataset_multi_test.iter_generator(step=cfg.t_his)
    for data, _,act in data_gen_multi_test:
        num_samples += 1
        data_group.append(torch.from_numpy(data))
        acts += [act]
    data_group = torch.cat(data_group,dim = 0)
    data_group = data_group.numpy()
    all_data = data_group[..., 1:, :].reshape(data_group.shape[0], data_group.shape[1], -1)
    gt_group = all_data[:, cfg.t_his:, :]
    all_start_pose = all_data[:, cfg.t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    num_mult = []
    for i in tqdm(range(pd.shape[0])):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, cfg.t_his:, :])
        num_mult.append(len(ind[0]))
    num_mult = np.array(num_mult)
    logger.info('=' * 80)
    logger.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{len(num_mult)}')
    logger.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{len(num_mult)}')
    logger.info('done...')
    logger.info('=' * 80)
    return {'traj_gt_arr': traj_gt_arr,
            'data_group': data_group,
            'gt_group': gt_group,
            'num_samples': num_samples,
            'acts': acts}