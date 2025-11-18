import yaml
import os
import torch
import numpy as np

def generate_pad(padding, t_his, t_pred):
    zero_index = None
    if padding == 'Zero':
        idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
        zero_index = max(idx_pad)
    elif padding == 'Repeat':
        idx_pad = list(range(t_his)) * int(((t_pred + t_his) / t_his))
        # [0, 1, 2,....,24, 0, 1, 2,....,24, 0, 1, 2,...., 24...]
    elif padding == 'LastFrame':
        idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
        # [0, 1, 2,....,24, 24, 24,.....]
    else:
        raise NotImplementedError(f"unknown padding method: {padding}")
    return idx_pad, zero_index

def get_dct_matrix(N, is_torch=True):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = torch.from_numpy(dct_m)
        idct_m = torch.from_numpy(idct_m)
    return dct_m, idct_m

def get_log_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = '_' + str(len(dirs))

    return log_dir_index

def update_config(cfg, args_dict):
    """
    update some configuration related to args
        - merge args to cfg
        - dct, idct matrix
        - save path dir
    """
    for k, v in args_dict.items():
        setattr(cfg, k, v)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfg.dtype = dtype
    cfg.device = torch.device(cfg.device) if torch.cuda.is_available() else torch.device('cpu')
    cfg.dct_m, cfg.idct_m = get_dct_matrix(cfg.t_pred + cfg.t_his)
    cfg.dct_m_all = cfg.dct_m.float().to(cfg.device)
    cfg.idct_m_all = cfg.idct_m.float().to(cfg.device)

    index = get_log_dir_index(cfg.base_dir)
    if args_dict['mode'] == ('train' or 'pred' or 'eval'):
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir, args_dict['cfg'] + index)
    else:
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir, args_dict['mode'] + index)
    os.makedirs(cfg.cfg_dir, exist_ok=True)
    cfg.model_dir = '%s/models' % cfg.cfg_dir
    cfg.result_dir = '%s/results' % cfg.cfg_dir
    cfg.log_dir = '%s/log' % cfg.cfg_dir
    cfg.tb_dir = '%s/tb' % cfg.cfg_dir
    cfg.gif_dir = '%s/out' % cfg.cfg_dir
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.tb_dir, exist_ok=True)
    os.makedirs(cfg.gif_dir, exist_ok=True)
    cfg.model_path = os.path.join(cfg.model_dir)

    return cfg

class Config:
    def __init__(self, cfg_id, test=False):
        self.id = cfg_id
        cfg_name = './cfg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))
        # create dirs
        self.base_dir = 'inference' if test else './results'
        os.makedirs(self.base_dir, exist_ok=True)

        self.dataset = cfg.get('dataset', 'h36m')
        self.normalize_data = cfg.get('normalize_data', False)
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']
        self.batch_size = cfg['batch_size']
        self.start_epoch = cfg['start_epoch']
        self.num_epoch = cfg['num_epoch']
        self.lr = cfg['lr']
        self.gamma = cfg['gamma']
        self.padding = cfg['padding']
        self.augmentation = cfg.get('augmentation',0)
        self.stride = cfg.get('stride',0)
        self.rota_prob = cfg.get('rota_prob',0)
        self.multimodal_threshold = cfg.get('multimodal_threshold',0)

        self.scheduler = cfg['scheduler']
        self.noise_steps = cfg['noise_steps']
        self.sample_steps = cfg['sample_steps']
        self.etas_min = cfg['etas_min']
        self.etas_max = cfg['etas_max']
        self.kappa = cfg['kappa']
        self.power = cfg['power']
        self.objective = cfg['objective']
        self.loss = cfg['loss']

        self.n_pre = cfg['n_pre']
        self.refine_n_pre = cfg['refine_n_pre']
        self.num_layers = cfg['num_layers']
        self.num_heads = cfg['num_heads']
        self.latent_dims = cfg['latent_dims']
        self.dropout = cfg['dropout']
        self.mod_train = cfg['mod_train']
        self.mod_test = cfg['mod_test']
        self.save_model_interval =  cfg['save_model_interval']
        self.metrics_epoch = cfg['metrics_epoch']
        self.ckpt = cfg['ckpt']
        self.div_k = cfg['div_k']
        self.sample = cfg['sample']

        self.gcn_linear_size = cfg['gcn_linear_size']
        self.gcn_dropout = cfg['gcn_dropout']
        self.gcn_layers = cfg['gcn_layers']

        # indirect variable
        if self.dataset == 'h36m':
            self.joint_num = 16
        elif self.dataset == 'amass':
            self.joint_num = 21
        else:
            self.joint_num = 14
        self.idx_pad, self.zero_index = generate_pad(self.padding, self.t_his, self.t_pred)

        self.num_data_sample = cfg.get('num_data_sample','')
        self.num_val_data_sample = cfg.get('num_val_data_sample','')
        self.multimodal_path = cfg.get('multimodal_path','')
        self.data_candi_path = cfg.get('data_candi_path','')
        self.Complete = cfg.get('Complete','')
        self.dct_norm_enable = cfg.get('dct_norm_enable','')

