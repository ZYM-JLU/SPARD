import argparse
import torch
import random
import os
from models.noise_predictor import NoisePredictor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from config import Config, update_config
from data_loader.dataset_amass import DatasetAMASS
from data_loader.dataset_h36m import DatasetH36M
from models.model import SPARD
from models.train import Trainer, DistillTrainer, npTrainer
from utils.log import create_logger
from utils.scripts import display_exp_setting
from tensorboardX import SummaryWriter


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='h36m', help='h36m or amass')
    parser.add_argument('--mode', default='np_eval', help='train / distill / np / eval / distill_eval / np_eval')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1.e-4)
    parser.add_argument('--milestone', type=list, default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--padding', type=str, default="LastFrame")
    parser.add_argument('--augumentation', type=int, default=5)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--rota_prob', type=int, default=1)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    ##diffusion
    parser.add_argument('--scheduler', type=str, default="exponential")
    parser.add_argument('--noise_steps', type=int, default=5)
    parser.add_argument('--sample_steps', type=int, default=5)
    parser.add_argument('--etas_min', type=float, default=0.04)
    parser.add_argument('--etas_max', type=float, default=0.99)
    parser.add_argument('--kappa', type=float, default=4.0)
    parser.add_argument('--power', type=float, default=0.3)
    parser.add_argument('--objective', type=str, default="x0")
    parser.add_argument("--loss", type=str, default="mae")
    ##model
    parser.add_argument('--n_pre', type=int, default=20)
    parser.add_argument('--refine_n_pre', type=int, default=125)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--latent_dims', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--mod_train', type=float, default=0.5)
    parser.add_argument('--mod_test', type=float, default=0.3)
    parser.add_argument('--save_model_interval', type=int, default=100)
    parser.add_argument('--metrics_epoch', type=int, default=50)
    parser.add_argument('--div_k', type=int, default=10)
    parser.add_argument("--sample", type=str, default="ddim", help='ddim or onestep or np')
    parser.add_argument("--gcn_linear_size", type=int, default=256)
    parser.add_argument("--gcn_dropout", type=float, default=0.5)
    parser.add_argument("--gcn_layers", type=int, default=6)
    parser.add_argument('--ckpt', type=str, default="")
    ## distill
    parser.add_argument("--use_condition", type=bool, default=True)
    parser.add_argument('--distill_ckpt', type=str, default="")
    ## noise_predictor
    parser.add_argument("--np_gcn_linear_size", type=int, default=256)
    parser.add_argument("--np_gcn_dropout", type=float, default=0.5)
    parser.add_argument("--np_gcn_layers", type=int, default=6)
    parser.add_argument("--np_alpha", type=float, default=3.0)
    parser.add_argument('--np_ckpt', type=str, default="")
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_model(cfg):
    model = SPARD(input_feats=3 * cfg.joint_num,
                  gcn_input=cfg.refine_n_pre,
                  n_pre=cfg.n_pre,
                  num_layers=cfg.num_layers,
                  num_heads=cfg.num_heads,
                  latent_dim=cfg.latent_dims,
                  dropout=cfg.dropout,
                  device=cfg.device,
                  noise_steps=cfg.noise_steps,
                  etas_min=cfg.etas_min,
                  etas_max=cfg.etas_max,
                  kappa=cfg.kappa,
                  power=cfg.power,
                  sample_steps=cfg.sample_steps,
                  scheduler=cfg.scheduler,
                  objective=cfg.objective,
                  gcn_linear_size=cfg.gcn_linear_size,
                  gcn_dropout=cfg.gcn_dropout,
                  gcn_layers=cfg.gcn_layers,
                  div_k=cfg.div_k).to(cfg.device)
    return model


def create_noise_predictor_model(cfg):
    np_model = NoisePredictor(node_n=3 * cfg.joint_num,
                              input_feature=cfg.n_pre,
                              hidden_feature=cfg.np_gcn_linear_size,
                              p_dropout=cfg.np_gcn_dropout,
                              num_stage=cfg.np_gcn_layers).to(cfg.device)
    return np_model


if __name__ == '__main__':
    args = set_args()
    set_seed(args.seed)
    cfg = Config(f'{args.cfg}', test=((args.mode != 'train')))
    cfg = update_config(cfg, vars(args))

    # 设置采样方法
    cfg.sample = args.sample

    if cfg.dataset == 'h36m':
        dataset_cls = DatasetH36M
    else:
        dataset_cls = DatasetAMASS
    """logger"""
    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)
    ##model
    model = create_model(cfg)

    if args.mode == 'train':
        trainer = Trainer(model=model, dataset=dataset_cls, cfg=cfg, logger=logger, tb_logger=tb_logger)
        trainer.train()
    elif args.mode == 'distill':
        teacher_model = create_model(cfg)
        if args.ckpt:
            ckpt = torch.load(args.ckpt, map_location=torch.device(cfg.device))
            model.load_state_dict(ckpt)
            teacher_model.load_state_dict(ckpt)
        distill_trainer = DistillTrainer(model=model, teacher_model=teacher_model, dataset=dataset_cls, cfg=cfg,
                                         logger=logger, tb_logger=tb_logger)
        distill_trainer.train()
    elif args.mode == 'np':
        np_model = create_noise_predictor_model(cfg)
        if args.distill_ckpt:
            distill_ckpt = torch.load(args.distill_ckpt, map_location=torch.device(cfg.device))
            model.load_state_dict(distill_ckpt)
        trainer = npTrainer(np_model=np_model, model=model, dataset=dataset_cls, cfg=cfg, logger=logger,
                            tb_logger=tb_logger)
        trainer.train()
    elif args.mode == 'eval':
        if args.ckpt:
            ckpt = torch.load(args.ckpt, map_location=torch.device(cfg.device))
            model.load_state_dict(ckpt)
        trainer = Trainer(model=model, dataset=dataset_cls, cfg=cfg, logger=logger, tb_logger=tb_logger)
        trainer.evaluation()
    elif args.mode == 'distill_eval':
        if args.distill_ckpt:
            distill_ckpt = torch.load(args.distill_ckpt, map_location=torch.device(cfg.device))
            model.load_state_dict(distill_ckpt)
        trainer = Trainer(model=model, dataset=dataset_cls, cfg=cfg, logger=logger, tb_logger=tb_logger)
        trainer.evaluation()
    elif args.mode == 'np_eval':
        if args.distill_ckpt:
            distill_ckpt = torch.load(args.distill_ckpt, map_location=torch.device(cfg.device))
            model.load_state_dict(distill_ckpt)
        np_model = create_noise_predictor_model(cfg)
        if args.np_ckpt:
            np_ckpt = torch.load(args.np_ckpt, map_location=torch.device(cfg.device))
            np_model.load_state_dict(np_ckpt)
        trainer = npTrainer(np_model=np_model, model=model, dataset=dataset_cls, cfg=cfg, logger=logger,
                            tb_logger=tb_logger)
        trainer.evaluation()