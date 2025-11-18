import torch
from torch import nn
import math
import numpy as np
from tqdm import tqdm


def sqrt_beta_schedule(timesteps, s=0.0001):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = 1 - torch.sqrt(t + s)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sine_noise_schedule(timesteps,beta_min, beta_max):
    t = torch.linspace(0, timesteps, timesteps)
    beta_t = beta_min + (beta_max - beta_min) * (0.5 * (1 + torch.sin(math.pi * (t / timesteps - 0.5))))
    return beta_t

def sigmoid_beta_schedule(timesteps, start=-3., end=3., tau=0.7, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class var_schedulr:
    def __init__(self, noise_steps=1000,sample_steps = 200, m_min = 0.001, m_max = 0.999, scheduler='Linear',device = "cuda:0",var_scale = 1,max_var = 1,objective = "grad"):
        self.scheduler = scheduler
        self.noise_steps = noise_steps
        self.m_min = m_min
        self.m_max = m_max
        self.device = device
        self.max_var = max_var
        self.m_t = self.prepare_noise_schedule().to(device)
        self.m_tminus = torch.cat((torch.tensor([0], dtype=torch.float32).to(device), self.m_t[:-1]))
        self.variance_t = 2. * (self.m_t - self.m_t.pow(2)) * self.max_var
        self.variance_tminus = torch.cat((torch.tensor([0.], dtype=torch.float32).to(device), self.variance_t[:-1]))
        self.variance_t_tminus = self.variance_t - self.variance_tminus * ((1. - self.m_t) / (1. - self.m_tminus)).pow(2)
        self.posterior_variance_t = self.variance_t_tminus * self.variance_tminus / self.variance_t
        midsteps = torch.arange(self.noise_steps - 1, 1,step=-((self.noise_steps - 1) / (sample_steps - 2))).long()
        self.sample_steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
        self.var_scale = var_scale
        self.obj = objective

    def prepare_noise_schedule(self):
        if self.scheduler == 'Linear':
            return torch.linspace(self.m_min, self.m_max, self.noise_steps, dtype=torch.float32)
        elif self.scheduler == 'Cosine':
            return cosine_beta_schedule(self.noise_steps)
        # elif self.scheduler == 'Sqrt':
        #     return sqrt_beta_schedule(self.noise_steps)
        # elif self.scheduler == 'Sigmoid':
        #     return sigmoid_beta_schedule(self.noise_steps)
        elif self.scheduler == "Sin":
            m_t = 1.0075 ** torch.linspace(0, self.noise_steps, self.noise_steps, dtype=torch.float32)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
            return m_t
        elif self.scheduler == "Sine":
            return sine_noise_schedule(self.noise_steps,self.m_min,self.m_max)
        else:
            raise NotImplementedError(f"unknown scheduler: {self.scheduler}")



    def noise_motion(self, past_dct, y_dct,t):
        noise = torch.randn_like(y_dct)
        m_t = self.extract(self.m_t, t, y_dct.shape)
        var_t = self.extract(self.variance_t, t, y_dct.shape)
        sigma_t = torch.sqrt(var_t)
        objective = None
        if self.obj== 'grad':
            objective = m_t * (past_dct - y_dct) + sigma_t * noise
        elif self.obj == 'noise':
            objective = noise
        elif self.obj == 'x0':
            objective = y_dct
        elif self.obj == 'mx0':
            objective = m_t * - y_dct + sigma_t * noise
        return (1. - m_t) * y_dct + m_t * past_dct + sigma_t * noise, objective

    def extract(self,a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def getnext(self,x0_recon,x_t,y,t,n_t):
        m_t = self.extract(self.m_t, t, x_t.shape)
        m_nt = self.extract(self.m_t, n_t, x_t.shape)
        var_t = self.extract(self.variance_t, t, x_t.shape)
        var_nt = self.extract(self.variance_t, n_t, x_t.shape)
        sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
        sigma_t = torch.sqrt(sigma2_t)
        noise = torch.randn_like(x0_recon)
        # x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + var_nt * torch.randn_like(y)
        x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                        (x_t - (1. - m_t) * x0_recon - m_t * y)
        return x_tminus_mean + self.var_scale * sigma_t * noise


    def predict_x0_from_objective(self,y_noise_dct, past_dct, t, objective_recon):
        if self.obj == 'grad':
            x0_recon = y_noise_dct - objective_recon
            return x0_recon
        elif self.obj == 'noise':
            m_t = self.extract(self.m_t, t, y_noise_dct.shape)
            var_t = self.extract(self.variance_t, t, y_noise_dct.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (y_noise_dct - m_t * past_dct - sigma_t * objective_recon) / (1. - m_t)
            return x0_recon
        elif self.obj == "x0":
            return objective_recon
        elif self.obj == 'mx0':
            m_t = self.extract(self.m_t, t, y_noise_dct.shape)
            x0_recon = y_noise_dct - objective_recon  - m_t * past_dct
            return x0_recon

class diffusion_var_schedulr:
    def __init__(self, noise_steps = 20, sample_steps = 20, etas_min = 0.04,  etas_max=0.99, scheduler='exponential',kappa = 1.0,power = 0.3, device="cuda:0",objective = "x0"):
        self.scheduler = scheduler
        self.noise_steps = noise_steps
        self.etas_min = etas_min
        self.etas_max = etas_max
        self.device = device
        self.sqrt_etas = torch.tensor(self.get_eta_schedule(
            scheduler,
            num_diffusion_timesteps = noise_steps,
            min_noise_level = etas_min,
            etas_end = etas_max,
            kappa = kappa,
            power = power
        ),dtype=torch.float32,device = device)
        self.etas = self.sqrt_etas ** 2
        self.etas_prev = torch.cat((torch.tensor([0.0],dtype=torch.float32,device = device), self.etas[:-1]))
        self.alpha = self.etas - self.etas_prev
        self.posterior_variance = kappa ** 2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = torch.cat(
            (torch.tensor([self.posterior_variance[1]], dtype=torch.float32, device = device), self.posterior_variance[1:])
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance_clipped)
        self.ddim_coef1 = self.etas_prev * self.etas
        self.ddim_coef2 = self.etas_prev / self.etas
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas
        self.obj = objective
        self.kappa = kappa
        if sample_steps == noise_steps:
            self.sample_steps = torch.arange(self.noise_steps - 1, -1, step=-1).long()
        elif sample_steps == 1:
            self.sample_steps = torch.Tensor([self.noise_steps - 1]).long()
        elif sample_steps == 2:
            self.sample_steps = torch.Tensor([self.noise_steps - 1, 0]).long()
        else:
            midsteps = torch.arange(self.noise_steps - 1, 1, step=-((self.noise_steps - 1) / (sample_steps - 2))).long()
            self.sample_steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)

    def get_consist_sample(num = 1):
        if num == 1:
            return

    def scalings_for_boundary_conditions(self,timestep: torch.Tensor, sigma_data: float = 0.5, timestep_scaling: float = 10.0) -> tuple:
        c_skip = sigma_data ** 2 / ((timestep * timestep_scaling) ** 2 + sigma_data ** 2)
        c_out = (timestep * timestep_scaling) / ((timestep * timestep_scaling) ** 2 + sigma_data ** 2) ** 0.5
        return c_skip, c_out

    def get_eta_schedule(self,
            schedule_name,
            num_diffusion_timesteps,
            min_noise_level,
            etas_end=0.99,
            kappa=1.0,
            power=0.3):
        """
        Get a pre-defined eta schedule for the given name.

        The eta schedule library consists of eta schedules which remain similar
        in the limit of num_diffusion_timesteps.
        """
        if schedule_name == 'exponential':
            etas_start = min(min_noise_level / kappa, min_noise_level)
            increaser = math.exp(1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start))
            base = np.ones([num_diffusion_timesteps, ]) * increaser
            power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
            power_timestep *= (num_diffusion_timesteps - 1)
            sqrt_etas = np.power(base, power_timestep) * etas_start
            return sqrt_etas
        if schedule_name == "linear":
            sqrt_etas = np.linspace(min_noise_level, etas_end, num = num_diffusion_timesteps)
            return sqrt_etas
        return None

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def prior_sample(self,past_dct,noise = None):
        if noise is None:
            noise = torch.randn_like(past_dct)
        t = torch.tensor([self.sample_steps[0], ] * past_dct.shape[0], device=past_dct.device)
        return past_dct + self.extract(self.kappa * self.sqrt_etas,t,past_dct.shape) * noise

    def noise_motion(self, past_dct, predict_dct, t, noise = None):
        if noise is None:
            noise = torch.randn_like(predict_dct)
        etas = self.extract(self.etas, t, predict_dct.shape)
        sqrt_etas = self.extract(self.kappa * self.sqrt_etas, t, predict_dct.shape)
        return etas * (past_dct - predict_dct) + predict_dct + sqrt_etas * noise

    def q_posterior_mean_variance(self, x_0, x_t, t):
        mean = self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_t + self.extract(self.posterior_mean_coef2, t, x_0.shape) * x_0
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean,posterior_variance,posterior_log_variance_clipped

    def ddim_q_posterior_mean_variance(self, x_t, t):
        ddim_coef1 = self.extract(self.ddim_coef1, t, x_t.shape)  # etas_pre*etas
        ddim_coef2 = self.extract(self.ddim_coef2, t, x_t.shape)  # etas_pre/etas
        etas_prev = self.extract(self.etas_prev, t, x_t.shape)
        k = (1 - etas_prev + torch.sqrt(ddim_coef1) - torch.sqrt(ddim_coef2))
        m = torch.sqrt(ddim_coef2)
        j = (etas_prev - torch.sqrt(ddim_coef1))
        return k,m,j

    def ddim_q_posterior(self,x_t,t,n_t):
        etas = self.extract(self.etas,t,x_t.shape)
        etas_prev = self.extract(self.etas,n_t,x_t.shape)
        ddim_coef1 = etas * etas_prev
        ddim_coef2 = etas_prev / etas
        k = (1 - etas_prev + torch.sqrt(ddim_coef1) - torch.sqrt(ddim_coef2))
        m = torch.sqrt(ddim_coef2)
        j = (etas_prev - torch.sqrt(ddim_coef1))
        return k,m,j
