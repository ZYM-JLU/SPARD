import numpy as np
import torch
def absolute2relative(x, parents, invert=False, x0=None):
    """
    x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert
    x0: [1,..., jn, 3]
    parents: [-1,0,1 ...]
    """
    if not invert:
        xt = x[..., 1:, :] - x[..., parents[1:], :]
        xt = xt / np.linalg.norm(xt, axis=-1, keepdims=True)
        return xt
    else:
        jn = x0.shape[-2]
        limb_l = np.linalg.norm(x0[..., 1:, :] - x0[..., parents[1:], :], axis=-1, keepdims=True)
        xt = x * limb_l
        xt0 = np.zeros_like(xt[..., :1, :])
        xt = np.concatenate([xt0, xt], axis=-2)
        for i in range(1, jn):
            xt[..., i, :] = xt[..., parents[i], :] + xt[..., i, :]
        return xt

def absolute2relative_torch(x, parents, invert=False, x0=None):
    """
    x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert
    x0: [1,..., jn, 3]
    parents: [-1,0,1 ...]
    """
    if not invert:
        xt = x[..., 1:, :] - x[..., parents[1:], :]
        xt = xt / torch.norm(xt, dim=-1, keepdim=True)
        return xt
    else:
        jn = x0.shape[-2]
        limb_l = torch.norm(x0[..., 1:, :] - x0[..., parents[1:], :], dim=-1, keepdim=True)
        xt = x * limb_l
        xt0 = torch.zeros_like(xt[..., :1, :])
        xt = torch.cat([xt0, xt], dim=-2)
        for i in range(1, jn):
            xt[..., i, :] = xt[..., parents[i], :] + xt[..., i, :]
        return xt

def padding_traj(traj, idx_pad):
    traj_pad = traj[..., idx_pad, :]
    return traj_pad


def compute_all_metrics(pred, gt, gt_multi):
    """
    calculate all metrics

    Args:
        pred: candidate prediction, shape as [50, t_pred, 3 * joints_num]
        gt: ground truth, shape as [1, t_pred, 3 * joints_num]
        gt_multi: multi-modal ground truth, shape as [multi_modal, t_pred, 3 * joints_num]

    Returns:
        diversity, ade, fde, mmade, mmfde
    """
    if pred.shape[0] == 1:
        diversity = 0.0
    dist_diverse = torch.pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist_diverse.mean()
    gt_multi = torch.from_numpy(gt_multi).to(gt.device)
    gt_multi_gt = torch.cat([gt_multi, gt], dim=0)
    gt_multi_gt = gt_multi_gt[None, ...]
    pred = pred[:, None, ...]
    diff_multi = pred - gt_multi_gt
    dist = torch.linalg.norm(diff_multi, dim=3)
    # we can reuse 'dist' to optimize metrics calculation
    mmfde, _ = dist[:, :-1, -1].min(dim=0)
    mmfde = mmfde.mean()
    mmade, _ = dist[:, :-1].mean(dim=2).min(dim=0)
    mmade = mmade.mean()
    ade, _ = dist[:, -1].mean(dim=1).min(dim=0)
    fde, _ = dist[:, -1, -1].min(dim=0)
    ade = ade.mean()
    fde = fde.mean()
    return diversity, ade, fde, mmade, mmfde