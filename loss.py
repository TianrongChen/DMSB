
import torch
import util
from ipdb import set_trace as debug

def compute_sb_DSB_train(opt, label, label_aux,dyn, ts, ms, policy_opt, return_z=False, itr=None):
    """ Implementation of Eq (18,19) in our main paper.
    """
    dt      = dyn.dt
    zs = policy_opt(ms,ts)
    g_ts = dyn.g(ts)
    g_ts = g_ts[:,None,None,None] if util.is_image_dataset(opt) else g_ts[:,None]
    loss = torch.nn.functional.mse_loss(g_ts*dt*zs,label)
    return loss, zs if return_z else loss
