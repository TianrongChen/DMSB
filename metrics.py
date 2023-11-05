import numpy as np
import torch
import torch.nn.functional as F
from ipdb import set_trace as debug
import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import pairwise_distances
from ot.sliced import sliced_wasserstein_distance,max_sliced_wasserstein_distance
import util

def metric_build(opt):
    metrics = {
        'SWD':sliced_wasserstein_distance,
        'MMD':MMD_loss(),
        'MWD':max_sliced_wasserstein_distance
    }
    return [metrics.get(key) for key in opt.metrics]

def compute_metrics(opt, pred_traj, ref_data, metrics, runner,stage):
    '''
    pred_traj: [batch_size, interval, data_dim] torch.Tensor
    ref_data: [num_dist, batch_size, data_dim], torch.Tensor, we use whole ref data which is similar to FID computation
    The reference data and prediction are all the marignals. We delete the leave one out (--LOO) marginal during the training, but we still evaluate them during here.
    '''
    sample_size     = 1000
    dist_time       = np.linspace(0, opt.interval-1, opt.num_dist).astype(int) #we delete a distribution when LOO during training, so num_dist is same as original marginal
    pred_idx        = np.random.choice(pred_traj.shape[0], sample_size, replace=False) #random sample from batch
    pred_data       = pred_traj[pred_idx][:,dist_time,0:opt.data_dim[0]] # [samp_bs, num_dist, data_dim] 
    pred_data       = pred_data.transpose(1,0,2)/opt.data_scale # [num_dist, samp_bs, data_dim]
    
    for metric_idx, metric in enumerate(metrics): #loop over metrics
        avg_metric  = 0
        for idx,(pred,ref) in enumerate(zip(pred_data, ref_data)):
            if idx==0:
                continue # First marginal does not need to be evaluate. We do not generate it, just ground truth.
            if opt.metrics[metric_idx] == 'MMD': 
                ref_idx = np.random.choice(ref.shape[0], sample_size, replace=False)
                ref     = torch.Tensor(ref[ref_idx])
                pred    = torch.Tensor(pred)

            loss        = metric(pred,ref)
            avg_metric += loss
            print(util.green('{} for time{} is {}'.format(opt.metrics[metric_idx], idx,loss)))
            runner.log_tb(stage, loss, '{}_t{}'.format(opt.metrics[metric_idx],idx),'SB_forward')

        avg_metric = avg_metric/(opt.num_dist-1)
        print('AVERAGE {} IS {}'.format(opt.metrics[metric_idx],avg_metric))
        runner.log_tb(stage, avg_metric, '{}_avg'.format(opt.metrics[metric_idx]), 'SB_forward') 

    return pred_data

class MMD_loss(torch.nn.Module):
    '''
    fork from: https://github.com/ZongxianLee/MMD_Loss.Pytorch
    '''
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss