import numpy as np
import pandas as pd
import torch
import torch.distributions as td
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from prefetch_generator import BackgroundGenerator
from sklearn.model_selection import train_test_split
import util
from ipdb import set_trace as debug

def get_data_dim(opt, problem_name):
    return {
        'gmm':          [2],
        'semicircle':   [2],
        'checkerboard': [2],
        'RNAsc':        [opt.RNA_dim],
        'petal':        [2],
    }.get(problem_name)

class Sampler:
    def __init__(self, distribution, batch_size, device):
        self.distribution = distribution
        self.batch_size = batch_size
        self.device = device

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def sample(self, batch):
        if batch is None:
            batch = self.batch_size
        return self.distribution.sample([batch]).to(self.device)

def build(opt):
    print(util.magenta("build problem..."))

    opt.data_dim = get_data_dim(opt, opt.problem_name)
    
    distribution_builder = {
        'gmm':          gmm_builder,
        'semicircle':   SemiCircle_builder,
        'RNAsc':        RNAsc_builder,
        'navi':         navi_builder,
        'petal':        Petal_builder,
    }.get(opt.problem_name)

    dists           = distribution_builder(opt)
    opt.num_dist    = len(dists)
    return dists

def gmm_builder(opt):
    assert opt.problem_name == 'gmm'
    # ----- pT -----
    rad=4
    var=0.1
    dists           = [
                        UniGaussian(opt,[-0,0],var= var),
                        MixMultiVariateNormal(opt.samp_bs,num=4,radius=rad,var=var),
                        MixMultiVariateNormal(opt.samp_bs,num=8,radius=rad,var=var),
                        UniGaussian(opt,[0,0],var= var),
                    ]
    
    return dists

def navi_builder(opt):
    assert opt.problem_name == 'navi'
    # ----- pT -----
    dists           = [
                        UniGaussian(opt,[-3,0],var=0.1),
                        UniGaussian(opt,[-2,0],  var=0.1),
                        UniGaussian(opt,[0,2],  var=0.1),
                        UniGaussian(opt,[2,0],  var=0.1),
                    ]
    
    return dists
    
def gmm_builder(opt):
    assert opt.problem_name == 'gmm'
    # ----- pT -----
    rad=4
    var=0.15
    dists           = [
                        UniGaussian(opt,[0,0],var= var),
                        MixMultiVariateNormal(opt.samp_bs,num=4,radius=rad,var=var),
                        MixMultiVariateNormal(opt.samp_bs,num=8,radius=rad,var=var),
                        UniGaussian(opt,[0,0],var= var),
                    ]
    
    return dists


def threemode(opt):
    # ----- pT -----
    rad=2
    var=0.1
    dists           = [
                        UniGaussian(opt,[-0,0],var= torch.Tensor([[1,0],[0,0.1]])),
                        MixMultiVariateNormal(opt.samp_bs,num=2,radius=rad,var=var,bias=torch.Tensor([0,3])),
                    ]
    
    return dists

def navi_builder(opt):
    assert opt.problem_name == 'navi'
    # ----- pT -----
    dists           = [
                        UniGaussian(opt,[-3,0],var=0.1),
                        UniGaussian(opt,[-2,0],  var=0.1),
                        UniGaussian(opt,[0,2],  var=0.1),
                        UniGaussian(opt,[2,0],  var=0.1),
                    ]
    
    return dists

def SemiCircle_builder(opt):
    assert opt.problem_name == 'semicircle'
    num     = 4
    # arc     = np.pi/(num-1)
    # vars    = [0.2*(ii+1) for ii in range(num)]
    # xs      = [np.cos(arc*idx)*5 for idx in range(num)]
    # ys      = [np.sin(arc*idx)*5 for idx in range(num)]
    # dists   = [UniGaussian(opt,[x,y],var = var) for x,y,var in zip (xs,ys, vars)]
    vars    = [0.1 for ii in range(num)]
    dists   = [UniGaussian(opt,means,var = var) for means,var in zip ([[-2,0],[-2,4],[2,4],[2,0]], vars)]
    return dists


def Petal_builder(opt):
    df      = make_diamonds(4000, 0.25, 5)
    df      = np.array(df)
    df[:,0] = df[:,0]-1
    df      = df.astype('float32')
    timestamps  = df[:,0]

    tokens      = [0,1,2,3,4]

    datasets    = []
    for token in tokens:
        data= df[np.where(token == timestamps)][:,1:]
        data= data*5

        datasets.append(data)
    dists       = [DataSampler(dataset, opt.samp_bs, opt.device) for dataset in datasets]
    return dists

def RNAsc_builder(opt):
    assert opt.problem_name == 'RNAsc'
    _dict       = np.load('data/RNAsc/ProcessedData/eb_velocity_v5.npz')
    datas        = _dict['pcs']
    scaler      = StandardScaler() #Same as TrajectoryNet implementation. see Ln332 in https://github.com/KrishnaswamyLab/TrajectoryNet/blob/master/TrajectoryNet/dataset.py.
    scaler.fit(datas)
    datas       = scaler.transform(datas)
    datas       = datas[:,0:opt.RNA_dim]
    datas       = datas.astype('float32')
    timestamps  = _dict['sample_labels']
    tokens       = np.arange(0,opt.num_marg)
    datasets    = [datas[np.where(token == timestamps)] for token in tokens]
    dists       = [DataSampler(dataset, opt.samp_bs, opt.device) for dataset in datasets]
    return dists


################################
#######Utility Functions########
################################
class UniGaussian:
    def __init__(self, opt, mean, var=1.):

        # build mu's and sigma's
        self.batch_size = opt.samp_bs
        var             = var*torch.eye(opt.data_dim[-1]) if isinstance(var,float) else var
        self.dist       = td.MultivariateNormal(torch.Tensor(mean), var)

    def sample(self):
        samples= self.dist.sample([self.batch_size])
        return samples



class MixMultiVariateNormal:
    def __init__(self, batch_size, radius=6, num=4, sigmas=None, var=1, mean=None,bias=0):


        self.bias=bias
        if mean is not None:
            mus = mean
        else: 
            # build mu's and sigma's
            arc = 2*np.pi/num
            xs = [np.cos(arc*idx)*radius for idx in range(num)]
            ys = [np.sin(arc*idx)*radius for idx in range(num)]
            mus = [torch.Tensor([x,y]) for x,y in zip(xs,ys)]

        dim = len(mus[0])
        sigmas = [var*torch.eye(dim) for _ in range(num)] if sigmas is None else sigmas

        if batch_size%num!=0:
            raise ValueError('batch size must be devided by number of gaussian')
        self.num = num
        self.batch_size = batch_size
        self.dists=[
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]

    def log_prob(self,x):
        # assume equally-weighted
        densities=[torch.exp(dist.log_prob(x)) for dist in self.dists]
        return torch.log(sum(densities)/len(self.dists))
    def sample(self):
        ind_sample = self.batch_size/self.num
        samples=[dist.sample([int(ind_sample)]) for dist in self.dists]
        samples=torch.cat(samples,dim=0)
        samples=samples+self.bias
        return samples

#For sanity check, the GMM should work w/o the predefined the clustered velocity##############
# class MixMultiVariateNormalStandard:
#     def __init__(self, batch_size, radius=6, num=4, sigmas=None, var=1, mean=None,bias=0):
#         arc = 2 * np.pi / num
#         xs = [np.cos(arc * idx) * radius for idx in range(num)]
#         ys = [np.sin(arc * idx) * radius for idx in range(num)]
#         self.batch_size=batch_size
#         mix = td.Categorical(
#             torch.ones(
#                 num,
#             )
#         )
#         comp = td.Independent(td.Normal(torch.Tensor([[x, y] for x, y in zip(xs, ys)]), var*torch.ones(num, 2)), 1)
#         self.dists = td.MixtureSameFamily(mix, comp)


#     def log_prob(self,x):
#         # assume equally-weighted
#         densities=[torch.exp(dist.log_prob(x)) for dist in self.dists]
#         return torch.log(sum(densities)/len(self.dists))
#     def sample(self):
#         samples=self.dists.sample([self.batch_size])
#         return samples
#For sanity check, the GMM should work w/o the predefined the clustered velocity##############

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def setup_loader(dataset, batch_size):
    g=torch.Generator(device='cuda')
    sampler=torch.utils.data.RandomSampler(dataset,replacement=True,num_samples=batch_size,generator=torch.Generator(device='cuda'))
    train_loader = DataLoaderX(
                                dataset, 
                                batch_size=batch_size,
                                # shuffle=True,
                                num_workers=0,
                                drop_last=True,
                                sampler=sampler,
                                generator=g,)
    print("number of samples: {}".format(len(dataset)))

    # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/image_datasets.py#L52-L53
    # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/train_util.py#L166
    while True:
        yield from train_loader


class DataSampler: # a dump data sampler
    def __init__(self, dataset, batch_size, device,ratio=0.15):
        self.num_sample = len(dataset)
        #Vanilla split dataste
        train_idx, val_idx  = train_test_split(list(range(len(dataset))), test_size=ratio)
        #Training data
        self.dataloader = setup_loader(dataset[train_idx, ...], batch_size)
        #whole dataset for evaluate as ground truth
        self.ground_truth = dataset
        # test sample
        self.test_sample = dataset[val_idx,...]

        self.batch_size = batch_size
        self.device = device

    def sample(self):
        data = next(self.dataloader)
        return data.to(self.device)

def construct_diamond(
    points_per_petal:int=200,
    petal_width:float=0.25,
    direction:str='y'
):
    '''
    Arguments:
    ----------
        points_per_petal (int). Defaults to `200`. Number of points per petal.
        petal_width (float): Defaults to `0.25`. How narrow the diamonds are.
        direction (str): Defaults to 'y'. Options `'y'` or `'x'`. Whether to make vertical
            or horizontal diamonds.
    Returns:
    ---------
        points (numpy.ndarray): the 2d array of points. 
    '''
    n_side  = int(points_per_petal/2)
    axis_1  = np.concatenate((
                np.linspace(0, petal_width, int(n_side/2)), 
                np.linspace(petal_width, 0, int(n_side/2))
            ))
    axis_2  = np.linspace(0, 1, n_side)
    axes    = (axis_1, axis_2) if direction == 'y' else (axis_2, axis_1)
    points  = np.vstack(axes).T
    points  = np.vstack((points, -1*points))
    points  = np.vstack((points, np.vstack((points[:, 0], -1*points[:, 1])).T))
    return points

def make_diamonds(
    points_per_petal:   int=200,
    petal_width:        float=0.25,
    colors:             int=5,
    scale_factor:       float=30,
    use_gaussian:       bool=True   
):
    '''
    Arguments:
    ----------
        points_per_petal (int). Defaults to `200`. Number of points per petal.
        petal_width (float): Defaults to `0.25`. How narrow the diamonds are.
        colors (int): Defaults to `5`. The number of timesteps (colors) to produce.
        scale_factor (float): Defaults to `30`. How much to scale the noise by 
            (larger values make samller noise).
        use_gaussian (bool): Defaults to `True`. Whether to use random or gaussian noise.
    Returns:
    ---------
        df (pandas.DataFrame): DataFrame with columns `samples`, `x`, `y`, where `samples`
            are the time index (corresponds to colors) 
    '''    
    upper   = construct_diamond(points_per_petal, petal_width, 'y')
    lower   = construct_diamond(points_per_petal, petal_width, 'x')
    data    = np.vstack((upper, lower)) 
    
    noise_fn    = np.random.randn if use_gaussian else np.random.rand
    noise       = noise_fn(*data.shape) / scale_factor
    data        = data + noise
    df          = pd.DataFrame(data, columns=['d1', 'd2'])
    
    c_values        = np.linspace(colors, 1, colors)
    c_thresholds    = np.linspace(1, 0+1/(colors+1), colors)
    
    df.insert(0, 'samples', colors)
    df['samples'] = colors 
    for value, threshold in zip(c_values, c_thresholds):
        index = ((np.abs(df.d1) <= threshold) & (np.abs(df.d2) <= threshold))
        df.loc[index, 'samples'] = value
    df.set_index('samples')
    return df