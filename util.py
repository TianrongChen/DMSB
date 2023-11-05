import os,sys,re
from posixpath import split
#import phate
import numpy as np
import shutil
import termcolor
import pathlib
from scipy import linalg
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.utils as tu
from torch.nn.functional import adaptive_avg_pool2d
import sklearn.decomposition # PCA
import sklearn.manifold # t-SNE
import matplotlib
from matplotlib.lines import Line2D
try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from ipdb import set_trace as debug


# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def is_image_dataset(opt):
    return opt.problem_name in ['mnist','cifar10','celebA32','celebA64']

def is_toy_dataset(opt):
    return opt.problem_name in ['gmm','checkerboard', 'moon-to-spiral','semicircle', 'RNAsc','navi','petal','gaussian2']

def use_vp_sde(opt):
    return opt.sde_type == 'vp'

def evaluate_stage(opt, stage):
    """ Determine what metrics to evaluate for the current stage,
    if metrics is None, use the frequency in opt to decide it.
    """
    match = lambda freq: (freq>0 and stage%freq==0)
    return [match(opt.snapshot_freq), match(opt.ckpt_freq)]

def get_time(sec):
    h = int(sec//3600)
    m = int((sec//60)%60)
    s = sec%60
    return h,m,s

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def flatten_dim01(x):
    # (dim0, dim1, *dim2) --> (dim0x1, *dim2)
    return x.reshape(-1, *x.shape[2:])

def unflatten_dim01(x, dim01):
    # (dim0x1, *dim2) --> (dim0, dim1, *dim2)
    return x.reshape(*dim01, *x.shape[1:])

def compute_z_norm(zs, dt):
    # Given zs.shape = [batch, timesteps, *z_dim], return E[\int 0.5*norm(z)*dt],
    # where the norm is taken over z_dim, the integral is taken over timesteps,
    # and the expectation is taken over batch.
    zs = zs.reshape(*zs.shape[:2],-1)
    return 0.5 * zs.norm(dim=2).sum(dim=1).mean(dim=0) * dt

def create_traj_sampler(trajs):
    for traj in trajs:
        yield traj

def get_load_it(load_name):
    nums = re.findall('[0-9]+', load_name)
    assert len(nums)>0
    if 'stage' in load_name and 'dsm' in load_name:
        return int(nums[-2])
    return int(nums[-1])

def restore_checkpoint(opt, runner, load_name):
    assert load_name is not None
    print(green("#loading checkpoint {}...".format(load_name)))
    full_keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b','v_dists']

    with torch.cuda.device(opt.gpu):
        checkpoint = torch.load(load_name,map_location=opt.device)
        ckpt_keys=[*checkpoint.keys()]
        for k in ckpt_keys:
            if k =='v_dists':
                runner.v_dists = checkpoint['v_dists']
                print('load v successfullly')
            else:
                getattr(runner,k).load_state_dict(checkpoint[k])

    if len(full_keys)!=len(ckpt_keys):
        value = { k for k in set(full_keys) - set(ckpt_keys) }
        print(green("#warning: does not load model for {}, check is it correct".format(value)))
    else:
        print(green('#successfully loaded all the modules'))

        # Note: Copy the avergage parameter to policy. This seems to improve performance for
        # DSM warmup training (yet not sure whether it's true for SB in general)
        # runner.ema_f.copy_to()
        # runner.ema_b.copy_to()
        # print(green('#loading form ema shadow parameter for polices'))

    print(magenta("#######summary of checkpoint##########"))

def save_checkpoint(opt, runner, keys, stage_it, dsm_train_it=None):
    checkpoint = {}
    fn = opt.ckpt_path + "/stage_{0}{1}.npz".format(
        stage_it, '_dsm{}'.format(dsm_train_it) if dsm_train_it is not None else ''
    )
    with torch.cuda.device(opt.gpu):
        for k in keys:
            if k =='v_dists':
                checkpoint[k] = runner.v_dists
            else:
                checkpoint[k] = getattr(runner,k).state_dict()
        torch.save(checkpoint, fn)
    print(green("checkpoint saved: {}".format(fn)))

def save_PCA_traj(opt, fn, traj, n_snapshot=None, direction=None,sampled_data=None):
    fn_pdf = os.path.join('results', opt.dir, fn+'{}.pdf'.format('' if sampled_data is None else '1step'))
    n_snapshot=opt.num_dist
    total_steps = opt.interval
    sample_steps= np.linspace(0, total_steps-1, n_snapshot).astype(int)
    if sampled_data is None:
        data = traj[:, sample_steps, :] 
        data = data.transpose(1,0,2)
        data = data.reshape(-1, data.shape[-1])
    else:
        data = sampled_data

    color_bar = sample_steps
    batch = opt.samp_bs
    color_bar = color_bar.repeat(batch)/total_steps

    xs,vs = split_joint(opt, data)
    Y_pca1 = xs[:,0:2]
    Y_pca2 = xs[:,2:4]

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20, 10))
    # cmap =['Spectral','Spectral']
    cmap = 'Spectral'
    #plotting PCA
    ax1.scatter(Y_pca1[:,0],Y_pca1[:,1], s=5, c=color_bar,cmap=cmap, vmin=0, vmax=1)
    # ax1.scatter(Y_tsne[:,0],Y_tsne[:,1], s=5, c=color_bar,cmap=cmap, vmin=0, vmax=1)
    ax2.scatter(Y_pca2[:,0],Y_pca2[:,1], s=5, c=color_bar,cmap=cmap, vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(fn_pdf)
    plt.clf()


def set_ax(ax):
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_xticks([])
    ax.set_yticks([])
    
def save_PCA_traj2(opt, fn, data,gt):
    fn_npy = os.path.join('results', opt.dir, 'replaced_traj.npy')
    fn_pdf = os.path.join('results', opt.dir, fn+'.pdf')
    fig, axs = plt.subplots(4,len(data), figsize=(40, 25))
    # for i in range
    for ii,ax in enumerate(axs[0]):
        set_ax(ax)
        ax.scatter(data[ii][:,0],data[ii][:,1],s=3)

    for ii,ax in enumerate(axs[1]):
        set_ax(ax)
        ax.scatter(data[ii][:,2],data[ii][:,3],s=3)

    for ii,ax in enumerate(axs[2]):
        set_ax(ax)
        ax.scatter(gt[ii][:,0],gt[ii][:,1],s=3)

    for ii,ax in enumerate(axs[3]):
        set_ax(ax)
        ax.scatter(gt[ii][:,2],gt[ii][:,3],s=3)


    plt.tight_layout()
    plt.savefig(fn_pdf)
    np.save(fn_npy,np.array(data))
    plt.clf()


def viz(opt, fn, ms, n_snapshot, direction):
    plt_fnc={'semicircle':save_toy_traj,
            'petal':save_petal_traj,
            'gmm':save_toy_seg_traj}.get(opt.problem_name)
    plt_fnc(opt, fn, ms, n_snapshot, direction)
    
def plot_traj(data, num_samp,ax,edg=None,cmap='viridis',dim=[0,1],alpha=1):
    total_sample    = data.shape[1]
    bs_idx          = np.linspace(0,data.shape[1]-1,num_samp).astype(int)
    num_marg        = data.shape[0]
    _colors = np.linspace(0,1,num_marg)
    for ii in range(num_marg):
        ax.scatter(data[ii,bs_idx,dim[0]],data[ii,bs_idx,dim[1]], s=250, c=_colors[ii].repeat(bs_idx.shape[0]),cmap=cmap, vmin=0, vmax=1,edgecolors=edg,alpha=alpha)

def preprocess_whole_traj(data, num_marg):
    '''
    input [bs,timesteps,dim]
    output [marg, bs, dim]
    '''
    total_step      = data.shape[1]
    sample_steps    = np.linspace(0, total_step-1, num_marg).astype(int)
    data            = data.transpose(1,0,2)
    data            = data[sample_steps,...]
    return data

def save_petal_traj(opt, fn, traj, n_snapshot=None, direction=None):
    lims            = [-6, 6]
    data            = []
    fn_npy          = os.path.join('results', opt.dir, 'replaced_traj.npy')
    fn_pdf          = os.path.join('results', opt.dir, fn+'.pdf')
    fnt_size        = 80
    fig, axss = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)
    path_timesteps  = 100
    traj_timesteps  = 5
    plot_bs         = 80
    path            = preprocess_whole_traj(traj,path_timesteps)
    traj            = preprocess_whole_traj(traj,traj_timesteps)
    for name,ax,data in zip(['Samples','Traj'],axss,[traj,path]):
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.grid()
        # set_ax(ax)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        plot_traj(data,plot_bs,ax)
        ax.set_title(name,size=fnt_size)

    _colors = np.linspace(0,1,traj.shape[0])
    lgd_list=[]
    for j in range(len(_colors)):
        col     = _colors[j]
        rgba    = matplotlib.cm.get_cmap('viridis')(col)
        _name   =r'$t_{}$'.format(j)
        lgd_list.append(Line2D([0], [0], marker='o', color='w', label=_name,
                            markerfacecolor=rgba, markersize=40))

    fig.legend(handles=lgd_list, loc='upper right', bbox_to_anchor=(1.11,0.7), ncol=1,fancybox=True, shadow=True, prop={'size': 50} )
                            
    fig.tight_layout()
    plt.savefig(fn_pdf)
    np.save(fn_npy,traj)
    plt.clf()

def save_toy_traj(opt, fn, traj, n_snapshot=None, direction=None):
    #form of traj: [bs, interval, x_dim=2]
    # fn_npy = os.path.join('results', opt.dir, fn+'.npy')
    fn_npy = os.path.join('results', opt.dir, 'replaced_traj.npy')
    fn_pdf = os.path.join('results', opt.dir, fn+'.pdf')
    n_snapshot=opt.num_dist
    lims = get_lims(opt)

    total_steps = traj.shape[1]
    sample_steps= np.linspace(0, total_steps-1, n_snapshot).astype(int)
    traj_steps  = np.linspace(0, total_steps-1, 10).astype(int)
    if n_snapshot is None: # only store t=0
        plt.scatter(traj[:,0,0],traj[:,0,1], s=5)
        plt.xlim(*lims)
        plt.ylim(*lims)
    else:
        # sample_steps=np.concatenate(res)
        fig, axss = plt.subplots(1, 2)
        fig.set_size_inches(20, 10)
        # col = np.arange(traj.shape[0])
        # cmap=['viridis','plasma']
        cmap =['Spectral','Spectral'] if opt.problem_name =='RNAsc' else ['Blues','Reds']
        colors= ['b','r']
        # means=traj.mean(axis=0)
        num_samp_lines = 20
        random_idx = np.random.choice(traj.shape[0], num_samp_lines, replace=False)
        means=traj[random_idx,...]
        for i in range(2):
            ax=axss[i]
            _colors = np.linspace(0.5,1.5,len(traj_steps))
            for idx,step in enumerate(traj_steps):
                ax.scatter(traj[:,step,2*i],traj[:,step,2*i+1], s=5, c=_colors[idx].repeat(traj.shape[0]),cmap='Greys', vmin=0, vmax=2)
                ax.set_xlim(*lims[i])
                ax.set_ylim(*lims[i])

            _colors = np.linspace(0,1,len(sample_steps)) if opt.problem_name =='RNAsc' else np.linspace(0.5,1,len(sample_steps))
            for idx,step in enumerate(sample_steps):
                ax.scatter(traj[:,step,2*i],traj[:,step,2*i+1], s=5, c=_colors[idx].repeat(traj.shape[0]), alpha=0.6,vmin=0, vmax=1,cmap=cmap[i])
                ax.set_xlim(*lims[i])
                ax.set_ylim(*lims[i])
                # ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
            
                # ax.plot(means[:,2*i],means[:,2*i+1],color=colors[i],linewidth=5,alpha=0.5)
            for ii in range(num_samp_lines):
                ax.plot(means[ii,:,2*i],means[ii,:,2*i+1],color=colors[i],linewidth=2,alpha=0.5)
                ax.set_title('position' if i==0 else 'velocity',size=40)
    fig.suptitle('NFE = {}'.format(opt.interval-1),size=40)
    fig.tight_layout()
    plt.savefig(fn_pdf)
    if direction=='forward': np.save(fn_npy,traj)
    plt.clf()

def save_toy_seg_traj(opt, fn, traj, n_snapshot=None, direction=None):
    #form of traj: [bs, interval, x_dim=2]
    fn_npy = os.path.join('results', opt.dir, 'replaced_traj.npy')
    fn_pdf = os.path.join('results', opt.dir, fn+'.pdf')
    n_snapshot=5
    lims = {
        'gmm': [-8, 8],
        'petal': [-8, 8],
        'checkerboard': [-7, 7],
        'moon-to-spiral':[-20, 20],
    }.get(opt.problem_name)

    if n_snapshot is None: # only store t=0
        plt.scatter(traj[:,0,0],traj[:,0,1], s=5)
        plt.xlim(*lims)
        plt.ylim(*lims)
    else:
        total_steps = traj.shape[1]
        # sample_steps= np.linspace(0, total_steps-1, n_snapshot).astype(int)
        dist_idxs   = np.linspace(0, opt.interval-1, opt.num_dist).astype(int)
        res=[]
        for i in range(len(dist_idxs)-1):
            arr = np.linspace(dist_idxs[i],dist_idxs[i+1],n_snapshot).astype(int)
            res.append(arr)
        # sample_steps=np.concatenate(res)
        fig, axss = plt.subplots(2*(opt.num_dist-1), n_snapshot)
        fig.set_size_inches(n_snapshot*6, 20)
        col = np.arange(traj.shape[0])
        cmap=['viridis','plasma']
        for i in range(2):
            for j in range(len(res)):
                axs=axss[i*len(res)+j]
                sample_steps=res[j]
                for ax, step in zip(axs, sample_steps):
                    alpha=1 if step in dist_idxs else 0.1
                    ax.scatter(traj[:,step,2*i],traj[:,step,2*i+1], s=5, c=col,cmap=cmap[i], alpha=alpha)
                    ax.set_xlim(*lims)
                    ax.set_ylim(*lims)
                    ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
        fig.tight_layout()
    plt.savefig(fn_pdf)
    if direction=='forward': 
        np.save(fn_npy,traj)
    plt.clf()




def save_generated_data(opt, x):
    x = norm_data(opt,x)
    x = torch.squeeze(x)
    for i in range(x.shape[0]):
        fn = os.path.join(opt.generated_data_path, 'img{}.jpg'.format(i))
        tu.save_image(x[i,...], fn)


def norm_data(opt,x):
    if opt.problem_name=='mnist':
        x=x.repeat(1,3,1,1)
    _max=torch.max(torch.max(x,dim=-1)[0],dim=-1)[0][...,None,None]
    _min=torch.min(torch.min(x,dim=-1)[0],dim=-1)[0][...,None,None]
    x=(x-_min)/(_max-_min)
    return x

def check_duplication(opt):
    plt_dir='plots/'+opt.dir
    ckpt_dir='checkpoint/'+opt.group+'/'+opt.name
    runs_dir='runs/'+opt.log_fn
    plt_flag=os.path.isdir(plt_dir)
    ckpt_flag=os.path.isdir(ckpt_dir)
    run_flag=os.path.isdir(runs_dir)
    tot_flag= plt_flag or ckpt_flag or run_flag
    print([plt_flag,ckpt_flag,run_flag])
    if tot_flag:
        decision=input('Exist duplicated folder, do you want to overwrite it? [y/n]')

        if 'y' in decision:
            try:
                shutil.rmtree(plt_dir)
            except:
                pass
            try: 
                shutil.rmtree(ckpt_dir)
            except:
                pass
            try:
                shutil.rmtree(runs_dir)
            except:
                pass
        else:
            sys.exit()


def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]




def split_joint(opt, m):
    samp_bs = m.shape[0]
    single_dim = opt.data_dim[0]
    x       = m[:,0:single_dim,...]
    v       = m[:,single_dim:,...]
    return x,v

def time_sample(_interval, direction,num_samp):
    if direction=='backward':
        return torch.randint(1,_interval,(num_samp,))
    else:
        return torch.randint(0,_interval-1,(num_samp,))
    
def time_arange(_interval, direction):
    if direction=='backward':
        return torch.arange(1,_interval).cpu()
    else:
        return torch.arange(0,_interval-1).cpu()

def get_idx_npy(arr,value):
    return torch.where(arr==value)[0][0].item()

def get_func_mesher(opt, ts, grid_n, func, out_dim=2):
    if func is None: return None
    lims = get_lims(opt)[0]
    X1, X2, XS = create_mesh(opt, grid_n, lims)
    out_shape = [grid_n,grid_n] if out_dim==1 else [grid_n,grid_n,out_dim]

    def mesher():
        arg_xs = XS.detach()
        # arg_ts = ts[0].repeat(grid_n**2).detach()
        arg_ts = ts.repeat(grid_n**2).detach()
        arg_ts = arg_ts[:,None,None,None] if is_image_dataset(opt) else arg_ts[:,None]
        fn_out = func(arg_xs, arg_ts)
        return X1, X2, to_numpy(fn_out.reshape(*out_shape))

    return mesher

def get_lims(opt):
    lims = {
        'gmm': [[-10, 10],[-40, 40]],
        'RNAsc': [[-35, 35],[-100, 100]],
        'semicircle':[[-10, 10],[-40, 40]],
        'petal':[[-7, 7],[-7,-7]],
        'checkerboard': [-7, 7],
        'moon-to-spiral':[-20, 20],
        'navi':[[-5, 5],[-3, 3]],
    }.get(opt.problem_name)
    return lims

def to_numpy(t):
    return t.detach().cpu().numpy()

def create_mesh(opt, n_grid, lims, convert_to_numpy=True):
    import warnings

    _x = torch.linspace(*(lims+[n_grid]))

    # Suppress warning about indexing arg becoming required.
    with warnings.catch_warnings():
        X, Y = torch.meshgrid(_x, _x)

    xs = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(opt.device)
    return [to_numpy(X), to_numpy(Y), xs] if convert_to_numpy else [X, Y, xs]

