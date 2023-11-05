from posixpath import split
import numpy as np
import abc
from tqdm import tqdm
from functools import partial
import torch
import time
import util
import loss
from ipdb import set_trace as debug

def _assert_increasing(name, ts):
    assert (ts[1:] > ts[:-1]).all(), '{} must be strictly increasing'.format(name)

def get_ts(opt,ts, rollout, dist_idx):
    init_idx,term_idx       =  rollout
    assert not(opt.LOO==init_idx or opt.LOO==term_idx), 'LOO cannot be the initial or terminal distribution'
    init_t_idx, term_t_idx  = dist_idx[init_idx], dist_idx[term_idx]
    _ts                     = ts[init_t_idx:term_t_idx+1]
    return _ts

def build(opt, x_dists, v_dists):
    print(util.magenta("build base sde..."))
    return SimpleSDE(opt, x_dists, v_dists)


class BaseSDE(metaclass=abc.ABCMeta):
    def __init__(self, opt, x_dists, v_dists):
        self.opt        = opt
        self.dt         = opt.T/opt.interval
        self.dists      = x_dists
        self.ts         = torch.linspace(opt.t0, opt.T, opt.interval)
        self.dist_idx   = np.linspace(0, opt.interval-1, len(self.dists)).astype(int)
        self.dist_ts    = self.ts[self.dist_idx]
        self.bdy_xv     = {'forward':None,'backward':None}
        self.next_reuse_bdy_xv  = False
        self.prev_v_boundary    = v_dists

            
    @abc.abstractmethod
    def _f(self, x, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _g(self, x, t):
        raise NotImplementedError

    def f(self, m, t, direction):
        sign = 1. if direction=='forward' else -1.
        return sign * self._f(m,t)

    def g(self, t):
        return self._g(t)

    def dw(self, m, dt=None):
        dt = self.dt if dt is None else dt
        x,v=util.split_joint(self.opt, m)
        _dw = torch.randn_like(x)*np.sqrt(dt)
        return torch.cat([torch.zeros_like(_dw),_dw],dim=-1)

    def propagate(self, t, m, z, direction, f=None, dw=None, dt=None):
        g = self.g(  t)
        dt = self.dt if dt is None else dt
        x,v = util.split_joint(self.opt, m)
        f = self.f(x,v,t,direction) if f is None else f
        dw = self.dw(x,dt) if dw is None else dw
        z = torch.cat([torch.zeros_like(z,),z],dim=-1)
        return m + (f + g*z)*dt + g*dw

    def propagate_x0_trick(self, x, policy, direction):
        """ propagate x0 by a tiny step """
        t0  = torch.Tensor([0])
        dt0 = self.opt.t0 - 0
        assert dt0 > 0
        z0  = policy(x,t0)
        return self.propagate(t0, x, z0, direction, dt=dt0)


    def sample_xv(self, x_dist, prev_v, t, corrector=None,test=False):
        if test and self.opt.problem_name=='RNAsc':
            xs = torch.Tensor(x_dist.test_sample) #test set is way much smaller than training. We repeat it to have batch size which will not add any information for the fairness of testing. repeat first dimension of xs to be N
            xs = xs.repeat(int(self.opt.samp_bs/xs.shape[0])+1,1)
            xs = xs[0:self.opt.samp_bs,...]
        else:
            xs = x_dist.sample()

        xs = xs*self.opt.data_scale #Constant scaling for the data, default is 1.
        if self.opt.v_sampling=='langevin':
            #Using langevin dynamics to simulate the velocity based on the current ground truth state. eq.9b and proposition 4.5.
            m   = torch.cat([xs,prev_v],dim=-1)
            vs  = self.corrector_langevin_update(t, m, corrector, denoise_xT=False,num_corrector=self.opt.num_corrector_bdy)
        elif self.opt.v_sampling=='gaussian':
            vs  = torch.randn_like(xs)
        else:
            raise RuntimeError
        return torch.cat([xs,vs],dim=1)
    

    def resample_m(self,m,t, idx, corrector):
        opt         = self.opt
        dist_idx    = util.get_idx_npy(self.dist_ts,t)
        print('resampleing position and state from time step {} at time {}'.format(dist_idx, idx))

        _, base_v   = util.split_joint(self.opt,m)
        m           =self.sample_xv(self.dists[dist_idx], base_v, t,corrector=corrector)
        x,v         = util.split_joint(opt, m)
        self.prev_v_boundary[dist_idx] = v
        return m

    def sample_traj(self, 
                    ts, 
                    policy, 
                    corrector=None, 
                    save_traj=True, 
                    rollout=False, 
                    resample=False, 
                    test= False, 
                    ode_drift=None):
        '''
        sample a trajectory from the given policy.

        ts:         time steps to sample, if None, use self.ts
        policy:     a policy object
        corrector:  a corrector object
        save_traj:  whether to save the trajectory
        rollout:    determine the starting point and end point of rollout trajectory
        resample:   whether to resample the position and velocity at intermediate timesteps between the two boundary points of rollout traj
        test:       whether it is the test phase.
        ode_drift:  if not None, use ode_drift to propagate the trajectory
        '''

        # first we need to know whether we're doing forward or backward sampling
        opt         = self.opt
        direction   = policy.direction
        ode_sign    = -1 if direction =='forward' else 1
        assert direction in ['forward','backward']
        loading_dist_idx, saving_dist_idx = {'forward':[rollout[0],rollout[1]],
                                            'backward':[rollout[1],rollout[0]],
                                            }.get(direction)
        # set up ts and init_distribution

        ts  = get_ts(self.opt, ts, rollout, self.dist_idx)
        _assert_increasing('ts', ts)
        assert  rollout[0]<rollout[1]

        if direction == 'backward': ts = torch.flip(ts,dims=[0])


        ms      = torch.empty((opt.samp_bs, len(ts), opt.data_dim[-1]*2)) if save_traj else None
        zs      = torch.empty((opt.samp_bs, len(ts), *opt.data_dim)) if save_traj else None
        label   = torch.empty_like(zs)
        train_ts= torch.zeros(len(ts))
        _ts     =  tqdm(ts,desc=util.yellow("Propagating Dynamics..."))
        for idx, t in enumerate(_ts):
            t_idx = idx if direction=='forward' else len(ts)-idx-1
            if idx==0:
                if self.next_reuse_bdy_xv and test==False:
                    print('-----------resampling from sampled data {}-----------'.format(direction))
                    assert resample
                    m                       = self.bdy_xv[direction]
                    self.next_reuse_bdy_xv  = False
                else:
                    prev_v  = self.prev_v_boundary[loading_dist_idx]
                    m       = self.sample_xv(self.dists[loading_dist_idx], prev_v, t,corrector=corrector,test=test)

            elif idx == len(_ts)-1:
                pass
            else:
                if t in self.dist_ts:
                    dist_idx    = util.get_idx_npy(self.dist_ts,t)
                    if resample and dist_idx!=opt.LOO: #make sure the leave out marginal wont be used during training
                        m = self.resample_m(m,t,idx,corrector)
                        
            f   = self.f(m,t,direction)
            if ode_drift is not None:
                dw  = torch.zeros_like(self.dw(m))
                z   = ode_sign*ode_drift(m,t) 
            else:
                dw  = self.dw(m)
                z   = policy(m,t)

            _,v_dw  = util.split_joint(self.opt, dw)

            if save_traj:
                ms[:,t_idx,...] = m
                zs[:,t_idx,...] =  self.g(t)*(-v_dw-self.dt*z)
                train_ts[t_idx] = t

            m = self.propagate(t, m, z, direction, f=f, dw=dw)

            if corrector is not None and opt.num_corrector_mid!=0 and test:
                _xs, _  = util.split_joint(opt,m)
                vs      = self.corrector_langevin_update(t, m, corrector, denoise_xT=False, num_corrector=opt.num_corrector_mid)
                m       =   torch.cat([_xs,vs],dim=-1)

            if save_traj:
                z_aux               = policy(m,t)
                label[:,t_idx,...]  = self.g(t)*(-v_dw-self.dt*z_aux)

        m_term          = m
        x_term,v_term   = util.split_joint(opt, m_term)
        endtime=time.time()

        if resample:
            self.prev_v_boundary[saving_dist_idx]=v_term
            print('saving previous velocity for {} th distribution'.format(saving_dist_idx))

        elif not test:
            # After optmize Kbridge, the boundary will be used for the next BI Kboundry. eq.10b in the paper.
            oppo_dir = {'forward':'backward',
                        'backward':'forward'}.get(direction)
            self.bdy_xv[oppo_dir]   = m_term
            self.next_reuse_bdy_xv  = True 

        res = [ms, zs, m_term, label, train_ts]
        return res

    def corrector_langevin_update(self, t, m, corrector, denoise_xT, num_corrector):
        opt     = self.opt
        batch   = m.shape[0]
        alpha_t = 1.
        g_t     = self.g(t)
        x,v     = util.split_joint(self.opt,m)
        for _ in range(num_corrector):
            # here, z = g * score
            z =  corrector(torch.cat([x,v],dim=-1),t)
            if z.sum()==0:
                # noise=torch.randn_like(z)
                # v = v +  g_t*np.sqrt(2)*noise, special case when initialize z = 0
                return v
            else:
                # score-based model : eps_{SGM} = 2 * alpha * (snr * \norm{noise/score} )^2
                # schrodinger bridge: eps_{SB}  = 2 * alpha * (snr * \norm{noise/z} )^2
                #                               = g^{-2} * eps_{SGM}
                z_avg_norm      = z.reshape(batch,-1).norm(dim=1).mean()
                eps_temp        = 2 * alpha_t * (opt.snr / z_avg_norm )**2
                noise           = torch.randn_like(z)
                noise_avg_norm  = noise.reshape(batch,-1).norm(dim=1).mean()
                eps             = eps_temp * (noise_avg_norm**2)

                # score-based model:  x <- x + eps_SGM * score + sqrt{2 * eps_SGM} * noise
                # schrodinger bridge: x <- x + g * eps_SB * z  + sqrt(2 * eps_SB) * g * noise
                #                     (so that drift and diffusion are of the same scale) 
                v = v + g_t*eps*z + g_t*torch.sqrt(2*eps)*noise

        if denoise_xT: v = v + g_t*z
        return v


class SimpleSDE(BaseSDE):
    def __init__(self, opt, x_dists, v_dists,var=3.0):
        super(SimpleSDE, self).__init__(opt, x_dists, v_dists)
        self.opt = opt
        self.var = opt.var
    def _f(self, m, t):
        x,v = util.split_joint(self.opt, m)
        return torch.cat([v,torch.zeros_like(v)],dim=-1)

    def _g(self, t):
        return torch.Tensor([self.var])

