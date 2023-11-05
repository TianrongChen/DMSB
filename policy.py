
import torch
import sde
import util

from ipdb import set_trace as debug

def build(opt, dyn, direction):
    print(util.magenta("build {} policy...".format(direction)))

    net_name    = getattr(opt, direction+'_net')
    net         = _build_net(opt, net_name)
    use_t_idx   = True
    scale_by_g  = True

    policy = SchrodingerBridgePolicy(
        opt, direction, dyn, net, use_t_idx=use_t_idx, scale_by_g=scale_by_g
    )

    print(util.red('number of parameters is {}'.format(util.count_parameters(policy))))
    policy.to(opt.device)

    return policy

def _build_net(opt, net_name):
    zero_out_last_layer = False    if opt.problem_name=='petal' else True

    if net_name == 'toy':
        assert util.is_toy_dataset(opt)
        from models.toy_model.Toy import build_toy
        net = build_toy(opt.data_dim[0]*2, zero_out_last_layer, device=opt.device, num_ResNet=opt.num_ResNet)
    else:
        raise RuntimeError('Do not have such network, please implement it.')

    return net

class SchrodingerBridgePolicy(torch.nn.Module):
    # note: scale_by_g matters only for pre-trained model
    def __init__(self, opt, direction, dyn, net, use_t_idx=False, scale_by_g=True):
        super(SchrodingerBridgePolicy,self).__init__()
        self.opt = opt
        self.direction = direction
        self.dyn = dyn
        self.net = net
        self.use_t_idx = use_t_idx
        self.scale_by_g = scale_by_g

    @ property
    def zero_out_last_layer(self):
        return self.net.zero_out_last_layer


    def forward(self, x, t):
        # make sure t.shape = [batch]
        t = t.squeeze()
        if t.dim()==0: t = t.repeat(x.shape[0])
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        if self.use_t_idx:
            t = t / self.opt.T * self.opt.interval

        out = self.net(x, t)

        # if the SB policy behaves as "Z" in FBSDE system,
        # the output should be scaled by the diffusion coefficient "g".
        if self.scale_by_g:
            g = self.dyn.g(t)
            out = out /  g

        return out


