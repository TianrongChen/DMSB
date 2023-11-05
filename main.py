from __future__ import absolute_import, division, print_function, unicode_literals
import colored_traceback.always
import sys
from ipdb import set_trace as debug
import pathlib
import logging
import torch
from runner import Runner
import util
import options
from git_utils import *


print(util.yellow("======================================================="))
print(util.yellow("Multi Marginal Momentum Likelihood-Training of Schrodinger Bridge"))
print(util.yellow("======================================================="))
print(util.magenta("setting configurations..."))
opt = options.set()
def main(opt):
    run_dir = pathlib.Path("results") / opt.dir
    setup_logger(run_dir)
    log_git_info(run_dir)
    log = logging.getLogger(__name__)  
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    
    run = Runner(opt)
    # ====== Training functions ======
    if opt.mode=='train':
        run.sb_alternate_train(opt)
    elif opt.mode=='eval':
        #Test this function
        assert opt.load is not None
        run.evaluate(opt, 0, rollout = [0,opt.num_dist-1], resample=False,ode_samp=False)
    else:
        raise RuntimeError()

if not opt.cpu:
    with torch.cuda.device(opt.gpu):
        main(opt)
else: main(opt)
