import ml_collections

def get_gmm_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed         = 42
  config.T            = 4.0
  config.interval     = 400
  config.t0           = 0
  config.problem_name = 'gmm'
  config.num_itr      = 1000
  # config.eval_itr     = 200
  config.forward_net  = 'toy'
  config.backward_net = 'toy'
  config.use_arange_t = True
  config.num_stage    = 8
  config.train_bs_x   = 256
  config.num_epoch    = 1
  config.v_sampling   = 'langevin'
  config.use_corrector= True
  config.num_correcotr_bdy= 3
  config.snr          = 0.1
  config.use_amp      = True
  config.var          = 0.5
  config.v_scale      = 3
  config.reg          = 1.0
  # sampling
  config.samp_bs      = 2000
  config.sde_type     = 'simple'
  config.ckpt_freq    = 5

  
  # optimization
#   config.optim = optim = ml_collections.ConfigDict()
  config.weight_decay = 0
  config.optimizer    = 'AdamW'
  config.lr           = 2e-4
  config.lr_gamma     = 0.999

  model_configs=None
  return config, model_configs

