import ml_collections
# python main.py  --dir semicircle --problem-name semicircle  --log-tb  --ckpt-freq 10 
def get_semicircle_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed         = 42
  config.T            = 3.0
  config.interval     = 300
  config.t0           = 0
  config.problem_name = 'gmm'
  config.num_itr      = 1000
  config.num_epoch    = 1
  config.num_stage    = 10
  # config.eval_itr     = 200
  config.forward_net  = 'toy'
  config.backward_net = 'toy'
  config.use_arange_t = True
  config.train_bs_x   = 256
  config.v_sampling   ='langevin'
  config.use_corrector= True
  config.snr          =0.15
  config.num_corrector_bdy = 3 #Config this, can be 1,3,5 in order to be aligned with paper
  config.use_amp      = True
  config.var          = 0.5
  config.v_scale      = 3
  config.reg          = 0.5
  # sampling
  config.samp_bs      = 2000
  config.sde_type     = 'simple'
  config.ckpt_freq    = 5
  
  # optimization
  config.weight_decay = 0
  config.optimizer    = 'AdamW'
  config.lr           = 1e-4
  config.lr_gamma   = 0.999

  model_configs=None
  return config, model_configs