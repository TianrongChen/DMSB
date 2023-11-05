import ml_collections

# main.py --problem-name RNAsc --log-tb --ckpt-freq 5  --dir RNA                                                                

def get_RNA_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed         = 42
  config.T            = 4.0
  config.interval     = 400
  config.t0           = 0
  config.problem_name = 'gmm'
  config.num_itr      = 1000
  config.num_epoch    = 1
  config.num_stage    = 15
  config.forward_net  = 'toy'
  config.backward_net = 'toy'
  config.use_arange_t = True
  config.train_bs_x   = 256
  config.v_sampling   ='langevin'
  config.use_corrector= True
  config.snr          =0.15
  config.num_corrector_bdy = 1 #Config this, can be 1,3,5 in order to be aligned with paper
  config.use_amp      = True
  config.var          = 0.4
  config.v_scale      = 0.01
  config.reg          = 0.5
  config.RNA_dim      = 100
  # sampling
  config.samp_bs      = 4000
  config.sde_type     = 'simple'
  config.ckpt_freq    = 5
  # optimization
  config.weight_decay = 0
  config.optimizer    = 'AdamW'
  config.lr           = 2e-4
  config.lr_gamma   = 0.999

  model_configs=None
  return config, model_configs

