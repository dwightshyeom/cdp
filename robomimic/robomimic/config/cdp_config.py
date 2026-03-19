from robomimic.config.base_config import BaseConfig

class CDPConfig(BaseConfig):
    ALGO_NAME = "cdp"

    def train_config(self):
        """
        Setting up training parameters for CDP.
        """
        super(CDPConfig, self).train_config()

        self.train.hdf5_load_next_obs = False
        self.train.seq_length = 5 # Matches window_size for observations
        self.train.frame_stack = 1 
    
    def algo_config(self):
        # Optimization Parameters
        self.algo.optim_params.policy.optimizer_type = "adam"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.99
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [100]
        self.algo.optim_params.policy.learning_rate.kwargs.weight_decay = 0.0
        self.algo.optim_params.policy.regularization.L2 = 0.00          

        # Horizon Parameters (from window_size and future_seq_length)
        self.algo.horizon.observation_horizon = 5
        self.algo.horizon.action_horizon = 1 
        self.algo.horizon.prediction_horizon = 5
        
        # Classifier-Free Guidance (CFG) Parameters
        self.algo.cfg.enabled = True
        self.algo.cfg.uncond_prob = 0.1   # cond_mask_prob
        self.algo.cfg.cond_lambda = 2.0
        self.algo.cfg.goal_obs_keys = ["push_distance"]  # obs keys used as the CFG condition

        # Network Architecture (DiffusionGPT)
        self.algo.model.type = "diffusion_gpt"
        self.algo.model.embed_dim = 240
        self.algo.model.num_hidden_layers = 4
        self.algo.model.n_heads = 12
        self.algo.model.attn_pdrop = 0.05
        self.algo.model.resid_pdrop = 0.05
        self.algo.model.linear_output = True

        # Time & Sigma Embedding
        self.algo.model.embed_type = 'Linear'
        self.algo.model.time_embed_dim = 240

        # Exponential Moving Average (EMA)
        self.algo.ema.enabled = True
        self.algo.ema.update_after_step = 1
        self.algo.ema.inv_gamma = 1.0
        self.algo.ema.power = 0.999 # decay
        self.algo.ema.min_value = 0.0
        self.algo.ema.max_value = 0.9999

        # Sampler & Score-Matching Parameters
        self.algo.sampler.num_inference_steps = 3
        self.algo.sampler.sampler_type = 'ddim'
        self.algo.sampler.sigma_data = 0.5
        self.algo.sampler.rho = 5.0
        self.algo.sampler.sigma_min = 0.05 
        self.algo.sampler.sigma_max = 1.0 
        
        # Log-logistic sample density distribution for continuous-time training
        self.algo.sampler.sigma_sample_density_type = 'loglogistic'
        self.algo.sampler.sigma_sample_density_mean = -0.6
        self.algo.sampler.sigma_sample_density_std = 1.6
        self.algo.sampler.noise_scheduler = 'exponential'