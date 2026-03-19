'''
python3 robomimic/scripts/train.py --config robomimic/exps/templates/cdp.json --dataset ../robosuite/custom_dataset/low_dim_boxpush_120.hdf5
'''

import math
import copy
import logging

import torch
import torch.nn as nn
import numpy as np

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.models.cdp_nets import DiffusionTransformer, KDenoiser 

logger = logging.getLogger(__name__)

# K-DIFFUSION SAMPLING & MATH HELPERS
def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

# Constructs an exponential noise schedule between sigma_max and sigma_min
def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)

# Draws samples from a log-logistic distribution for continuous-time training.
def rand_log_logistic(shape, loc=0., scale=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
    return u.logit().mul(scale).add(loc).exp().to(dtype)

# Converts a denoiser output to a Karras ODE derivative.
def to_d(action, sigma, denoised):
    dims_to_append = action.ndim - sigma.ndim
    sigma_appended = sigma[(...,) + (None,) * dims_to_append]
    return (action - denoised) / sigma_appended

# DDIM sampler
@torch.no_grad()
def sample_ddim(model, state, action, goal, sigmas, extra_args=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for i in range(len(sigmas) - 1):
        denoised = model(state, action, goal, sigmas[i] * s_in, **extra_args)
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised
    return action

# Wraps the Denoiser during inference to apply Classifier-Free Guidance
class CFGWrapper(nn.Module):
    def __init__(self, model, cond_lambda):
        super().__init__()
        self.model = model
        self.cond_lambda = cond_lambda
        
    def forward(self, state, action, goal, sigma):
        out_cond = self.model(state, action, goal, sigma, uncond=False)
        out_uncond = self.model(state, action, goal, sigma, uncond=True)
        return out_uncond + self.cond_lambda * (out_cond - out_uncond)


# Registers cdp as a valid algorithm name in Robomimic
@register_algo_factory_func("cdp")
def algo_config_to_class(algo_config):
    return CDP, {}

class CDP(PolicyAlgo):
    # Initializes the Score Networks, Samplers, and EMA wrappers
    def _create_networks(self):
        goal_keys = list(self.algo_config.cfg.goal_obs_keys) if self.algo_config.cfg.enabled else []
        goal_key_set = set(goal_keys)

        obs_keys = [k for k in self.obs_config.modalities.obs.low_dim if k not in goal_key_set]

        state_dim = sum(self.obs_shapes[k][0] for k in obs_keys)
        goal_dim  = sum(self.obs_shapes[k][0] for k in goal_keys) if self.algo_config.cfg.enabled else 0

        # Initialize the core sequence transformer
        inner_model = DiffusionTransformer(
            state_dim=state_dim,
            action_dim=self.ac_dim,
            goal_dim=goal_dim,
            goal_conditioned=self.algo_config.cfg.enabled,
            embed_dim=self.algo_config.model.embed_dim,
            embed_pdrop=0.0,
            attn_pdrop=self.algo_config.model.attn_pdrop,
            resid_pdrop=self.algo_config.model.resid_pdrop,
            n_layers=self.algo_config.model.num_hidden_layers,
            n_heads=self.algo_config.model.n_heads,
            goal_seq_len=1 if self.algo_config.cfg.enabled else 0,
            obs_seq_len=self.algo_config.horizon.observation_horizon,
            goal_drop=self.algo_config.cfg.uncond_prob,
            linear_output=self.algo_config.model.linear_output,
            device=self.device
        )
        
        # Wrap with Karras Preconditioner
        self.nets = nn.ModuleDict()
        self.nets["policy"] = KDenoiser(inner_model, sigma_data=self.algo_config.sampler.sigma_data)
        self.nets = self.nets.to(self.device)

        # Initialize Optimizer
        self.optimizer = torch.optim.Adam(
            self.nets["policy"].parameters(),
            lr=self.algo_config.optim_params.policy.learning_rate.initial,
            weight_decay=self.algo_config.optim_params.policy.learning_rate.kwargs.weight_decay
        )

        # Initialize Exponential Moving Average
        if self.algo_config.ema.enabled:
            self.nets["ema_policy"] = copy.deepcopy(self.nets["policy"])
            for param in self.nets["ema_policy"].parameters():
                param.requires_grad = False

        # Rolling observation buffer used during rollouts
        self._obs_buffer = None

    # Moves batch data to GPU, casts to float32, and builds the goal dict for training.
    def process_batch_for_training(self, batch):
        batch = TensorUtils.to_device(batch, self.device)
        batch = TensorUtils.to_float(batch)

        # Extract goal condition keys from obs
        if self.algo_config.cfg.enabled:
            goal_keys = list(self.algo_config.cfg.goal_obs_keys)
            if goal_keys:
                batch["goal"] = {k: batch["obs"][k] for k in goal_keys if k in batch["obs"]}
        return batch

    # Overrides the base postprocessing to exclude goal condition keys from obs normalization.
    def postprocess_batch_for_training(self, batch, obs_normalization_stats):
        if obs_normalization_stats is not None:
            goal_key_set = set(self.algo_config.cfg.goal_obs_keys) if self.algo_config.cfg.enabled else set()
            # Temporarily stash goal keys out of batch["obs"] so normalize_dict doesn't see them
            stashed = {}
            for k in goal_key_set:
                if k in batch["obs"]:
                    stashed[k] = batch["obs"].pop(k)
            filtered_stats = {k: v for k, v in obs_normalization_stats.items() if k not in goal_key_set}
            batch = super().postprocess_batch_for_training(batch, obs_normalization_stats=filtered_stats)
            batch["obs"].update(stashed)
        else:
            batch = super().postprocess_batch_for_training(batch, obs_normalization_stats=None)
        return batch

    def _format_obs(self, obs_dict):
        goal_key_set = set(self.algo_config.cfg.goal_obs_keys) if self.algo_config.cfg.enabled else set()
        # Exclude goal keys so push_distance is not part of the state vector
        obs_list = [obs_dict[k] for k in self.obs_config.modalities.obs.low_dim if k not in goal_key_set]
        x = torch.cat(obs_list, dim=-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x
    
    def _format_goal(self, goal_dict):
        if goal_dict is None or not self.algo_config.cfg.enabled:
            return None
        goal_keys = list(self.algo_config.cfg.goal_obs_keys)
        goal_list = [goal_dict[k] for k in goal_keys if k in goal_dict]
        if len(goal_list) == 0:
            return None

        g = torch.cat(goal_list, dim=-1)
        if g.dim() == 2:
            g = g.unsqueeze(1)
        # Enforce sequence length of 1 for the condition token
        return g[:, 0:1, :]

    # Core continuous-time Score Matching loop.
    def train_on_batch(self, batch, epoch, validate=False):      
        states = self._format_obs(batch["obs"])
        actions = batch["actions"]
        goals = self._format_goal(batch.get("goal"))
        
        B = actions.shape[0]

        # Sample sigmas from the Log-Logistic distribution
        sigmas = rand_log_logistic(
            shape=(B,),
            loc=self.algo_config.sampler.sigma_sample_density_mean,
            scale=self.algo_config.sampler.sigma_sample_density_std,
            min_value=self.algo_config.sampler.sigma_min,
            max_value=self.algo_config.sampler.sigma_max,
            device=self.device
        )
        
        # Sample target noise
        noise = torch.randn_like(actions)

        # Compute Denoising Loss
        loss = self.nets["policy"].loss(
            state=states, 
            action=actions, 
            goal=goals, 
            noise=noise, 
            sigma=sigmas
        )

        # Optimization Step
        if not validate:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # EMA Update
            if self.algo_config.ema.enabled:
                decay = self.algo_config.ema.power
                with torch.no_grad():
                    for param, ema_param in zip(self.nets["policy"].parameters(), self.nets["ema_policy"].parameters()):
                        ema_param.data.copy_(decay * ema_param.data + (1 - decay) * param.data)

        return {"Loss": loss.item()}

    def reset(self):
        self._obs_buffer = None

    # Inference loop using Classifier-Free Guidance and DDIM sampling
    def get_action(self, obs_dict, goal_dict=None):
        T_obs = self.algo_config.horizon.observation_horizon

        # Build rolling observation buffer
        single_state = self._format_obs(obs_dict)  # (B, 1, D)
        if self._obs_buffer is None:
            self._obs_buffer = single_state.repeat(1, T_obs, 1)  # (B, T_obs, D)
        else:
            # Slide the window: drop oldest, append newest
            self._obs_buffer = torch.cat([self._obs_buffer[:, 1:, :], single_state], dim=1)
        states = self._obs_buffer  # (B, T_obs, D)

        # During rollout goal_dict is None; extract the condition from obs_dict instead
        if goal_dict is None and self.algo_config.cfg.enabled:
            goal_keys = list(self.algo_config.cfg.goal_obs_keys)
            goal_dict = {k: obs_dict[k] for k in goal_keys if k in obs_dict}
        goals = self._format_goal(goal_dict)

        B = states.shape[0]
        T_pred = self.algo_config.horizon.prediction_horizon

        # Use the smooth EMA weights for rollouts if enabled
        model_to_use = self.nets["ema_policy"] if self.algo_config.ema.enabled else self.nets["policy"]

        # 1. Wrap model in CFG extrapolation logic
        cfg_model = CFGWrapper(model_to_use, self.algo_config.cfg.cond_lambda)

        # 2. Initialize pure noise scaled by max sigma
        action_noise = torch.randn(B, T_pred, self.ac_dim, device=self.device) * self.algo_config.sampler.sigma_max

        # 3. Retrieve discrete inference timesteps
        sigmas = get_sigmas_exponential(
            self.algo_config.sampler.num_inference_steps,
            self.algo_config.sampler.sigma_min,
            self.algo_config.sampler.sigma_max,
            device=self.device
        )

        # 4. Denoise using DDIM
        action_pred = sample_ddim(cfg_model, states, action_noise, goals, sigmas)

        # 5. Extract only the current action (last predicted step)
        current_action = action_pred[:, -1, :]

        return current_action.cpu().numpy()

    def log_info(self, info):
        return info