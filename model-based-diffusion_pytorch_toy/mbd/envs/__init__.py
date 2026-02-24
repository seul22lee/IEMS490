import torch
import jax
import jax.numpy as jnp
import numpy as np
import os
from brax import envs as brax_envs

# 개별 환경 파일 임포트
from .pushT import PushT
from .hopper import Hopper
from .car2d import Car2d
from .tide_env import TiDEDynamicsEnv  # TiDE 환경 클래스

class BraxPyTorchWrapper:
    def __init__(self, env_instance, device="cuda"):
        self._env = env_instance
        self.device = device
        self.observation_size = getattr(env_instance, "observation_size", None)
        self.action_size = getattr(env_instance, "action_size", None)
        
        # 환경 속성(rew_xref, obs_center 등)을 PyTorch 텐서로 미리 변환하여 저장
        for attr in ["obs_center", "obs_radius", "xref", "rew_xref", "dt"]:
            if hasattr(env_instance, attr):
                val = getattr(env_instance, attr)
                if isinstance(val, (jax.Array, jnp.ndarray, np.ndarray)):
                    setattr(self, attr, torch.from_numpy(np.array(val)).to(self.device).float())
                else:
                    setattr(self, attr, val)

    def _ensure_jax(self, x):
        if isinstance(x, torch.Tensor):
            return jnp.array(x.detach().cpu().numpy())
        return jnp.array(x)

    def _ensure_torch(self, x):
        if isinstance(x, (jax.Array, jnp.ndarray, np.ndarray)):
            return torch.from_numpy(np.array(x)).to(self.device).float()
        return x

    def reset(self, seed):
        seed_val = seed.item() if isinstance(seed, torch.Tensor) else seed
        rng = jax.random.PRNGKey(seed_val)
        state = self._env.reset(rng)
        return jax.tree_util.tree_map(self._ensure_torch, state)

    def step(self, state, action: torch.Tensor):
        jax_action = self._ensure_jax(action)
        jax_state = jax.tree_util.tree_map(self._ensure_jax, state)
        
        if jax_action.ndim > 1:
            if jax_state.obs.ndim < jax_action.ndim:
                vmap_step = jax.vmap(self._env.step, in_axes=(None, 0))
            else:
                vmap_step = jax.vmap(self._env.step, in_axes=(0, 0))
            new_state = vmap_step(jax_state, jax_action)
        else:
            new_state = self._env.step(jax_state, jax_action)
        
        return jax_tree_util.tree_map(self._ensure_torch, new_state)

    def eval_xref_logpd(self, xs: torch.Tensor):
        if hasattr(self._env, "eval_xref_logpd"):
            jax_xs = self._ensure_jax(xs)
            v_eval = jax.vmap(self._env.eval_xref_logpd)
            res = v_eval(jax_xs)
            return self._ensure_torch(res)
        return torch.zeros(xs.shape[0], device=self.device)

    def render(self, ax, xs):
        if hasattr(self._env, "render"):
            render_xs = np.array(xs.detach().cpu()) if isinstance(xs, torch.Tensor) else np.array(xs)
            return self._env.render(ax, render_xs)

def get_env(env_name: str, device: str = "cuda"):
    """환경 이름에 따라 적절한 래퍼 또는 커스텀 환경을 반환합니다."""
    
    # 1. TiDE 커스텀 환경 처리 (학습된 가중치 로드 포함)
    if env_name == "tide_env":
        # 세울님의 모델 파라미터 (학습 시와 동일하게 유지)
        model_params = {
            "input_dim": 3, "output_dim": 2, "future_cov_dim": 1,
            "static_cov_dim": 0, "input_chunck_length": 10, "output_chunk_length": 10,
            "nr_params": 2, "num_encoder_layers": 2, "num_decoder_layers": 2,
            "decoder_output_dim": 4, "hidden_size": 256, "temporal_decoder_hidden": 64,
            "temporal_width_past": 4, "temporal_width_future": 4, "use_layer_norm": True, "dropout": 0.1
        }
        
        # 지정하신 가중치 파일 위치
        weight_path = "/home/ftk3187/github/IEMS490/model-based-diffusion_pytorch_tide/mbd/envs/nominal_params_w10_mid_noise_stable_final.pkl"
        
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
            
        return TiDEDynamicsEnv(model_params, weight_path, device=device)

    # 2. 기존 JAX/Brax 기반 환경 처리
    if env_name == "pushT": 
        env_inst = PushT()
    elif env_name == "hopper": 
        env_inst = Hopper()
    elif env_name == "car2d": 
        env_inst = Car2d()
    elif env_name in ["ant", "halfcheetah", "humanoidrun", "walker2d"]:
        env_inst = brax_envs.get_environment(env_name=env_name, backend="positional")
    else:
        env_inst = brax_envs.get_environment(env_name=env_name, backend="positional")

    return BraxPyTorchWrapper(env_inst, device=device)