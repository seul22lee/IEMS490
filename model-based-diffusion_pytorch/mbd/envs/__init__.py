import torch
import jax
import jax.numpy as jnp
import numpy as np
from brax import envs as brax_envs

# 개별 환경 파일 임포트
from .pushT import PushT
from .hopper import Hopper
from .humanoidstandup import HumanoidStandup
from .humanoidtrack import HumanoidTrack
from .humanoidrun import HumanoidRun
from .walker2d import Walker2d
from .cartpole import Cartpole
from .car2d import Car2d

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
                # JAX/Numpy 배열인 경우에만 변환
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
        # 전체 State 구조체를 Torch로 변환
        return jax.tree_util.tree_map(self._ensure_torch, state)

    def step(self, state, action: torch.Tensor):
        # 1. 입력을 JAX로 변환
        jax_action = self._ensure_jax(action)
        jax_state = jax.tree_util.tree_map(self._ensure_jax, state)
        
        # 2. 배치 처리 여부 동적 결정
        if jax_action.ndim > 1:
            # 초기 상태(단일)와 액션(배치)이 섞인 경우 처리
            if jax_state.obs.ndim < jax_action.ndim:
                vmap_step = jax.vmap(self._env.step, in_axes=(None, 0))
            else:
                vmap_step = jax.vmap(self._env.step, in_axes=(0, 0))
            new_state = vmap_step(jax_state, jax_action)
        else:
            new_state = self._env.step(jax_state, jax_action)
        
        # 3. 결과를 다시 PyTorch로 변환
        return jax.tree_util.tree_map(self._ensure_torch, new_state)

    def eval_xref_logpd(self, xs: torch.Tensor):
        if hasattr(self._env, "eval_xref_logpd"):
            jax_xs = self._ensure_jax(xs)
            # 궤적 평가 시 vmap 적용
            v_eval = jax.vmap(self._env.eval_xref_logpd)
            res = v_eval(jax_xs)
            return self._ensure_torch(res)
        return torch.zeros(xs.shape[0], device=self.device)

    def render(self, ax, xs):
        if hasattr(self._env, "render"):
            render_xs = np.array(xs.detach().cpu()) if isinstance(xs, torch.Tensor) else np.array(xs)
            return self._env.render(ax, render_xs)

def get_env(env_name: str, device: str = "cuda"):
    if env_name == "pushT": env_inst = PushT()
    elif env_name == "hopper": env_inst = Hopper()
    elif env_name == "car2d": env_inst = Car2d() #
    elif env_name in ["ant", "halfcheetah", "humanoidrun", "walker2d"]:
        env_inst = brax_envs.get_environment(env_name=env_name, backend="positional")
    else:
        env_inst = brax_envs.get_environment(env_name=env_name, backend="positional")

    return BraxPyTorchWrapper(env_inst, device=device)