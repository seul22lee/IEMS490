import torch
import numpy as np
import jax
from jax import numpy as jp
from brax import base
from brax.envs.base import PipelineEnv, State as BraxState
from brax.io import mjcf
from etils import epath

# Note: We keep the core logic in JAX for physics simulation speed, 
# but wrap the inputs/outputs to work seamlessly with PyTorch.

class Hopper:
    def __init__(self, device="cuda"):
        path = epath.resource_path("brax") / "envs/assets/hopper.xml"
        self.sys = mjcf.load(path)
        self._reset_noise_scale = 5e-3
        
        # Brax internal environment
        self._env = PipelineEnv(sys=self.sys, backend="positional", n_frames=20)
        self.observation_size = 11 # Based on position(4) + velocity(7) approx
        self.action_size = self.sys.act_size()
        self.device = device

    def _to_jax(self, x):
        if isinstance(x, torch.Tensor):
            return jax.device_put(jp.array(x.detach().cpu().numpy()))
        return x

    def _to_torch(self, x):
        if isinstance(x, (jax.Array, jp.ndarray)):
            return torch.from_numpy(np.array(x)).to(self.device)
        return x

    def reset(self, seed: torch.Tensor) -> BraxState:
        """Resets the environment to an initial state."""
        # Convert torch seed to JAX PRNGKey
        rng = jax.random.PRNGKey(seed.item() if seed.dim() == 0 else seed[0].item())
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=low, maxval=hi)

        pipeline_state = self._env.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state)
        
        # Wrap the state to contain PyTorch tensors
        state = BraxState(
            pipeline_state=pipeline_state, 
            obs=self._to_torch(obs), 
            reward=torch.tensor(0.0, device=self.device), 
            done=torch.tensor(0.0, device=self.device), 
            metrics={}
        )
        return state

    def step(self, state: BraxState, action: torch.Tensor) -> BraxState:
        """Runs one timestep of the environment's dynamics."""
        # 1. Convert PyTorch action to JAX
        jax_action = self._to_jax(action)
        
        # 2. Step the JAX-based simulation
        pipeline_state = self._env.pipeline_step(state.pipeline_state, jax_action)

        # 3. Get observations and rewards
        obs = self._get_obs(pipeline_state)
        reward = self._get_reward(pipeline_state)

        # 4. Return state with PyTorch tensors
        return state.replace(
            pipeline_state=pipeline_state,
            obs=self._to_torch(obs),
            reward=self._to_torch(reward),
            done=torch.tensor(0.0, device=self.device)
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations (Internal JAX)."""
        position = pipeline_state.q
        # position = position.at[1].set(pipeline_state.x.pos[0, 2])
        # Note: Brax's at[].set() syntax
        new_pos = jp.array(position)
        new_pos = new_pos.at[1].set(pipeline_state.x.pos[0, 2])
        velocity = jp.clip(pipeline_state.qd, -10, 10)

        return jp.concatenate((new_pos, velocity))

    def _get_reward(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment reward (Internal JAX)."""
        return (
            pipeline_state.x.pos[0, 0] + 
            - jp.clip(jp.abs(pipeline_state.x.pos[0, 2] - 1.0), -1.0, 1.0) * 0.5
        )