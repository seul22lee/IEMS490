import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt

import mbd

import torch
import numpy as np


def car_dynamics(x, u):
    # x = x.at[3].set(jnp.clip(x[3], -2.0, 2.0))
    return jnp.array(
        [
            u[1] * jnp.sin(x[2])*3.0,  # x_dot
            u[1] * jnp.cos(x[2])*3.0,  # y_dot
            u[0] * jnp.pi / 3 * 2.0,  # theta_dot
            # u[1] * 6.0,  # v_dot
        ]
    )


def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def check_collision(x, obs_center, obs_radius):
    dist2objs = jnp.linalg.norm(x[:2] - obs_center, axis=1)
    return jnp.any(dist2objs < obs_radius)


@struct.dataclass
class State:
    pipeline_state: jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


class Car2d:
    def __init__(self):
        self.dt = 0.1
        self.H = 50
        r_obs = 0.3
        self.obs_center = jnp.array(
            [
                [-r_obs * 3, r_obs * 2],
                [-r_obs * 2, r_obs * 2],
                [-r_obs * 1, r_obs * 2],
                [0.0, r_obs * 2],
                [0.0, r_obs * 1],
                [0.0, 0.0],
                [0.0, -r_obs * 1],
                [-r_obs * 3, -r_obs * 2],
                [-r_obs * 2, -r_obs * 2],
                [-r_obs * 1, -r_obs * 2],
                [0.0, -r_obs * 2],
            ]
        )
        self.obs_radius = r_obs  # Radius of the obstacle
        self.x0 = jnp.array([-0.5, 0.0, jnp.pi*3/2])
        self.xg = jnp.array([0.5, 0.0, 0.0])
        self.xref = jnp.load(f"{mbd.__path__[0]}/assets/car2d_xref.npy")
        # self.xref = jnp.load(f"{mbd.__path__[0]}/../figure/car2d_xref.npy")
        xref_diff = jnp.diff(self.xref, axis=0)
        theta = jnp.arctan2(xref_diff[:, 0], xref_diff[:, 1])
        self.thetaref = jnp.append(theta, theta[-1])
        self.rew_xref = jax.vmap(self.get_reward)(self.xref).mean()

    def reset(self, rng: jax.Array):
        """Resets the environment to an initial state."""
        return State(self.x0, self.x0, 0.0, 0.0)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        action = jnp.clip(action, -1.0, 1.0)
        q = state.pipeline_state
        q_new = rk4(car_dynamics, state.pipeline_state, action, self.dt)
        collide = check_collision(q_new, self.obs_center, self.obs_radius)
        q = jnp.where(collide, q, q_new)
        reward = self.get_reward(q)
        return state.replace(pipeline_state=q, obs=q, reward=reward, done=0.0)

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, q):
        reward = (
            1.0 - (jnp.clip(jnp.linalg.norm(q[:2] - self.xg[:2]), 0.0, 0.2) / 0.2) ** 2
        )
        return reward

    @partial(jax.jit, static_argnums=(0,))
    def eval_xref_logpd(self, xs):
        xs_err = xs[:, :2] - self.xref[:, :2]
        # theta_err = xs[:, 3] - self.thetaref
        logpd = 0.0-(
            (jnp.clip(jnp.linalg.norm(xs_err, axis=-1), 0.0, 0.5) / 0.5) ** 2
        ).mean(axis=-1)
        return logpd

    @property
    def action_size(self):
        return 2

    @property
    def observation_size(self):
        return 3

    def render(self, ax, xs: jnp.ndarray):
        # obstacles
        for i in range(self.obs_center.shape[0]):
            circle = plt.Circle(
                self.obs_center[i, :], self.obs_radius, color="k", fill=True, alpha=0.5
            )
            ax.add_artist(circle)
        # ax.quiver(
        #     xs[:, 0],
        #     xs[:, 1],
        #     jnp.sin(xs[:, 2]),
        #     jnp.cos(xs[:, 2]),
        #     range(self.H + 1),
        #     cmap="Reds",
        # )
        ax.scatter(xs[:, 0], xs[:, 1], c=range(self.H + 1), cmap="Reds")
        ax.plot(xs[:, 0], xs[:, 1], "r-", label="Car path")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(True)
        # ax.set_title("Car 2D")


class Car2dPyTorch:
    def __init__(self, device="cuda"):
        self._env = Car2d()  # 원본 JAX 환경 객체 생성
        self.device = device
        self.observation_size = self._env.observation_size
        self.action_size = self._env.action_size
        self.obs_center = self._env.obs_center
        self.obs_radius = self._env.obs_radius
        self.xref = self._env.xref
        self.rew_xref = self._env.rew_xref

    def reset(self, seed=None):
        # Car2d.reset은 seed(rng)를 인자로 받으므로 JAX용 키 생성
        import jax
        rng = jax.random.PRNGKey(0)
        state_jax = self._env.reset(rng)
        
        # JAX State를 PyTorch 친화적인 형태로 변환
        return state_jax.replace(
            obs=torch.from_numpy(np.array(state_jax.obs)).to(self.device).float(),
            reward=torch.tensor(float(state_jax.reward), device=self.device).float(),
            pipeline_state=state_jax.pipeline_state # 내부 시뮬레이션 상태는 JAX 배열 유지
        )

    def step(self, state, action: torch.Tensor):
        # 1. Action 변환: PyTorch -> JAX
        jax_action = jax.device_put(jnp.array(action.detach().cpu().numpy()))
        
        # 2. Step 실행 (JAX 엔진 사용)
        new_state_jax = self._env.step(state, jax_action)

        # 3. 결과 변환: JAX -> PyTorch
        return new_state_jax.replace(
            obs=torch.from_numpy(np.array(new_state_jax.obs)).to(self.device).float(),
            reward=torch.tensor(float(new_state_jax.reward), device=self.device).float(),
            done=torch.tensor(0.0, device=self.device).float()
        )

    def eval_xref_logpd(self, xs: torch.Tensor):
        # 가이드를 위한 궤적 평가 함수 래핑
        jax_xs = jax.device_put(jnp.array(xs.detach().cpu().numpy()))
        res = self._env.eval_xref_logpd(jax_xs)
        return torch.from_numpy(np.array(res)).to(self.device).float()

    def render(self, ax, xs):
        # 시각화 시에는 다시 JAX/Numpy 형태로 변환해서 전달
        if isinstance(xs, torch.Tensor):
            xs = xs.detach().cpu().numpy()
        return self._env.render(ax, xs)