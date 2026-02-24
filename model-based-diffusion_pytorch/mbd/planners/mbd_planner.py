import functools
import os
import torch
import numpy as np
from dataclasses import dataclass
import tyro
from tqdm import tqdm
from matplotlib import pyplot as plt

import mbd

## load config
@dataclass
class Args:
    # exp
    seed: int = 0
    disable_recommended_params: bool = False
    not_render: bool = False
    # env
    env_name: str = (
        "ant"  # "humanoidstandup", "ant", "halfcheetah", "hopper", "walker2d", "car2d"
    )
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 50  # horizon
    Ndiffuse: int = 100  # number of diffusion steps
    temp_sample: float = 0.1  # temperature for sampling
    beta0: float = 1e-4  # initial beta
    betaT: float = 1e-2  # final beta
    enable_demo: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_diffusion(args: Args):

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ## setup env

    # recommended temperature for envs
    temp_recommend = {
        "ant": 0.1,
        "halfcheetah": 0.4,
        "hopper": 0.1,
        "humanoidstandup": 0.1,
        "humanoidrun": 0.1,
        "walker2d": 0.1,
        "pushT": 0.2,
    }
    Ndiffuse_recommend = {
        "pushT": 200,
        "humanoidrun": 300,
    }
    Nsample_recommend = {
        "humanoidrun": 8192,
    }
    Hsample_recommend = {
        "pushT": 40,
    }
    if not args.disable_recommended_params:
        args.temp_sample = temp_recommend.get(args.env_name, args.temp_sample)
        args.Ndiffuse = Ndiffuse_recommend.get(args.env_name, args.Ndiffuse)
        args.Nsample = Nsample_recommend.get(args.env_name, args.Nsample)
        args.Hsample = Hsample_recommend.get(args.env_name, args.Hsample)
        print(f"override temp_sample to {args.temp_sample}")
    
    env = mbd.envs.get_env(args.env_name)
    Nx = env.observation_size
    Nu = env.action_size
    
    # env functions
    # Note: Assuming env.step and env.reset are compatible with PyTorch or wrapped
    step_env = env.step
    reset_env = env.reset
    rollout_us = functools.partial(mbd.utils.rollout_us, step_env)

    state_init = reset_env(torch.tensor(args.seed, device=device))

    ## run diffusion

    betas = torch.linspace(args.beta0, args.betaT, args.Ndiffuse, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    sigmas = torch.sqrt(1 - alphas_bar)
    
    # sigmas_cond calculation
    alphas_bar_prev = torch.cat([torch.tensor([1.0], device=device), alphas_bar[:-1]])
    Sigmas_cond = (1 - alphas) * (1 - torch.sqrt(alphas_bar_prev)) / (1 - alphas_bar)
    sigmas_cond = torch.sqrt(Sigmas_cond)
    sigmas_cond[0] = 0.0
    print(f"init sigma = {sigmas[-1]:.2e}")

    YN = torch.zeros([args.Hsample, Nu], device=device) # seul // warm start using last action

    # seul // gpu parallelization
    def reverse_once(i, rng_seed, Ybar_i, state_init):
        Yi = Ybar_i * torch.sqrt(alphas_bar[i])

        # sample from q_i (gaussian neighnorhood sampling)
        eps_u = torch.randn((args.Nsample, args.Hsample, Nu), device=device) 
        Y0s = eps_u * sigmas[i] + Ybar_i
        Y0s = torch.clamp(Y0s, -1.0, 1.0) # seul // sampled control sequences ; n

        # esitimate mu_0tm1
        # seul // dynamics rollout , rewss: reward trajectory, qs: state trajectory
        rewss, qs = rollout_us(state_init, Y0s) 
        rews = rewss.mean(axis=-1) # seul // reward mean (over trajectory)
        rew_std = rews.std() # seul // reward std (over trajectory)
        rew_std = torch.where(rew_std < 1e-4, torch.tensor(1.0, device=device), rew_std)
        rew_mean = rews.mean()
        
        # seul // args.temp_sample is the temperature for sampling...
        logp0 = (rews - rew_mean) / rew_std / args.temp_sample 

        # evalulate demo
        if args.enable_demo:
            xref_logpds = env.eval_xref_logpd(qs)
            xref_logpds = xref_logpds - xref_logpds.max()
            logpdemo = (
                (xref_logpds + env.rew_xref - rew_mean) / rew_std / args.temp_sample
            )
            demo_mask = logpdemo > logp0
            logp0 = torch.where(demo_mask, logpdemo, logp0)
            logp0 = (logp0 - logp0.mean()) / logp0.std() / args.temp_sample

        weights = torch.softmax(logp0, dim=0)
        Ybar = torch.einsum("n,nij->ij", weights, Y0s)  # NOTE: update only with reward

        # seul // score function ; How to move Yi to Ybar direction
        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + torch.sqrt(alphas_bar[i]) * Ybar) 
        Yim1 = 1 / torch.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)

        alpha_bar_prev_val = alphas_bar[i-1] if i > 0 else torch.tensor(1.0, device=device)
        Ybar_im1 = Yim1 / torch.sqrt(alpha_bar_prev_val)

        return Ybar_im1, rews.mean()

    # run reverse
    def reverse(YN, seed):
        Yi = YN
        Ybars = []
        # seul // diffusion process : iteratively refine the control sequence Yi...
        # seul // reverse_once = gaussian sampling + rollout + reward evaluation + weighted mean + score function evaluation + update Yi
        with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                Yi, rew = reverse_once(i, seed, Yi, state_init)
                Ybars.append(Yi.detach().cpu().numpy())
                # Update the progress bar's suffix to show the current reward
                pbar.set_postfix({"rew": f"{rew.item():.2e}"})
        return np.array(Ybars)

    Yi_history = reverse(YN, args.seed)
    Yi_last = torch.tensor(Yi_history[-1], device=device)

    if not args.not_render:
        path = f"{mbd.__path__[0]}/../results/{args.env_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(f"{path}/mu_0ts.npy", Yi_history)
        
        if args.env_name == "car2d":
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            # rollout
            xs = [state_init.pipeline_state.cpu().numpy()]
            state = state_init
            for t in range(Yi_history.shape[1]):
                state = step_env(state, Yi_last[t])
                xs.append(state.pipeline_state.cpu().numpy())
            xs = np.array(xs)
            env.render(ax, xs)
        # 173번 라인 근처 수정
            if args.enable_demo and hasattr(env, "xref"):
                # env.xref가 텐서인 경우 cpu로 옮기고 numpy로 변환
                xref_plot = env.xref.detach().cpu().numpy() if torch.is_tensor(env.xref) else env.xref
                ax.plot(xref_plot[:, 0], xref_plot[:, 1], "g--", label="RRT path")
                ax.legend()
            ax.legend()
            plt.savefig(f"{path}/rollout.png")
        else:
            render_us = functools.partial(
                mbd.utils.render_us,
                step_env,
                env.sys, # tree_replace logic might need adjustment depending on mbd
            )
            webpage = render_us(state_init, Yi_last)
            with open(f"{path}/rollout.html", "w") as f:
                f.write(webpage)
                
    rewss_final, _ = rollout_us(state_init, Yi_last.unsqueeze(0))
    rew_final = rewss_final.mean().item()

    return rew_final


if __name__ == "__main__":
    rew_final = run_diffusion(args=tyro.cli(Args))
    print(f"final reward = {rew_final:.2e}")