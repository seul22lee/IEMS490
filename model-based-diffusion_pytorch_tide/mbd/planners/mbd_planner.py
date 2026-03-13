import torch
import numpy as np
from tqdm import tqdm
import tyro
from dataclasses import dataclass
import mbd.envs

@dataclass
class Args:
    env_name: str = "tide_env"
    seed: int = 42
    Nsample: int = 4096
    Hsample: int = 10
    Ndiffuse: int = 50
    temp_sample: float = 0.1
    beta0: float = 0.0001
    betaT: float = 0.02
    device: str = "cuda"

def reverse_once(i, seed, Yi, state_init, planning_data, env, args, alphas, alphas_bar, sigmas):
    """
    외부(MPC 루프 등)에서 호출 가능하도록 밖으로 꺼낸 Diffusion 1-step 최적화 함수
    """
    device = Yi.device
    Nu = env.action_size
    
    # 1. Diffusion Step Scheduling 변수 설정
    alpha = alphas[i]
    alpha_bar = alphas_bar[i]
    sigma = sigmas[i]
    
    # 2. Gaussian Sampling & Noise Injection
    # Yi는 현재 step의 noisy action sequence
    eps_u = torch.randn((args.Nsample, args.Hsample, Nu), device=device)
    Y0s = eps_u * sigma + Yi
    Y0s = torch.clamp(Y0s, -1.0, 1.0) # Action 범위 제한

    # 3. Reward Evaluation (TiDE 전용 one-shot 함수 사용)
    # env는 TiDEDynamicsEnv 인스턴스여야 함
    with torch.no_grad():
        qs = env.predict_one_shot(state_init, Y0s) 
        rews = env.get_reward_one_shot(qs, Y0s, planning_data)

    # 4. Weighting & Softmax (Exponential Tilting)
    rew_std = rews.std()
    rew_std = torch.where(rew_std < 1e-4, torch.tensor(1.0, device=device), rew_std)
    logp0 = (rews - rews.mean()) / rew_std / args.temp_sample
    weights = torch.softmax(logp0, dim=0).float()
    
    # 5. Weighted Average (Denoiser output approximation)
    Ybar = torch.einsum("n,nij->ij", weights, Y0s)

    # 6. Reverse Diffusion Step (Score-based update)
    curr_Yi = Yi * torch.sqrt(alpha_bar)
    score = 1 / (1.0 - alpha_bar) * (-curr_Yi + torch.sqrt(alpha_bar) * Ybar)
    Yim1 = 1 / torch.sqrt(alpha) * (curr_Yi + (1.0 - alpha_bar) * score)
    
    # 다음 루프를 위해 alpha_bar_prev로 정규화 해제
    alpha_bar_prev = alphas_bar[i-1] if i > 0 else torch.tensor(1.0, device=device)
    Yi_next = Yim1 / torch.sqrt(alpha_bar_prev)
    
    return Yi_next, rews.mean()

def run_diffusion(args: Args):
    """단일 타임스텝에 대한 테스팅용 메인 함수"""
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    env = mbd.envs.get_env(args.env_name, device=args.device)
    Nu = env.action_size
    
    # Scheduling 준비
    betas = torch.linspace(args.beta0, args.betaT, args.Ndiffuse, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    sigmas = torch.sqrt(1 - alphas_bar)

    # 초기 Noisy Action (YN)
    Yi = torch.zeros([args.Hsample, Nu], device=device).float()
    
    # dummy data (실제 실행시에는 planning_data가 필요)
    planning_data = {} 
    state_init = torch.zeros((1, 10, env.observation_size + Nu), device=device)

    pbar = tqdm(range(args.Ndiffuse - 1, -1, -1), desc="Diffusing")
    for i in pbar:
        Yi, rew = reverse_once(i, args.seed, Yi, state_init, planning_data, env, args, alphas, alphas_bar, sigmas)
        pbar.set_postfix(rew=f"{rew.item():.2e}")

    return Yi

if __name__ == "__main__":
    run_diffusion(tyro.cli(Args))