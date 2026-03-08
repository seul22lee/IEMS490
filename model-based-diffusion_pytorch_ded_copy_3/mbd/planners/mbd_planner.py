import torch
import numpy as np
from tqdm import tqdm
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    env_name: str = "tide_env"
    seed: int = 42
    Nsample: int = 1024         # 사용자 요청: 1024
    Hsample: int = 50           # TiDE p50 사양 대응
    Ndiffuse: int = 20         # 사용자 요청: 50
    temp_sample: float = 0.1    # 논문의 lambda (온도)
    beta0: float = 0.0005
    betaT: float = 0.01
    w_tracking: float = 10.0     # Tracking 가중치
    w_smooth: float = 20       # Smoothness 가중치
    w_constraint: float = 1500   # Constraint 가중치 (p_g 조절)
    device: str = "cuda"
    sigma_min: float = 0.02
    noise_scale: float = 0.3
    w_u0: float = 0.1  # 초기 행동 Y0에 대한 가중치 (논문에서 lambda_u0)

def reverse_once(i, Yi, state_init, planning_data, env, args, alphas, alphas_bar, sigmas):
    device = Yi.device
    alpha = alphas[i]
    alpha_bar = alphas_bar[i]

    # ⭐ sigma floor 적용
    sigma = torch.clamp(sigmas[i], min=args.sigma_min)

    # 1. Sampling
    eps_u = torch.randn((args.Nsample, args.Hsample, env.action_size), device=device)
    Y0s = Yi + eps_u * sigma * args.noise_scale
    Y0s = torch.clamp(Y0s, -1.0, 1.0)

    # 2. Reward
    with torch.no_grad():
        tide_output = env.predict_one_shot(
            state_init,
            Y0s,
            planning_data["fix_cov_future"]
        )
        logp0 = env.get_reward_mbd(
            tide_output,
            planning_data['ref'],
            Y0s,
            planning_data['con'],
            planning_data,
            args
        )

    # 3. Weighted mean
    weights = torch.softmax(logp0, dim=0).float()
    Ybar = torch.einsum("n,nij->ij", weights, Y0s)

    # # 4. Score
    # score = (torch.sqrt(alpha_bar) / (1 - alpha_bar)) * (Ybar - Yi)

    # # 5. Update
    # Y_im1 = Yi + sigma * score

    # print("logp0 mean/std:", logp0.mean().item(), logp0.std().item())

    # weights = torch.softmax(logp0, dim=0)
    # print("weights max/min:", weights.max().item(), weights.min().item())

    # return torch.clamp(Y_im1, -1.0, 1.0)

    curr_Yi = Yi * torch.sqrt(alpha_bar)

    score = 1.0 / (1.0 - alpha_bar) * (
        -curr_Yi + torch.sqrt(alpha_bar) * Ybar
    )

    Yim1 = 1.0 / torch.sqrt(alpha) * (
        curr_Yi + (1.0 - alpha_bar) * score
    )

    alpha_bar_prev = alphas_bar[i-1] if i > 0 else torch.tensor(1.0, device=Yi.device)

    Yi_next = Yim1 / torch.sqrt(alpha_bar_prev)

    return torch.clamp(Yi_next, -1.0, 1.0)


def run_diffusion_main(args: Args, env, state_history, planning_data):
    device = torch.device(args.device)
    
    # Scheduling 준비
    betas = torch.linspace(args.beta0, args.betaT, args.Ndiffuse, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    sigmas = torch.sqrt(1.0 - alphas_bar) # 추가

    # 초기 Noisy Action
    Yi = torch.randn([args.Hsample, env.action_size], device=device).float()
    
    # Backward Process
    for i in range(args.Ndiffuse - 1, -1, -1):
        # 리턴 값 개수 맞춰서 호출 (Yi, _ 제거)
        Yi = reverse_once(i, Yi, state_history, planning_data, env, args, alphas, alphas_bar, sigmas)
    
    return Yi[0] # 첫 번째 Action 반환
if __name__ == "__main__":
    # 별도 테스트 시 tyro cli 사용
    args = tyro.cli(Args)
    print(f"MBD Planner initialized with Nsample={args.Nsample}, Ndiffuse={args.Ndiffuse}")