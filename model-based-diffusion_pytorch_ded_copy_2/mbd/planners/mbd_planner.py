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
    Ndiffuse: int = 50          # 사용자 요청: 50
    temp_sample: float = 0.1    # 논문의 lambda (온도)
    beta0: float = 0.0001
    betaT: float = 0.02
    w_tracking: float = 1.0     # Tracking 가중치
    w_smooth: float = 1.0       # Smoothness 가중치
    w_constraint: float = 5.0   # Constraint 가중치 (p_g 조절)
    device: str = "cuda"

def reverse_once(i, Yi, state_init, planning_data, env, args, alphas, alphas_bar, sigmas):
    device = Yi.device
    alpha = alphas[i]
    alpha_bar = alphas_bar[i]
    sigma = sigmas[i]

    # 1. 샘플링: 현재 Yi 주변에서 Y0 후보군 생성
    eps_u = torch.randn((args.Nsample, args.Hsample, env.action_size), device=device)
    Y0s = Yi + eps_u * sigma
    Y0s = torch.clamp(Y0s, -1.0, 1.0) # 물리적 범위 제한

    # 2. Reward 평가
    with torch.no_grad():
        tide_output = env.predict_one_shot(state_init, Y0s) 
        logp0 = env.get_reward_mbd(tide_output, planning_data['ref'], Y0s, planning_data['con'], args)

    # 3. Softmax 가중 평균 Ybar 계산
    weights = torch.softmax(logp0, dim=0).float()
    Ybar = torch.einsum("n,nij->ij", weights, Y0s)

    # 4. Score 추정 (논문 Eq. 10c)
    score = (-Yi / (1.0 - alpha_bar)) + (torch.sqrt(alpha_bar) / (1.0 - alpha_bar)) * Ybar

    # 5. MCSA 업데이트 (논문 Eq. 6)
    Y_im1 = (1.0 / torch.sqrt(alpha)) * (Yi + (1.0 - alpha_bar) * score)
    
    # 다음 단계 입력을 위해 다시 한번 clamp
    return torch.clamp(Y_im1, -1.0, 1.0)

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