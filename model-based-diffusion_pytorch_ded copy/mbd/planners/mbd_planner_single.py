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
    env_name: str = "tide_env" # "ant", "car2d" 대신 커스텀 환경 명칭 사용
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 10    # TiDE Horizon (10)으로 수정
    Ndiffuse: int = 100  # number of diffusion steps
    temp_sample: float = 0.1  # temperature for sampling
    beta0: float = 1e-4  # initial beta
    betaT: float = 1e-2  # final beta
    enable_demo: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_diffusion(args: Args):

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ## 1. setup env
    env = mbd.envs.get_env(args.env_name)
    Nu = env.action_size
    
    ## 2. Generate Reference Trajectory & Constraints (세울님 로직 통합)
    scaling = 2
    # Step trajectory
    Ref_traj_step = torch.tensor(np.repeat(np.array([0, -scaling, -scaling, 0, scaling, scaling]), 30))
    
    # Sigmoid trajectory
    def sigmoid(x): return 1 / (1 + np.exp(-x / 4))
    x_sig = np.linspace(0, 50, 50)
    y_sig = -sigmoid(x_sig - 25) * scaling + scaling
    Ref_traj_sigmoid = torch.tensor(y_sig)

    # Sine trajectory
    x_sin = np.linspace(0, 2 * np.pi, 100)  
    y_sin = scaling * np.sin(x_sin) 
    Ref_traj_sin = torch.tensor(y_sin)

    # Concatenate all to make full Reference Trajectory
    Ref_traj = torch.cat((Ref_traj_step, Ref_traj_sigmoid, torch.zeros(20), Ref_traj_sin, Ref_traj_step, torch.zeros(20))).float()
    tot_step = len(Ref_traj)

    # Constraints (Upper/Lower)
    x_constraint = np.linspace(0, tot_step, tot_step)
    C_Ref_traj_up = torch.tensor(4 + 0.5 * np.sin(2 * np.pi * x_constraint / 100) - 0.004 * x_constraint).float()
    C_Ref_traj_low = -C_Ref_traj_up

    # 3. Scaling Constants
    x1_diff, x2_diff, u_diff = 9.32425809, 14.13760952, 9.99280029
    window = 10 # TiDE Lookback/Horizon

    # 4. Prepare Sliding Window Data
    # Diffusion 연산을 위해 데이터를 미리 잘라둡니다 (unfold)
    sliding_data_ref_x1 = Ref_traj.unfold(0, window, 1)
    sliding_data_con_x2_low = C_Ref_traj_low.unfold(0, window, 1)
    sliding_data_con_x2_up = C_Ref_traj_up.unfold(0, window, 1)

    # 시뮬레이션용 가상 궤적 데이터 (테스트용으로 0으로 초기화된 tensor 생성)
    # 실제 환경에서는 과거 10스텝의 실제 x, u 값이 들어와야 합니다.
    x_trajectory = torch.zeros((tot_step, 2), device=device)
    u_trajectory = torch.zeros((tot_step, 1), device=device)

    # 5. TiDE용 초기 상태(state_init) 및 플래닝 데이터(planning_data) 준비
    i_start = 100 
    
    # 과거 10스텝 데이터 구성 (1, 10, 3) -> 정규화 적용
    x_window = x_trajectory[i_start : i_start + 10, :] / torch.tensor([x1_diff/2, x2_diff/2], device=device)
    u_window = u_trajectory[i_start : i_start + 10, :] / (u_diff/2)
    state_init = torch.cat([x_window, u_window], dim=-1).unsqueeze(0).to(device)

    # 미래 10스텝 Reference & Constraint (i_start+10 시점부터의 윈도우)
    planning_data = {
        'x1_ref': sliding_data_ref_x1[i_start + 10].to(device) / (x1_diff/2),
        'x2_low': sliding_data_con_x2_low[i_start + 10].to(device) / (x2_diff/2),
        'x2_up': sliding_data_con_x2_up[i_start + 10].to(device) / (x2_diff/2),
    }

    ## 6. run diffusion 스케줄링 (구조 유지)
    betas = torch.linspace(args.beta0, args.betaT, args.Ndiffuse, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    sigmas = torch.sqrt(1 - alphas_bar)
    
    alphas_bar_prev = torch.cat([torch.tensor([1.0], device=device), alphas_bar[:-1]])
    Sigmas_cond = (1 - alphas) * (1 - torch.sqrt(alphas_bar_prev)) / (1 - alphas_bar)
    sigmas_cond = torch.sqrt(Sigmas_cond)
    sigmas_cond[0] = 0.0
    print(f"init sigma = {sigmas[-1]:.2e}")

    YN = torch.zeros([args.Hsample, Nu], device=device)

    # seul // tide modification 적용된 reverse_once
    def reverse_once(i, rng_seed, Ybar_i, state_init, planning_data):
        sigma = sigmas[i]
        # Gaussian sampling 및 궤적 Yi 계산 (구조 유지)
        Yi = Ybar_i * torch.sqrt(alphas_bar[i])
        eps_u = torch.randn((args.Nsample, args.Hsample, Nu), device=device) 
        Y0s = eps_u * sigma + Ybar_i
        Y0s = torch.clamp(Y0s, -1.0, 1.0)

        # 핵심: TiDE One-shot 예측 및 보상 호출
        qs = env.predict_one_shot(state_init, Y0s) 
        rews = env.get_reward_one_shot(qs, Y0s, planning_data) 
        
        # 보상 정규화 (구조 유지)
        rew_std = rews.std()
        rew_std = torch.where(rew_std < 1e-4, torch.tensor(1.0, device=device), rew_std)
        logp0 = (rews - rews.mean()) / rew_std / args.temp_sample 

        weights = torch.softmax(logp0, dim=0).float()
        Ybar = torch.einsum("n,nij->ij", weights, Y0s)

        # score function 및 다음 단계 계산 (구조 유지)
        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + torch.sqrt(alphas_bar[i]) * Ybar) 
        Yim1 = 1 / torch.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)

        alpha_bar_prev_val = alphas_bar[i-1] if i > 0 else torch.tensor(1.0, device=device)
        Ybar_im1 = Yim1 / torch.sqrt(alpha_bar_prev_val)

        return Ybar_im1, rews.mean()

    # planning_data 전달 가능하도록 수정
    def reverse(YN, seed, planning_data):
        Yi = YN
        Ybars = []
        with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                Yi, rew = reverse_once(i, seed, Yi, state_init, planning_data)
                Ybars.append(Yi.detach().cpu().numpy())
                pbar.set_postfix({"rew": f"{rew.item():.2e}"})
        return np.array(Ybars)

    # 실행
    Yi_history = reverse(YN, args.seed, planning_data)
    Yi_last = torch.tensor(Yi_history[-1], device=device)

    # 렌더링 및 결과 저장 (구조 유지)
    if not args.not_render:
        path = f"results/{args.env_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(f"{path}/mu_0ts.npy", Yi_history)
        
        # TiDE 결과 시각화
        final_qs = env.predict_one_shot(state_init, Yi_last.unsqueeze(0)).squeeze(0).cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.plot(final_qs[:, 0] * (x1_diff/2), label="x1 Optimized", color='r')
        plt.plot(planning_data['x1_ref'].cpu().numpy() * (x1_diff/2), 'r:', label="x1 Ref")
        plt.plot(final_qs[:, 1] * (x2_diff/2), label="x2 Result", color='gray')
        plt.legend()
        plt.savefig(f"{path}/tide_result.png")
        plt.show()

    return Yi_last.mean().item()


if __name__ == "__main__":
    rew_final = run_diffusion(args=tyro.cli(Args))
    print(f"final reward = {rew_final:.2e}")