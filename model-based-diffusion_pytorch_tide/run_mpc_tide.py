import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from mbd.planners.mbd_planner import Args, reverse_once
import mbd.envs
import pandas as pd


# ===============================
# 1. 물리 모델 (실제 시스템)
# ===============================
def f_ss(x0, u):
    A = np.array([[0.3, 0.1], [0.1, 0.2]])
    B = np.array([[0.5], [1.0]])
    mu = np.array([[0], [0]])
    w = np.array([[0.05], [0.1]])

    x_next = A @ x0 + B @ u + np.random.normal(mu, w, size=(2, 1))
    return x_next


# ===============================
# 2. MPC 실행 함수
# ===============================
import time
import os


def run_mpc(args: Args, tag="default"):

    device = torch.device(args.device)

    # ===============================
    # 결과 저장 폴더 생성
    # ===============================
    base_dir = "results"
    csv_dir = os.path.join(base_dir, "csv")
    png_dir = os.path.join(base_dir, "png")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    env = mbd.envs.get_env(args.env_name, device=args.device)

    x1_diff, x2_diff, u_diff = 9.32425809, 14.13760952, 9.99280029

    # ===============================
    # Reference trajectory 생성
    # ===============================
    scaling = 2
    Ref_traj_step = torch.tensor(np.repeat(np.array([0, -scaling, -scaling, 0, scaling, scaling]), 30))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x / 4))

    x = np.linspace(0, 50, 50)
    y = -sigmoid(x - 25) * scaling + scaling
    Ref_traj_sigmoid = torch.tensor(y)

    x_sin = np.linspace(0, 2 * np.pi, 100)
    y_sin = scaling * np.sin(x_sin)
    Ref_traj_sin = torch.tensor(y_sin)

    Ref_traj = torch.cat((
        Ref_traj_step,
        Ref_traj_sigmoid,
        torch.zeros(20),
        Ref_traj_sin,
        Ref_traj_step,
        torch.zeros(20)
    ))

    tot_step = len(Ref_traj)

    x_constraint = np.linspace(0, tot_step, tot_step)
    C_Ref_traj_up = torch.tensor(4 + 0.5 * np.sin(2 * np.pi * x_constraint / 100) - 0.004 * x_constraint)
    C_Ref_traj_low = -C_Ref_traj_up

    window = 10

    sliding_data_ref_x1 = Ref_traj.unfold(0, window, 1)
    sliding_data_con_x2_low = C_Ref_traj_low.unfold(0, window, 1)
    sliding_data_con_x2_up = C_Ref_traj_up.unfold(0, window, 1)

    num_sim_steps = 530
    x_trajectory = torch.zeros((num_sim_steps + window + 1, 2), device=device).float()
    u_trajectory = torch.zeros((num_sim_steps + window + 1, 1), device=device).float()

    # ===============================
    # Diffusion Scheduling
    # ===============================
    betas = torch.linspace(args.beta0, args.betaT, args.Ndiffuse, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    sigmas = torch.sqrt(1 - alphas_bar)

    # ===============================
    # 시간 측정 시작
    # ===============================
    start_time = time.time()

    # ===============================
    # MPC Loop
    # ===============================
    for i in tqdm(range(num_sim_steps), desc=f"MPC ({tag})"):

        curr_x_win = x_trajectory[i: i + window]
        curr_u_win = u_trajectory[i: i + window]
        state_init = torch.cat([curr_x_win, curr_u_win], dim=-1).unsqueeze(0)

        planning_data = {
            'x1_ref': sliding_data_ref_x1[i + window].to(device) / (x1_diff / 2),
            'x2_low': sliding_data_con_x2_low[i + window].to(device) / (x2_diff / 2),
            'x2_up': sliding_data_con_x2_up[i + window].to(device) / (x2_diff / 2),
        }

        Yi = torch.zeros([args.Hsample, env.action_size], device=device)

        for d in range(args.Ndiffuse - 1, -1, -1):
            Yi, _ = reverse_once(
                d,
                args.seed,
                Yi,
                state_init,
                planning_data,
                env,
                args,
                alphas,
                alphas_bar,
                sigmas
            )

        u_next_norm = Yi[0]
        u_trajectory[i + window] = u_next_norm

        u_real = (u_next_norm * (u_diff / 2)).cpu().numpy().reshape(1, 1)
        x_real = (x_trajectory[i + window] *
                  torch.tensor([x1_diff / 2, x2_diff / 2], device=device)
                  ).cpu().numpy().reshape(2, 1)

        x_next_real = f_ss(x_real, u_real)

        x_next_norm = torch.tensor(x_next_real, device=device).float().squeeze()
        x_next_norm[0] /= (x1_diff / 2)
        x_next_norm[1] /= (x2_diff / 2)

        x_trajectory[i + window + 1] = x_next_norm

    # ===============================
    # 시간 계산
    # ===============================
    end_time = time.time()
    total_time = end_time - start_time
    time_per_step = total_time / num_sim_steps

    print(f"\n[{tag}] Total Time: {total_time:.3f} sec")
    print(f"[{tag}] Avg Time per Step: {time_per_step:.6f} sec")

    # ===============================
    # 결과 시각화 저장
    # ===============================
    plt.figure(figsize=(12, 6))
    x_plot = x_trajectory.cpu().numpy()

    plt.plot(x_plot[:, 0] * (x1_diff / 2), label="x1 (State)")
    plt.plot(Ref_traj.numpy(), linestyle=':', label="x1 (Reference)")
    plt.plot(x_plot[:, 1] * (x2_diff / 2), label="x2 (State)")
    plt.plot(C_Ref_traj_up.numpy(), 'k--', alpha=0.3)
    plt.plot(C_Ref_traj_low.numpy(), 'k--', alpha=0.3)

    plt.legend()
    plt.title(f"MPC-Diffusion ({tag})")
    plt.savefig(os.path.join(png_dir, f"mpc_diffusion_result_lambda0.1_{tag}.png"))
    plt.close()

    # ===============================
    # CSV 저장
    # ===============================
    u_plot = u_trajectory.cpu().numpy()

    df = pd.DataFrame({
        "time_index": np.arange(len(x_plot)),
        "x1_state": x_plot[:, 0] * (x1_diff / 2),
        "x2_state": x_plot[:, 1] * (x2_diff / 2),
        "u_input": u_plot[:, 0] * (u_diff / 2),
        "elapsed_total_sec": total_time,
        "elapsed_per_step_sec": time_per_step
    })

    df.to_csv(os.path.join(csv_dir, f"mpc_diffusion_result_lambda0.1_{tag}.csv"), index=False)

# ===============================
# 3. DOE 실행 함수
# ===============================
def run_doe():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n===== DOE 1: Nsample =====")
    for ns in [1024, 2048, 4096]:
        args = Args(
            env_name="tide_env",
            Hsample=10,
            Nsample=ns,
            Ndiffuse=50,
            temp_sample=0.1,
            seed=42,
            device=device
        )
        run_mpc(args, tag=f"Nsample_{ns}")

    print("\n===== DOE 2: Ndiffuse =====")
    for nd in [10, 50, 100]:
        args = Args(
            env_name="tide_env",
            Hsample=10,
            Nsample=2048,
            Ndiffuse=nd,
            temp_sample=0.1,
            seed=42,
            device=device
        )
        run_mpc(args, tag=f"Ndiffuse_{nd}")


# ===============================
# 4. Main
# ===============================
# ===============================
# 4. Main (Single Run)
# ===============================
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = Args(
        env_name="tide_env",
        Hsample=10,
        Nsample=1024,   # 원하는 값
        Ndiffuse=50,    # 원하는 값
        temp_sample=0.1,
        seed=42,
        device=device
    )

    run_mpc(args, tag="Nsample1024_lambda0.1_Ndiff50_pres")