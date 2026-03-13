import os
import glob
import time
import shutil

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from mbd.planners.mbd_planner import Args, reverse_once
import mbd.envs


# ===============================
# Plot Style
# ===============================
plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "x1": "#ff0000",         # red
    "x2": "#808080",         # gray
    "ref": "#f4a09c",        # soft salmon
    "constraint": "#b3b3b3", # light gray
    "tide_x1": "#c9b400",    # yellow-gold
    "tide_x2": "#bfae00",    # olive-gold
    "u": "#1f77b4",          # blue
    "u_future": "#c9b400",   # yellow-gold
    "marker": "#000000",     # black
}


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
# 2. 애니메이션 생성
# ===============================
def create_animation(frame_dir, out_dir, tag, fps=5):
    png_files = sorted(glob.glob(os.path.join(frame_dir, "mpc_step_*.png")))
    if len(png_files) == 0:
        print(f"[{tag}] No PNG frames found in {frame_dir}")
        return

    images = [imageio.imread(png) for png in png_files]

    gif_path = os.path.join(out_dir, f"mpc_prediction_{tag}.gif")
    mp4_path = os.path.join(out_dir, f"mpc_prediction_{tag}.mp4")

    imageio.mimsave(gif_path, images, fps=fps)
    print(f"[{tag}] Saved GIF: {gif_path}")

    try:
        imageio.mimsave(mp4_path, images, fps=fps)
        print(f"[{tag}] Saved MP4: {mp4_path}")
    except Exception as e:
        print(f"[{tag}] MP4 저장 실패: {e}")
        print("[INFO] ffmpeg/imageio-ffmpeg 환경이 없으면 MP4가 실패할 수 있습니다. GIF는 정상 저장됩니다.")


# ===============================
# 3. 프레임 플롯 함수
# ===============================
def save_prediction_frame(
    i,
    window,
    args,
    tag,
    frame_dir,
    x_trajectory,
    u_trajectory,
    qs_future,
    Yi,
    Ref_traj,
    C_Ref_traj_up,
    C_Ref_traj_low,
    x1_diff,
    x2_diff,
    u_diff,
):
    x_plot = x_trajectory.detach().cpu().numpy()
    u_plot = u_trajectory.detach().cpu().numpy()
    qs_future_np = qs_future.detach().cpu().numpy()
    u_future_np = Yi.detach().cpu().numpy()

    current_idx = i + window
    hist_end = current_idx + 1

    t_hist = np.arange(hist_end)
    future_t = np.arange(current_idx + 1, current_idx + 1 + args.Hsample)

    fig, ax = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # ==========================
    # Top subplot: x1 / x2
    # ==========================
    ax[0].plot(
        t_hist,
        x_plot[:hist_end, 0] * (x1_diff / 2),
        color=COLORS["x1"],
        linewidth=2.0,
        label="x1"
    )
    ax[0].plot(
        t_hist,
        x_plot[:hist_end, 1] * (x2_diff / 2),
        color=COLORS["x2"],
        linewidth=2.0,
        label="x2"
    )

    ax[0].plot(
        np.arange(len(Ref_traj)),
        Ref_traj.cpu().numpy(),
        color=COLORS["ref"],
        linestyle=":",
        linewidth=1.2,
        label="x1 Reference"
    )

    ax[0].plot(
        np.arange(len(C_Ref_traj_up)),
        C_Ref_traj_up.cpu().numpy(),
        color=COLORS["constraint"],
        linestyle="--",
        linewidth=1.0,
        label="x2 Constraint"
    )
    ax[0].plot(
        np.arange(len(C_Ref_traj_low)),
        C_Ref_traj_low.cpu().numpy(),
        color=COLORS["constraint"],
        linestyle="--",
        linewidth=1.0
    )

    # TiDE horizon
    ax[0].plot(
        future_t,
        qs_future_np[:, 0] * (x1_diff / 2),
        linestyle="--",
        color=COLORS["tide_x1"],
        linewidth=2.0,
        label="TiDE x1 (horizon)"
    )
    ax[0].plot(
        future_t,
        qs_future_np[:, 1] * (x2_diff / 2),
        linestyle="--",
        color=COLORS["tide_x2"],
        linewidth=2.0,
        label="TiDE x2 (horizon)"
    )

    # current markers
    ax[0].scatter(
        current_idx,
        x_plot[current_idx, 0] * (x1_diff / 2),
        color=COLORS["marker"],
        s=18,
        zorder=5
    )
    ax[0].scatter(
        current_idx,
        x_plot[current_idx, 1] * (x2_diff / 2),
        color=COLORS["marker"],
        s=18,
        zorder=5
    )

    ax[0].set_ylabel("x1 / x2")
    ax[0].legend(loc="upper left", ncol=6, fontsize=10)
    ax[0].grid(True, alpha=0.3)

    # ==========================
    # Bottom subplot: u
    # ==========================
    ax[1].plot(
        t_hist,
        u_plot[:hist_end, 0] * (u_diff / 2),
        color=COLORS["u"],
        linewidth=2.0,
        label="u"
    )
    ax[1].plot(
        future_t,
        u_future_np[:, 0] * (u_diff / 2),
        linestyle="--",
        color=COLORS["u_future"],
        linewidth=2.0,
        label="u (horizon)"
    )

    ax[1].scatter(
        current_idx,
        u_plot[current_idx, 0] * (u_diff / 2),
        color=COLORS["marker"],
        s=18,
        zorder=5
    )

    ax[1].set_xlabel("Time Step", fontsize=14)
    ax[1].set_ylabel("u")
    ax[1].legend(loc="upper left", fontsize=10)
    ax[1].grid(True, alpha=0.3)

    fig.suptitle(f"MPC Prediction Horizon Visualization ({tag}) - step {i}", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(frame_dir, f"mpc_step_{i:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===============================
# 4. 최종 결과 플롯
# ===============================
def save_final_plot(
    png_dir,
    tag,
    x_trajectory,
    u_trajectory,
    Ref_traj,
    C_Ref_traj_up,
    C_Ref_traj_low,
    x1_diff,
    x2_diff,
    u_diff,
):
    x_plot = x_trajectory.detach().cpu().numpy()
    u_plot = u_trajectory.detach().cpu().numpy()
    t_all = np.arange(len(x_plot))

    fig, ax = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    ax[0].plot(t_all, x_plot[:, 0] * (x1_diff / 2), color=COLORS["x1"], linewidth=2.0, label="x1")
    ax[0].plot(t_all, x_plot[:, 1] * (x2_diff / 2), color=COLORS["x2"], linewidth=2.0, label="x2")
    ax[0].plot(
        np.arange(len(Ref_traj)),
        Ref_traj.cpu().numpy(),
        color=COLORS["ref"],
        linestyle=":",
        linewidth=1.2,
        label="x1 Reference"
    )
    ax[0].plot(
        np.arange(len(C_Ref_traj_up)),
        C_Ref_traj_up.cpu().numpy(),
        color=COLORS["constraint"],
        linestyle="--",
        linewidth=1.0,
        label="x2 Constraint"
    )
    ax[0].plot(
        np.arange(len(C_Ref_traj_low)),
        C_Ref_traj_low.cpu().numpy(),
        color=COLORS["constraint"],
        linestyle="--",
        linewidth=1.0
    )

    ax[0].set_ylabel("x1 / x2")
    ax[0].legend(loc="upper left", ncol=4, fontsize=10)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t_all, u_plot[:, 0] * (u_diff / 2), color=COLORS["u"], linewidth=2.0, label="u")
    ax[1].set_xlabel("Time Step", fontsize=14)
    ax[1].set_ylabel("u")
    ax[1].legend(loc="upper left", fontsize=10)
    ax[1].grid(True, alpha=0.3)

    fig.suptitle(f"MPC-Diffusion Final Result ({tag})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, f"mpc_diffusion_result_lambda0.1_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===============================
# 5. MPC 실행 함수
# ===============================
def run_mpc(args: Args, tag="default"):
    device = torch.device(args.device)

    # ===============================
    # 결과 저장 폴더 생성
    # ===============================
    base_dir = "results"
    csv_dir = os.path.join(base_dir, "csv")
    png_dir = os.path.join(base_dir, "png")
    frame_dir = os.path.join(png_dir, f"frames_{tag}")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    env = mbd.envs.get_env(args.env_name, device=args.device)

    x1_diff, x2_diff, u_diff = 9.32425809, 14.13760952, 9.99280029

    # ===============================
    # Reference trajectory 생성
    # ===============================
    scaling = 2
    Ref_traj_step = torch.tensor(
        np.repeat(np.array([0, -scaling, -scaling, 0, scaling, scaling]), 30),
        dtype=torch.float32,
        device=device
    )

    def sigmoid(x):
        return 1 / (1 + np.exp(-x / 4))

    x = np.linspace(0, 50, 50)
    y = -sigmoid(x - 25) * scaling + scaling
    Ref_traj_sigmoid = torch.tensor(y, dtype=torch.float32, device=device)

    x_sin = np.linspace(0, 2 * np.pi, 100)
    y_sin = scaling * np.sin(x_sin)
    Ref_traj_sin = torch.tensor(y_sin, dtype=torch.float32, device=device)

    Ref_traj = torch.cat((
        Ref_traj_step,
        Ref_traj_sigmoid,
        torch.zeros(20, dtype=torch.float32, device=device),
        Ref_traj_sin,
        Ref_traj_step,
        torch.zeros(20, dtype=torch.float32, device=device)
    ))

    tot_step = len(Ref_traj)

    x_constraint = np.linspace(0, tot_step - 1, tot_step)
    C_Ref_traj_up = torch.tensor(
        4 + 0.5 * np.sin(2 * np.pi * x_constraint / 100) - 0.004 * x_constraint,
        dtype=torch.float32,
        device=device
    )
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
            "x1_ref": sliding_data_ref_x1[i + window].to(device) / (x1_diff / 2),
            "x2_low": sliding_data_con_x2_low[i + window].to(device) / (x2_diff / 2),
            "x2_up": sliding_data_con_x2_up[i + window].to(device) / (x2_diff / 2),
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

        # 실제 MPC에 쓰는 입력은 첫 번째 control
        u_next_norm = Yi[0]

        # ===============================
        # Visualization용 TiDE future prediction
        # MPC 로직은 변경하지 않음
        # ===============================
        with torch.no_grad():
            u_future_norm = Yi.unsqueeze(0)   # (1, Hsample, 1)
            qs_future = env.predict_one_shot(state_init, u_future_norm).squeeze(0)  # (Hsample, 2)

        # 현재 step control 반영
        u_trajectory[i + window] = u_next_norm

        # 실제 시스템 rollout
        u_real = (u_next_norm * (u_diff / 2)).detach().cpu().numpy().reshape(1, 1)
        x_real = (
            x_trajectory[i + window]
            * torch.tensor([x1_diff / 2, x2_diff / 2], device=device)
        ).detach().cpu().numpy().reshape(2, 1)

        x_next_real = f_ss(x_real, u_real)

        x_next_norm = torch.tensor(x_next_real, device=device).float().squeeze()
        x_next_norm[0] /= (x1_diff / 2)
        x_next_norm[1] /= (x2_diff / 2)

        x_trajectory[i + window + 1] = x_next_norm

        # ===============================
        # 매 5 step마다 프레임 저장
        # ===============================
        if i % 5 == 0:
            save_prediction_frame(
                i=i,
                window=window,
                args=args,
                tag=tag,
                frame_dir=frame_dir,
                x_trajectory=x_trajectory,
                u_trajectory=u_trajectory,
                qs_future=qs_future,
                Yi=Yi,
                Ref_traj=Ref_traj,
                C_Ref_traj_up=C_Ref_traj_up,
                C_Ref_traj_low=C_Ref_traj_low,
                x1_diff=x1_diff,
                x2_diff=x2_diff,
                u_diff=u_diff,
            )

    # ===============================
    # 시간 계산
    # ===============================
    end_time = time.time()
    total_time = end_time - start_time
    time_per_step = total_time / num_sim_steps

    print(f"\n[{tag}] Total Time: {total_time:.3f} sec")
    print(f"[{tag}] Avg Time per Step: {time_per_step:.6f} sec")

    # ===============================
    # 최종 결과 plot 저장
    # ===============================
    save_final_plot(
        png_dir=png_dir,
        tag=tag,
        x_trajectory=x_trajectory,
        u_trajectory=u_trajectory,
        Ref_traj=Ref_traj,
        C_Ref_traj_up=C_Ref_traj_up,
        C_Ref_traj_low=C_Ref_traj_low,
        x1_diff=x1_diff,
        x2_diff=x2_diff,
        u_diff=u_diff,
    )

    # ===============================
    # CSV 저장
    # ===============================
    x_plot = x_trajectory.detach().cpu().numpy()
    u_plot = u_trajectory.detach().cpu().numpy()

    df = pd.DataFrame({
        "time_index": np.arange(len(x_plot)),
        "x1_state": x_plot[:, 0] * (x1_diff / 2),
        "x2_state": x_plot[:, 1] * (x2_diff / 2),
        "u_input": u_plot[:, 0] * (u_diff / 2),
        "elapsed_total_sec": total_time,
        "elapsed_per_step_sec": time_per_step
    })

    csv_path = os.path.join(csv_dir, f"mpc_diffusion_result_lambda0.1_{tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[{tag}] Saved CSV: {csv_path}")

    # ===============================
    # GIF / MP4 생성
    # ===============================
    create_animation(
        frame_dir=frame_dir,
        out_dir=png_dir,
        tag=tag,
        fps=5
    )


# ===============================
# 6. Main (Single Run)
# ===============================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = Args(
        env_name="tide_env",
        Hsample=10,
        Nsample=1024,
        Ndiffuse=50,
        temp_sample=0.1,
        seed=42,
        device=device
    )

    run_mpc(args, tag="Nsample1024_lambda0.1_Ndiff50_pres")