import torch
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from experiments import EXPERIMENTS
import json
import time

from nn_functions import surrogate
from moving_average import moving_average_1d
from GAMMA_obj_temp_depth import GAMMA_obj
import mbd.envs 
from mbd.planners.mbd_planner import Args, reverse_once


# ==========================================
# normalization
# ==========================================

x_min = torch.tensor([[0.0, 0.75, 0.75, 504.26]], dtype=torch.float32)
x_max = torch.tensor([[7.5, 20.0, 20.0, 732.298]], dtype=torch.float32)
y_min = torch.tensor([[436.608, -0.559]], dtype=torch.float32)
y_max = torch.tensor([[4509.855, 0.551]], dtype=torch.float32)


def normalize_x(x, dim_id):
    xmin = x_min.to(x.device)[0, dim_id]
    xmax = x_max.to(x.device)[0, dim_id]
    return 2 * (x - xmin) / (xmax - xmin) - 1


def inverse_normalize_x(x_norm, dim_id):
    xmin = x_min.to(x_norm.device)[0, dim_id]
    xmax = x_max.to(x_norm.device)[0, dim_id]
    return 0.5 * (x_norm + 1) * (xmax - xmin) + xmin


def normalize_y(y, dim_id):
    ymin = y_min.to(y.device)[0, dim_id]
    ymax = y_max.to(y.device)[0, dim_id]
    return 2 * (y - ymin) / (ymax - ymin) - 1


# ==========================================
# plotting
# ==========================================

def plot_fig(MPC_GAMMA, N_step, save_path=None):

    plt.figure(figsize=[8, 8])

    plt.subplot(3, 1, 1)
    plt.plot(MPC_GAMMA.x_past_save[:N_step, 0].detach().cpu().numpy())
    plt.plot(MPC_GAMMA.ref[:N_step].detach().cpu().numpy(), 'r--')
    plt.ylabel("Temp")

    plt.subplot(3, 1, 2)
    plt.plot(MPC_GAMMA.x_past_save[:N_step, 1].detach().cpu().numpy())
    plt.ylabel("Depth")

    plt.subplot(3, 1, 3)
    plt.plot(MPC_GAMMA.u_past_save[:N_step].detach().cpu().numpy())
    plt.ylabel("Laser")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()


# ==========================================
# diffusion step
# ==========================================

def run_one_step_diffusion(GAMMA_obj, env, mbd_args, P, window):

    device = torch.device(mbd_args.device)

    mp_temp_past_t = GAMMA_obj.x_past.T.unsqueeze(0).to(device)
    laser_past_t = GAMMA_obj.u_past.view(1, -1, 1).to(device)

    fix_cov_past = GAMMA_obj.fix_cov_all[
        GAMMA_obj.MPC_counter-window:GAMMA_obj.MPC_counter
    ]

    fix_cov_past_t = torch.as_tensor(
        fix_cov_past,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    fix_cov_past_s = normalize_x(fix_cov_past_t, dim_id=[0,1,2])
    laser_past_s = normalize_x(laser_past_t, dim_id=[3])
    mp_temp_past_s = normalize_y(mp_temp_past_t, dim_id=[0,1])

    state_init_s = torch.cat(
        (mp_temp_past_s, fix_cov_past_s, laser_past_s),
        dim=2
    )

    mp_temp_ref = GAMMA_obj.ref[
        GAMMA_obj.MPC_counter:GAMMA_obj.MPC_counter+P
    ]

    mp_temp_ref_t = torch.as_tensor(
        mp_temp_ref,
        dtype=torch.float32,
        device=device
    ).reshape(1,P,1)

    fix_cov_future = GAMMA_obj.fix_cov_all[
        GAMMA_obj.MPC_counter:GAMMA_obj.MPC_counter+P
    ]

    fix_cov_future_t = torch.as_tensor(
        fix_cov_future,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    u_prev = normalize_x(
        GAMMA_obj.u_past[-1].view(1,1),
        dim_id=[3]
    ).to(device)

    planning_data = {
        'ref': normalize_y(mp_temp_ref_t,dim_id=[0])[:,:,0].unsqueeze(-1),
        'con': torch.tensor([[0.1423,0.4126]]*P,device=device).reshape(1,P,2),
        'fix_cov_future': normalize_x(fix_cov_future_t,dim_id=[0,1,2]),
        'u_prev': u_prev
    }

    betas = torch.linspace(
        mbd_args.beta0,
        mbd_args.betaT,
        mbd_args.Ndiffuse,
        device=device
    )

    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas,dim=0)
    sigmas = torch.sqrt(1-alphas_bar)

    if not hasattr(GAMMA_obj,'u_refine_prev'):

        laser_future_guess = GAMMA_obj.u_past[-1].repeat(P)

        Yi = normalize_x(
            laser_future_guess.view(P,1),
            dim_id=[3]
        ).to(device)

    else:

        Yi = torch.cat([
            GAMMA_obj.u_refine_prev[1:],
            GAMMA_obj.u_refine_prev[-1:]
        ],dim=0)

    # ==========================
    # diffusion time start
    # ==========================

    if device.type=="cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for i in range(mbd_args.Ndiffuse-1,-1,-1):

        Yi = reverse_once(
            i,
            Yi,
            state_init_s,
            planning_data,
            env,
            mbd_args,
            alphas,
            alphas_bar,
            sigmas
        )

    if device.type=="cuda":
        torch.cuda.synchronize()

    diffusion_time = time.perf_counter() - start

    # ==========================
    # diffusion time end
    # ==========================

    GAMMA_obj.u_refine_prev = Yi.clone()

    u_applied = float(
        inverse_normalize_x(
            Yi[0,0].unsqueeze(0),
            dim_id=[3]
        )
    )

    x_current, depth_current = GAMMA_obj.run_sim_interval(u_applied)

    GAMMA_obj.x_past[:, :-1] = GAMMA_obj.x_past[:,1:]
    GAMMA_obj.x_past[0,-1] = x_current
    GAMMA_obj.x_past[1,-1] = depth_current

    GAMMA_obj.u_past[:-1] = GAMMA_obj.u_past[1:].clone()
    GAMMA_obj.u_past[-1] = u_applied

    GAMMA_obj.MPC_counter += 1

    new_state = torch.tensor(
        [[x_current,depth_current]],
        device=GAMMA_obj.x_past_save.device
    )

    GAMMA_obj.x_past_save = torch.cat(
        (GAMMA_obj.x_past_save,new_state),
        dim=0
    )

    new_u = torch.tensor(
        [[u_applied]],
        device=GAMMA_obj.u_past_save.device
    )

    GAMMA_obj.u_past_save = torch.cat(
        (GAMMA_obj.u_past_save,new_u),
        dim=0
    )

    return u_applied, diffusion_time


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    INPUT_DATA_DIR = "data"
    SIM_DIR_NAME = "single_track_square"
    BASE_LASER_FILE_DIR = "laser_power_profiles/csv"
    CLOUD_TARGET_BASE_PATH = "result"

    solidus_temp = 1600
    window = 50
    sim_interval = 5
    init_runs = 50
    P = 50

    weight_path = "TiDE_params_ensemble_partial_1.pkl"

    laser_num = 1

    csv_path = f"/home/ftk3187/github/DPC_research/02_DED/4_policy_0725/split_by_laser_power_number/laser_power_number_{laser_num}.csv"

    df = pd.read_csv(csv_path)

    if 'Laser_power' not in df.columns and 'laser_power' in df.columns:
        df.rename(columns={'laser_power':'Laser_power'},inplace=True)

    loc_Z = df["Z"].to_numpy().reshape(-1,1)
    dist_X = df["Dist_to_nearest_X"].to_numpy().reshape(-1,1)
    dist_Y = df["Dist_to_nearest_Y"].to_numpy().reshape(-1,1)

    fix_covariates_np = np.concatenate(
        (loc_Z,dist_X,dist_Y),
        axis=1
    )

    mp_temp_raw = df["melt_pool_temperature"].to_numpy()
    mp_temp = copy.deepcopy(mp_temp_raw)

    mp_temp[1:-2] = moving_average_1d(mp_temp_raw,4)

    laser_power_ref_np = df["Laser_power"].to_numpy().reshape(-1,1)

    mbd_args = Args(device=str(device))

    env = mbd.envs.get_env(
        "tide_env",
        weight_path,
        device=str(device)
    )

    GAMMA_class = GAMMA_obj(
        INPUT_DATA_DIR,
        SIM_DIR_NAME,
        BASE_LASER_FILE_DIR,
        CLOUD_TARGET_BASE_PATH,
        solidus_temp,
        window,
        init_runs,
        sim_interval,
        laser_power_number=laser_num
    )

    init_avg = GAMMA_class.run_initial_steps()

    init_avg = torch.tensor(
        init_avg,
        dtype=torch.float32
    ).to(device)[:, -window:]

    fix_covariates = torch.tensor(
        fix_covariates_np,
        dtype=torch.float32
    ).to(device)

    laser_power_ref = torch.tensor(
        laser_power_ref_np,
        dtype=torch.float32
    ).to(device)

    laser_power_past = laser_power_ref[:window]

    mp_temp_ref = torch.tensor(
        mp_temp,
        dtype=torch.float32
    ).to(device)

    GAMMA_class.ref = mp_temp_ref.clone()
    GAMMA_class.fix_cov_all = fix_covariates.clone()

    GAMMA_class.x_past = init_avg.clone()
    GAMMA_class.u_past = laser_power_past.clone()

    GAMMA_class.x_past_save = GAMMA_class.x_past.T.clone()
    GAMMA_class.u_past_save = GAMMA_class.u_past.clone()

    GAMMA_class.MPC_counter = window

    diffusion_times = []

    N_step = len(mp_temp_ref) - P

    for i in tqdm(range(N_step-P)):

        _, diff_time = run_one_step_diffusion(
            GAMMA_class,
            env,
            mbd_args,
            P,
            window
        )

        diffusion_times.append(diff_time)

    # ==========================================
    # save csv
    # ==========================================

    temp_series = GAMMA_class.x_past_save[:,0].cpu().numpy()
    depth_series = GAMMA_class.x_past_save[:,1].cpu().numpy()
    laser_series = GAMMA_class.u_past_save[:,0].cpu().numpy()
    ref_series = GAMMA_class.ref[:len(temp_series)].cpu().numpy()

    # diffusion time padding
    diff_times = [0]*window + diffusion_times

    df_out = pd.DataFrame({
        "temperature": temp_series,
        "depth": depth_series,
        "laser_power": laser_series,
        "reference_temp": ref_series,
        "diffusion_time_sec": diff_times[:len(temp_series)]
    })

    os.makedirs("results",exist_ok=True)

    save_path = "results/trajectory_with_time.csv"

    df_out.to_csv(save_path,index=False)

    print("saved:",save_path)