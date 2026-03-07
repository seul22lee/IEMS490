import torch
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

# 외부 모듈 임포트
from nn_functions import surrogate
from moving_average import moving_average_1d
from GAMMA_obj_temp_depth import GAMMA_obj
import mbd.envs 
from mbd.planners.mbd_planner import Args, reverse_once

# ==========================================
# 1. 노트북 원본 정규화 파라미터 및 함수 (Cell 5, 7 복사)
# ==========================================
# 노트북의 실제 값을 그대로 사용합니다.
x_min = torch.tensor([[0.0, 0.75, 0.75, 504.26]], dtype=torch.float32)
x_max = torch.tensor([[7.5, 20.0, 20.0, 732.298]], dtype=torch.float32)
y_min = torch.tensor([[436.608, -0.559]], dtype=torch.float32)
y_max = torch.tensor([[4509.855, 0.551]], dtype=torch.float32)

def normalize_x(x, dim_id):
    # dim_id에 해당하는 min/max 값을 가져와서 전체 x에 대해 연산 (차원 에러 방지)
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

def plot_fig(MPC_GAMMA, N_step, save_path=None):
    plt.figure(figsize=[8, 8])
    plt.subplot(3, 1, 1)
    plt.plot(MPC_GAMMA.x_past_save[:N_step, 0].detach().cpu().numpy(), label="GAMMA simulation")
    plt.plot(MPC_GAMMA.ref[:N_step].detach().cpu().numpy(), 'r--', label="Reference")
    plt.ylabel("Temp (K)"); plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(MPC_GAMMA.x_past_save[:N_step, 1].detach().cpu().numpy(), label="GAMMA simulation")
    plt.axhline(y=0.4126, color='k', linestyle='--'); plt.axhline(y=0.1423, color='k', linestyle='--')
    plt.ylabel("Depth (mm)")
    plt.subplot(3, 1, 3)
    plt.plot(MPC_GAMMA.u_past_save[:N_step].detach().cpu().numpy(), color='purple')
    plt.ylabel("Laser (W)"); plt.xlabel("Step")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200); plt.close()

# ==========================================
# 2. MBD 실행 함수 (노트북 Cell 6 로직 반영)
# ==========================================
def run_one_step_diffusion(GAMMA_obj, env, mbd_args, P, window):
    device = torch.device(mbd_args.device)

    # 과거 시퀀스 준비
    mp_temp_past_t = GAMMA_obj.x_past.T.unsqueeze(0).to(device)  # [1, 50, 2]
    laser_past_t = GAMMA_obj.u_past.view(1, -1, 1).to(device)     # [1, 50, 1]
    fix_cov_past = GAMMA_obj.fix_cov_all[GAMMA_obj.MPC_counter - window:GAMMA_obj.MPC_counter, :]
    fix_cov_past_t = torch.as_tensor(fix_cov_past, dtype=torch.float32, device=device).unsqueeze(0)

    # 정규화 (노트북 방식: laser_past_t가 1채널이어도 [3]번 파라미터로 정상 계산됨)
    fix_cov_past_s = normalize_x(fix_cov_past_t, dim_id=[0, 1, 2])
    laser_past_s = normalize_x(laser_past_t, dim_id=[3])
    mp_temp_past_s = normalize_y(mp_temp_past_t, dim_id=[0, 1])

    #state_init_s = torch.cat((fix_cov_past_s, laser_past_s, mp_temp_past_s), dim=2)
    state_init_s = torch.cat(
        (mp_temp_past_s, fix_cov_past_s, laser_past_s),
        dim=2
    )

    # 미래 Planning 데이터 (Reference & Constraints)
    mp_temp_ref = GAMMA_obj.ref[GAMMA_obj.MPC_counter:GAMMA_obj.MPC_counter + P]
    mp_temp_ref_t = torch.as_tensor(mp_temp_ref, dtype=torch.float32, device=device).reshape(1, P, 1)
    
    fix_cov_future = GAMMA_obj.fix_cov_all[GAMMA_obj.MPC_counter:GAMMA_obj.MPC_counter + P, :]
    fix_cov_future_t = torch.as_tensor(fix_cov_future, dtype=torch.float32, device=device).unsqueeze(0)

    planning_data = {
        'ref': normalize_y(mp_temp_ref_t, dim_id=[0])[:, :, 0].unsqueeze(-1),
        #'con': torch.tensor([[0.1423, 0.4126]] * P, device=device).reshape(1, P, 2), # 노트북 상수
        'con': torch.tensor([[0.075, 0.225]] * P, device=device).reshape(1, P, 2), # 노트북 상수
        'fix_cov_future': normalize_x(fix_cov_future_t, dim_id=[0, 1, 2])
    }

    # Diffusion Process (Warm Start 적용)
    betas = torch.linspace(mbd_args.beta0, mbd_args.betaT, mbd_args.Ndiffuse, device=device)
    alphas = 1.0 - betas; alphas_bar = torch.cumprod(alphas, dim=0); sigmas = torch.sqrt(1 - alphas_bar)

    if not hasattr(GAMMA_obj, 'u_refine_prev'):
        Yi = torch.zeros([P, env.action_size], device=device)
    else:
        # Receding Horizon Warm Start
        Yi = torch.cat([GAMMA_obj.u_refine_prev[1:], GAMMA_obj.u_refine_prev[-1:]], dim=0)

    for i in range(mbd_args.Ndiffuse - 1, -1, -1):
        # mbd_planner.py의 reverse_once 호출
        Yi = reverse_once(i, Yi, state_init_s, planning_data, env, mbd_args, alphas, alphas_bar, sigmas)

    GAMMA_obj.u_refine_prev = Yi.clone()
    u_applied = float(inverse_normalize_x(Yi[0, 0].unsqueeze(0), dim_id=[3]))

    # 시뮬레이션 물리 업데이트 및 저장 (노트북 Cell 6 후반부)
    x_current, depth_current = GAMMA_obj.run_sim_interval(u_applied)

    GAMMA_obj.x_past[:, :-1] = GAMMA_obj.x_past[:, 1:]
    GAMMA_obj.x_past[0, -1] = x_current
    GAMMA_obj.x_past[1, -1] = depth_current
    GAMMA_obj.u_past[:-1] = GAMMA_obj.u_past[1:].clone()
    GAMMA_obj.u_past[-1] = u_applied
    GAMMA_obj.MPC_counter += 1

    # 시각화 데이터 기록
    new_state = torch.tensor([[x_current, depth_current]], device=GAMMA_obj.x_past_save.device)
    GAMMA_obj.x_past_save = torch.cat((GAMMA_obj.x_past_save, new_state), dim=0)
    new_u = torch.tensor([[u_applied]], device=GAMMA_obj.u_past_save.device)
    GAMMA_obj.u_past_save = torch.cat((GAMMA_obj.u_past_save, new_u), dim=0)

    print("state_init_s:", state_init_s.shape)
    print("fix_cov_future:", planning_data["fix_cov_future"].shape)
    print("env.action_size:", env.action_size)
    print("model.future_cov_dim:", env.model.future_cov_dim)

    return u_applied

# ==========================================
# 4. Main 실행부 (노트북 Step 11 로직과 100% 동일화)
# ==========================================
if __name__ == "__main__":
    # 노트북과 동일한 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mbd_args = Args(device=str(device))
    
    # ── 노트북 상수 설정 (Cell 11에서 복사) ──────────────────────
    INPUT_DATA_DIR = "data"
    SIM_DIR_NAME = "single_track_square"
    BASE_LASER_FILE_DIR = "laser_power_profiles/csv"
    CLOUD_TARGET_BASE_PATH = "result"
    solidus_temp = 1600 # 노트북 기준
    window = 50
    sim_interval = 5
    init_runs = 50
    P = 50

    # TiDE 환경 로드
    weight_path = "TiDE_params_single_track_square_MV_temp_depth_less_cov_0915_w50_p50.pkl"
    env = mbd.envs.get_env("tide_env", weight_path, device=str(device))

    # 결과 저장 경로
    save_dir = "./results/plots"
    os.makedirs(save_dir, exist_ok=True)

    # ── 루프 시작 (노트북 range(1, 15) 반영) ───────────────────────────
    for laser_num in range(1, 2):
        try:
            # Step 1: 노트북에 명시된 CSV 경로 (절대 경로)
            csv_path = f"/home/ftk3187/github/DPC_research/02_DED/4_policy_0725/split_by_laser_power_number/laser_power_number_{laser_num}.csv"
            df = pd.read_csv(csv_path)

            # [수정] 컬럼명 오류 방지: 대소문자 확인 (노트북은 'Laser_power' 사용)
            if 'Laser_power' not in df.columns and 'laser_power' in df.columns:
                df.rename(columns={'laser_power': 'Laser_power'}, inplace=True)

            # Step 2: GAMMA 객체 생성
            GAMMA_class = GAMMA_obj(INPUT_DATA_DIR, SIM_DIR_NAME, BASE_LASER_FILE_DIR,
                                    CLOUD_TARGET_BASE_PATH, solidus_temp, window,
                                    init_runs, sim_interval, laser_power_number=laser_num)

            # Step 3: 초기 스텝 실행 (노트북 로직)
            init_avg = GAMMA_class.run_initial_steps()
            init_avg = torch.tensor(init_avg, dtype=torch.float32).to(device)[:, -window:]

            # Step 4: 공변량 및 참조 궤적 처리 (노트북 Step 4 복사)
            loc_Z = df["Z"].to_numpy().reshape(-1, 1)
            dist_X = df["Dist_to_nearest_X"].to_numpy().reshape(-1, 1)
            dist_Y = df["Dist_to_nearest_Y"].to_numpy().reshape(-1, 1)
            fix_covariates = torch.tensor(np.concatenate((loc_Z, dist_X, dist_Y), axis=1), dtype=torch.float32).to(device)

            laser_power_ref = torch.tensor(df["Laser_power"].to_numpy().reshape(-1, 1), dtype=torch.float32).to(device)
            laser_power_past = laser_power_ref[:window]

            mp_temp_raw = df["melt_pool_temperature"].to_numpy()
            mp_temp = copy.deepcopy(mp_temp_raw)
            mp_temp[1:-2] = moving_average_1d(mp_temp_raw, 4)
            mp_temp_ref = torch.tensor(mp_temp, dtype=torch.float32).to(device)

            # Step 5: GAMMA_class 변수 주입 (노트북 Step 5 복사)
            GAMMA_class.ref = mp_temp_ref.clone()
            GAMMA_class.fix_cov_all = fix_covariates.clone()
            GAMMA_class.x_past = init_avg.clone()   # 노트북 명칭 확인: x_past
            GAMMA_class.u_past = laser_power_past.clone()
            GAMMA_class.x_past_save = GAMMA_class.x_past.T.clone()
            GAMMA_class.u_past_save = GAMMA_class.u_past.clone()
            GAMMA_class.MPC_counter = window
            
            # x_hat_current 등 추가 상태 초기화 (노트북 로직)
            GAMMA_class.x_hat_current = GAMMA_class.x_past[:, -1]
            GAMMA_class.x_sys_current = GAMMA_class.x_past[:, -1].reshape(2, 1)

            # Step 6: 시뮬레이션 루프
            N_step = len(mp_temp_ref) - P
            
            # 중간 플롯을 저장할 디렉토리 생성
            laser_plot_dir = os.path.join(save_dir, f"laser_{laser_num}")
            os.makedirs(laser_plot_dir, exist_ok=True)

            # i 루프 범위 노트북과 동일화 (N_step - P)
            for i in tqdm(range(N_step - P), desc=f"Laser {laser_num} (MBD)"):
                # MBD 기반 제어 1스텝 실행
                run_one_step_diffusion(GAMMA_class, env, mbd_args, P, window)
                
                # 100개 timestep마다 플롯 저장
                current_step = i + 1
                if current_step % 100 == 0:
                    mid_plot_path = os.path.join(laser_plot_dir, f"step_{current_step}.png")
                    # plot_fig(객체, 현재까지 진행된 길이, 저장경로)
                    plot_fig(GAMMA_class, GAMMA_class.MPC_counter, save_path=mid_plot_path)

            # 최종 플롯 저장
            final_plot_path = os.path.join(save_dir, f"final_plot_laser_{laser_num}.png")
            plot_fig(GAMMA_class, GAMMA_class.MPC_counter, save_path=final_plot_path)

            print(f"✅ Completed laser_power_number {laser_num}")

        except Exception as e:
            print(f"❌ Error in laser_power_number {laser_num}: {e}")
            import traceback
            traceback.print_exc()