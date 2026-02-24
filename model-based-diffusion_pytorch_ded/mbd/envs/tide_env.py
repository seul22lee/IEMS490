import torch
import numpy as np
from .TiDE import TideModule 
# mbd/envs/tide_env.py 최상단
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .TiDE import TideModule

import pickle
import os
from .TiDE import TideModule

class TiDEDynamicsEnv:
    def __init__(self, model_params: dict, weight_path: str, device="cuda"):
        self.device = torch.device(device)
        
        # 1. .pkl 파일에서 모델 객체(인스턴스) 로드
        # weight_path가 '...final.pkl' 경로라고 가정합니다.
        with open(weight_path, 'rb') as file:
            nominal_params = pickle.load(file)
        
        # pkl 내부에 'model' 키로 저장된 객체를 사용
        self.model = nominal_params['model']
        
        # 2. .pth 파일에서 가중치(State Dict) 로드
        # .pkl 경로를 기반으로 .pth 경로를 생성합니다.
        pth_path = weight_path.replace('.pkl', '.pth')
        if os.path.exists(pth_path):
            # 이전에 사용하신 방식 그대로 load_state_dict 적용
            self.model.load_state_dict(torch.load(pth_path, map_location=self.device))
        else:
            print(f"Warning: .pth file not found at {pth_path}. Using model from .pkl.")

        self.model.to(self.device).eval()

        # 스케일링 상수 (세울님의 설정값 유지)
        self.x1_diff = 9.32425809 
        self.x2_diff = 14.13760952
        self.u_diff = 9.99280029
        
        # TiDE 모델 사양에 따른 크기 설정
        self.observation_size = self.model.output_dim # 2 (x1, x2)
        self.action_size = self.model.future_cov_dim     # 1 (u)
        self.H = self.model.output_chunk_length       # 10

    def predict_one_shot(self, state_init, u_sequence):
        """
        state_init: (1, 10, 3) -> 과거 [x1, x2, u] 정규화된 데이터
        u_sequence: (Nsample, 10, 1) -> Diffusion 생성 액션
        """
        Nsample = u_sequence.shape[0]
        past_cov = state_init.repeat(Nsample, 1, 1).to(self.device)
        
        with torch.no_grad():
            x_hat = self.model([past_cov, u_sequence, None])
            # x_hat[0,:,:,1] 로직 반영 (평균/중앙값 예측 추출)
            idx = 1 if x_hat.shape[-1] > 1 else 0
            predicted_qs = x_hat[:, :, :, idx] 
        return predicted_qs

    # def get_reward_one_shot(self, trajectory, u_sequence, planning_data):
    #         """
    #         trajectory: (Nsample, 10, 2) - TiDE가 예측한 미래 상태
    #         u_sequence: (Nsample, 10, 1) - Diffusion이 샘플링한 제어 입력
    #         planning_data: {x1_ref, x2_low, x2_up} (각 10 길이)
    #         """
    #         # 가중치 설정 (DPC_loss 로직 반영 및 Diffusion 튜닝)
    #         smoothness_weight = 0.5  # u의 변화량에 대한 패널티 (진동 억제)
    #         constraint_weight = 50.0 # 제약 위반에 대한 강력한 패널티 (Violation 해결)

    #         # 1. x1 Reference Tracking (MSE 기반)
    #         x1_pred = trajectory[:, :, 0] # (Nsample, 10)
    #         x1_ref = planning_data['x1_ref'].to(self.device).unsqueeze(0) # (1, 10)
    #         # DPC_loss처럼 제곱 오차 사용 (더 민감한 최적화)
    #         tracking_error = (x1_pred - x1_ref) ** 2
    #         reward_tracking = -torch.sqrt(tracking_error.mean(dim=-1))

    #         # 2. Control Effort / Smoothness (u_t - u_{t-1})
    #         # u_diff 로직 반영
    #         u_diff = u_sequence[:, 1:, :] - u_sequence[:, :-1, :] # (Nsample, 9, 1)
    #         smooth_term = (u_diff ** 2).mean(dim=(1, 2))
    #         reward_smoothness = -torch.sqrt(smoothness_weight * smooth_term)

    #         # 3. x2 Constraints (Relu 제곱 기반 - 강력한 제약)
    #         x2_pred = trajectory[:, :, 1] # (Nsample, 10)
    #         x2_low = planning_data['x2_low'].to(self.device).unsqueeze(0) # (1, 10)
    #         x2_up = planning_data['x2_up'].to(self.device).unsqueeze(0) # (1, 10)
            
    #         # Hinge loss를 제곱하여 제약 위반 시 패널티가 기하급수적으로 커지게 설정
    #         low_v = torch.relu(x2_low - x2_pred) ** 2
    #         up_v = torch.relu(x2_pred - x2_up) ** 2
    #         constr_term = (low_v + up_v).mean(dim=-1)
    #         reward_constraint = -torch.sqrt(constraint_weight * constr_term)

    #         # 최종 리워드 (Diffusion은 이 값을 최대화하는 방향으로 샘플링함)
    #         return reward_tracking + reward_smoothness + reward_constraint
    
    def get_reward_one_shot(self, trajectory, u_sequence, planning_data):
        """
        논문의 Eq (2) p0(Y) ∝ pd(Y)pJ(Y)pg(Y) 방식을 따름
        """
        # 온도 파라미터 (논문의 lambda 역활, 작을수록 최적해에 집중)
        lambda_inv = 2 # 논문 약 0.1 
        
        # 1. Optimality (pJ): Tracking & Smoothness
        x1_pred = trajectory[:, :, 0]
        x1_ref = planning_data['x1_ref'].to(self.device).unsqueeze(0)
        tracking_cost = ((x1_pred - x1_ref) ** 2).mean(dim=-1)
        
        u_diff = u_sequence[:, 1:, :] - u_sequence[:, :-1, :]
        smooth_cost = (u_diff ** 2).mean(dim=(1, 2))
        
        # J(Y) = tracking + smoothness
        total_cost = tracking_cost + 0.5 * smooth_cost
        p_J = torch.exp(-total_cost / lambda_inv) # Optimality 분포

        # 2. Constraints (pg): x2 Boundary
        x2_pred = trajectory[:, :, 1]
        x2_low = planning_data['x2_low'].to(self.device).unsqueeze(0)
        x2_up = planning_data['x2_up'].to(self.device).unsqueeze(0)
        
        # 논문의 1(g <= 0) 지시함수를 부드러운 확률로 표현 (Constraint Violation Likelihood)
        # 위반이 0이면 exp(0)=1, 위반이 커지면 exp(-inf)=0으로 수렴
        violation = (torch.relu(x2_low - x2_pred)**2 + torch.relu(x2_pred - x2_up)**2).mean(dim=-1)
        constraint_scale = 300.0 
        p_g = torch.exp(-constraint_scale * violation) # Constraint 분포

        # 3. Target Distribution: p0 ∝ p_J * p_g (p_d는 모델 구조에 포함되어 있다고 가정)
        p_0 = p_J * p_g

        # Diffusion의 가이던스로 사용하기 위해 다시 log를 취함 (Numerical Stability를 위함)
        # 실제로는 log(p_J * p_g) = log(p_J) + log(p_g)와 같아지지만, 
        # '곱셈' 구조를 명시적으로 제어하고 싶을 때 이 개념을 사용합니다.
        return torch.log(p_0 + 1e-8)