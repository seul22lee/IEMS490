# mbd/envs/tide_env.py

import torch
import numpy as np
import pickle
import os
import sys
import io

# TiDE 모델 클래스가 있는 경로 설정
# TiDE.py가 tide_env.py와 같은 폴더에 있다고 가정합니다.
from .TiDE import TideModule 
# mbd/envs/tide_env.py 수정

import torch
import numpy as np
import pickle
import os
import sys
import io

# TiDE.py 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from TiDE import TideModule
except ImportError:
    import TiDE
    TideModule = TiDE.TideModule

class TiDEEngineeringEnv:
    def __init__(self, weight_path: str, device="cuda"):
        self.device = torch.device(device)
        
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight path not found: {weight_path}")

        # [핵심 수정] Unpickler 내부에서 외부 self.device에 접근하기 위해 클로저 사용
        target_device = self.device 

        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    # 로드 시점에 GPU 점유 문제를 피하기 위해 무조건 'cpu'로 먼저 읽습니다.
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                return super().find_class(module, name)

        try:
            with open(weight_path, 'rb') as file:
                params = CustomUnpickler(file).load()
            
            self.model = params['model']
            # CPU로 로드된 모델을 최종 타겟 장치(GPU 등)로 이동
            self.model.to(target_device).eval()
            
        except Exception as e:
            # torch.load가 직접 가능한 pkl 형식일 경우 대비
            with open(weight_path, 'rb') as file:
                params = torch.load(file, map_location='cpu', weights_only=False)
            self.model = params['model']
            self.model.to(target_device).eval()

        # 가중치 파일(.pth) 처리
        pth_path = weight_path.replace('.pkl', '.pth')
        if os.path.exists(pth_path):
            self.model.load_state_dict(torch.load(pth_path, map_location=target_device))
        
        self.L = self.model.input_chunk_length
        self.H = self.model.output_chunk_length
        self.future_cov_dim_total = self.model.future_cov_dim   # expected = 4
        self.action_size = 1                                    # laser only

        # 역규화 및 정규화를 위한 상수 (노트북 값 반영)
        self.x1_diff = 9.32425809
        self.x2_diff = 14.13760952
        self.u_diff = 9.99280029
        
    def predict_one_shot(self, state_history, u_sequence, fix_cov_future):
        Nsample = u_sequence.shape[0]

        past_cov = state_history.repeat(Nsample, 1, 1).to(self.device)
        future_fix = fix_cov_future.repeat(Nsample, 1, 1).to(self.device)

        # future_fix: (Nsample, P, 3)
        # u_sequence: (Nsample, P, 1)
        future_input = torch.cat((future_fix, u_sequence), dim=2)   # (Nsample, P, 4)

        with torch.no_grad():
            outputs = self.model((past_cov, future_input, None))
            if outputs.dim() == 4:
                predicted_x = outputs.median(dim=-1).values
            else:
                predicted_x = outputs

        return predicted_x

    # def get_reward_mbd(self, tide_output, reference, u_output, constraint, args):
    #     # 1. Optimality (pJ): Tracking & Smoothness
    #     temp_pred = tide_output[:, :, 0]
    #     temp_ref = reference[:, :, 0]
    #     tracking_loss = torch.sqrt(((temp_ref - temp_pred) ** 2).mean(dim=1))
        
    #     u_diff = u_output[:, 1:, :] - u_output[:, :-1, :]
    #     smoothness_loss = torch.sqrt((u_diff ** 2).mean(dim=(1, 2)))
        
    #     total_cost = args.w_tracking * tracking_loss + args.w_smooth * smoothness_loss
    #     p_J = torch.exp(-total_cost / args.temp_sample)

    #     # 2. Constraints (pg): Depth Boundary
    #     depth_pred = tide_output[:, :, 1]
    #     up_violation = torch.relu(depth_pred - constraint[:, :, 1]) ** 2
    #     constraint_loss = torch.sqrt(up_violation.mean(dim=1))
        
    #     p_g = torch.exp(-(args.w_constraint * constraint_loss) / args.temp_sample)

    #     # 3. 최종 Target log_p0
    #     log_p0 = torch.log(p_J * p_g + 1e-9)
    #     return log_p0

    # def get_reward_mbd(self, tide_output, reference, u_output, constraint, args):
    #     # tide_output: (Nsample, P, 2)
    #     # reference:   (1, P, 1)
    #     # u_output:    (Nsample, P, 1)
    #     # constraint:  (1, P, 2)

    #     temp_pred = tide_output[:, :, 0]
    #     temp_ref = reference[:, :, 0]

    #     # 1) tracking: RMSE 대신 MSE 그대로 사용
    #     tracking_loss = ((temp_ref - temp_pred) ** 2).mean(dim=1)

    #     # 2) smoothness
    #     u_diff = u_output[:, 1:, :] - u_output[:, :-1, :]
    #     smoothness_loss = (u_diff ** 2).mean(dim=(1, 2))

    #     # 3) upper-bound constraint
    #     depth_pred = tide_output[:, :, 1]
    #     up_violation = torch.relu(depth_pred - constraint[:, :, 1]) ** 2
    #     constraint_loss = up_violation.mean(dim=1)

    #     # 4) log reward를 직접 계산
    #     log_p0 = -(
    #         args.w_tracking * tracking_loss
    #         + args.w_smooth * smoothness_loss
    #         + args.w_constraint * constraint_loss
    #     ) / args.temp_sample

    #     print("temp_pred range:", temp_pred.min().item(), temp_pred.max().item())
    #     print("temp_ref range :", temp_ref.min().item(), temp_ref.max().item())

    #     print("tracking mean/std:",
    #         tracking_loss.mean().item(),
    #         tracking_loss.std().item())

    #     print("smooth mean/std:",
    #         smoothness_loss.mean().item(),
    #         smoothness_loss.std().item())

    #     print("constraint mean/std:",
    #         constraint_loss.mean().item(),
    #         constraint_loss.std().item())

    #     return log_p0

    def get_reward_mbd(self, tide_output, reference, u_output, constraint, planning_data, args):

        # ===============================
        # 1. Tracking term
        # ===============================
        temp_pred = tide_output[:, :, 0]
        temp_ref  = reference[:, :, 0]

        temp_err = temp_pred - temp_ref

        track_under = torch.relu(-temp_err) ** 2
        track_over  = torch.relu(temp_err) ** 2

        tracking_cost = (track_under + 4.0 * track_over).mean(dim=1)

        # ===============================
        # 2. Smoothness
        # ===============================
        u_diff = u_output[:, 1:, :] - u_output[:, :-1, :]
        smooth_cost = (u_diff ** 2).mean(dim=(1,2))

        # ===============================
        # 3. First-step jump penalty
        # ===============================
        u_prev = planning_data["u_prev"].view(1)
        u0 = u_output[:, 0, 0]

        jump_cost = (u0 - u_prev) ** 2

        # ===============================
        # 4. Constraint term
        # ===============================
        depth_pred = tide_output[:, :, 1]

        violation = torch.relu(depth_pred - constraint[:, :, 1]) ** 2
        constraint_cost = violation.mean(dim=1)

        # ===============================
        # 5. Optimality probability
        # ===============================
        total_cost = (
            args.w_tracking * tracking_cost +
            args.w_smooth   * smooth_cost +
            args.w_u0       * jump_cost
        )

        p_J = torch.exp(-total_cost / args.temp_sample)

        # ===============================
        # 6. Constraint probability
        # ===============================
        p_g = torch.exp(-args.w_constraint * constraint_cost)

        # ===============================
        # 7. Final distribution
        # ===============================
        p_0 = p_J * p_g

        log_p0 = torch.log(p_0 + 1e-8)

        # ===============================
        # Debug prints
        # ===============================
        print("temp_pred range:", temp_pred.min().item(), temp_pred.max().item())
        print("temp_ref range :", temp_ref.min().item(), temp_ref.max().item())

        print("tracking mean/std:",
            tracking_cost.mean().item(),
            tracking_cost.std().item())

        print("smooth mean/std:",
            smooth_cost.mean().item(),
            smooth_cost.std().item())

        print("jump mean/std:",
            jump_cost.mean().item(),
            jump_cost.std().item())

        print("constraint mean/std:",
            constraint_cost.mean().item(),
            constraint_cost.std().item())

        return log_p0