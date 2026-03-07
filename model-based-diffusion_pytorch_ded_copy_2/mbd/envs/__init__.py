# mbd/envs/__init__.py

from .tide_env import TiDEEngineeringEnv

def get_env(env_name, weight_path, device="cuda"):
    if env_name == "tide_env":
        return TiDEEngineeringEnv(weight_path, device)
    else:
        raise ValueError(f"Unknown environment: {env_name}")