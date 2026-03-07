import torch
# Note: brax.io.html is part of the JAX-based Brax library.
# If you are migrating the entire environment to PyTorch, you might need a different renderer.
# Here we keep the import to maintain functionality for Brax-based pipeline_states.
try:
    from brax.io import html
except ImportError:
    html = None


# evaluate the diffused uss
def eval_us(step_env, state, us):
    # In JAX, this used jax.lax.scan. In PyTorch, we use a loop for sequential rollout.
    # us shape: (..., H, Nu)
    H = us.shape[-2]
    rews = []
    curr_state = state
    
    for i in range(H):
        # Extract the action for the current timestep
        u = us[..., i, :]
        curr_state = step_env(curr_state, u)
        rews.append(curr_state.reward)
        
    # Stack rewards along the time dimension to match (..., H)
    return torch.stack(rews, dim=-1)

def rollout_us(step_env, state, us):
    # In JAX, this used jax.lax.scan. In PyTorch, we use a loop.
    # us shape: (..., H, Nu)
    H = us.shape[-2]
    rews = []
    pipeline_states = []
    curr_state = state
    
    for i in range(H):
        u = us[..., i, :]
        curr_state = step_env(curr_state, u)
        rews.append(curr_state.reward)
        pipeline_states.append(curr_state.pipeline_state)
        
    # Returns rewards: (..., H) and pipeline_states: (..., H, state_dim)
    return torch.stack(rews, dim=-1), torch.stack(pipeline_states, dim=-2)


def render_us(step_env, sys, state, us):
    rollout = []
    rew_sum = 0.0
    Hsample = us.shape[0]
    curr_state = state
    
    for i in range(Hsample):
        # Collect pipeline_state for Brax renderer
        rollout.append(curr_state.pipeline_state)
        curr_state = step_env(curr_state, us[i])
        rew_sum += curr_state.reward
        
    # Note: brax.io.html.render typically expects JAX-based objects.
    # If pipeline_states are now PyTorch tensors, they might need conversion
    # back to JAX/NumPy depending on the 'sys' implementation.
    if html is not None:
        return html.render(sys, rollout)
    return "Html rendering skipped: brax.io.html not found."