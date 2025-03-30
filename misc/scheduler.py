from diffusers import PNDMScheduler
import torch

scheduler = PNDMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule='scaled_linear',
    skip_prk_steps=True,
    steps_offset=1,
)

scheduler.set_timesteps(2)

# Hardcoded noisy sample (1x3x4x4)
noisy_sample = torch.tensor([[
    [[0.5, 0.4, 0.3, 0.2], [0.1, 0.0, -0.1, -0.2], [-0.3, -0.4, -0.5, -0.6], [-0.7, -0.8, -0.9, -1.0]],
    [[0.3, 0.25, 0.2, 0.15], [0.1, 0.05, 0.0, -0.05], [-0.1, -0.15, -0.2, -0.25], [-0.3, -0.35, -0.4, -0.45]],
    [[-0.2, -0.1, 0.0, 0.1], [0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9], [1.0, 0.9, 0.8, 0.7]]
]], dtype=torch.float32)

# Hardcoded predicted noise (1x3x4x4)
predicted_noise = torch.tensor([[
    [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
    [[0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05]],
    [[-0.05, -0.05, -0.05, -0.05], [-0.05, -0.05, -0.05, -0.05], [-0.05, -0.05, -0.05, -0.05],
     [-0.05, -0.05, -0.05, -0.05]],
]], dtype=torch.float32)

for i, timestep in enumerate(scheduler.timesteps):
    noisy_sample = scheduler.step(predicted_noise, timestep, noisy_sample).prev_sample
    print(noisy_sample)
