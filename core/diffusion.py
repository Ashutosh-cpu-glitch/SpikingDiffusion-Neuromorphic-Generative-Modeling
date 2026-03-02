import torch

class SpikingDiffusion:
    def __init__(self, timesteps=50, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = timesteps
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0, t, noise):
        """
        Forward diffusion:
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        """
        sqrt_ab = torch.sqrt(self.alpha_bar[t])[:, None]
        sqrt_1ab = torch.sqrt(1 - self.alpha_bar[t])[:, None]
        return sqrt_ab * x0 + sqrt_1ab * noise

    def poisson_noise(self, x):
        """
        Generate Poisson-like noise (spike flavored)
        """
        rate = torch.clamp(x, 0.01, 1.0)
        return torch.poisson(rate)
