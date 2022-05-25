import numpy as np


# from https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu=0, sigma=0.2, size=1, theta=0.15, dt=1e-2):
        self.size = size
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x_prev = np.zeros_like(self.mu)

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = np.zeros(self.size)
