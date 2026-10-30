import math
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd


@torch.no_grad()
def soft_update(target_net, current_net, tau: float):
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
        tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


def autocast_if_cuda(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_device(device_type, device_id):
    if torch.cuda.is_available() and device_id < torch.cuda.device_count():
        return f"{device_type}:{device_id}"
    return "cpu"


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        super().__init__(self.base_dist, [TanhTransform()])

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), device='cuda'):
        self.device = device
        self.mean = torch.zeros(shape, device=self.device)
        self.var = torch.ones(shape, device=self.device)
        self.epsilon = epsilon
        self.count = epsilon

    def update(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=self.device)
        self.update_from_moments(x.mean(dim=0), x.var(dim=0), x.shape[0])

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = m_2 / tot_count
        self.count = tot_count

    def get_states(self):
        return self.mean, self.var, self.count

    def load_state_dict(self, info):
        self.mean, self.var, self.count = info


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input):
        return torch.clamp(
            (input - self.mean) / torch.sqrt(self.var + self.epsilon),
            min=-self.clip_obs, max=self.clip_obs)

    def normalize_torch(self, input, device):
        mean = self.mean.detach().to(dtype=torch.float32, device=device)
        std = torch.sqrt(self.var + self.epsilon).detach().to(dtype=torch.float32, device=device)
        return torch.clamp((input - mean) / std, -self.clip_obs, self.clip_obs)


class TaskRewardNormalizer:
    """Wraps Normalizer to produce task reward in (0, 1)"""

    def __init__(self, epsilon: float = 1e-4, clip: float = 5.0):
        self._normalizer = Normalizer(input_dim=1, epsilon=epsilon, clip_obs=clip)

    def update(self, task_reward: torch.Tensor) -> None:
        self._normalizer.update(task_reward.detach().reshape(-1, 1).cpu().numpy())

    def normalize(self, task_reward: torch.Tensor) -> torch.Tensor:
        z = self._normalizer.normalize_torch(task_reward.reshape(-1, 1), device=task_reward.device)
        return torch.sigmoid(z).reshape(task_reward.shape)
