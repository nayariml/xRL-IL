import torch

# Implementation of ring buffer for trajectories


def create_storage(capacity, obs_dim, action_dim, add_obs_dim=0, device='cuda'):
    if isinstance(capacity, int):
        capacity = (capacity,)

    obs_shape = (*capacity, obs_dim) if isinstance(obs_dim, int) else (*capacity, *obs_dim)
    obs = torch.zeros(obs_shape, dtype=torch.float32, device=device)

    storage = {
        "obs": obs,
        "next_obs": torch.zeros_like(obs),
        "action": torch.zeros((*capacity, int(action_dim)), dtype=torch.float32, device=device),
        "reward": torch.zeros((*capacity, 1), dtype=torch.float32, device=device),
        "discount": torch.zeros((*capacity, 1), dtype=torch.float32, device=device),
    }

    if add_obs_dim and add_obs_dim > 0:
        add_shape = (*capacity, add_obs_dim) if isinstance(add_obs_dim, int) else (*capacity, *add_obs_dim)
        storage["add_obs"] = torch.zeros(add_shape, dtype=torch.float32, device=device)
        storage["add_next_obs"] = torch.zeros(add_shape, dtype=torch.float32, device=device)

    return storage


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, add_obs_dim=0, device='cpu'):
        self.device = device
        self.capacity = int(capacity)
        self.action_dim = int(action_dim)
        self.obs_dim = (obs_dim,) if isinstance(obs_dim, int) else tuple(obs_dim)
        self.has_add = bool(add_obs_dim) and add_obs_dim > 0
        self.add_obs_dim = (add_obs_dim,) if isinstance(add_obs_dim, int) else tuple(add_obs_dim or ())

        self.ptr = 0
        self.size = 0

        storage = create_storage(self.capacity, obs_dim, action_dim, add_obs_dim, device)
        self.keys = ["obs", "action", "reward", "next_obs", "discount"]
        if self.has_add:
            self.keys += ["add_obs", "add_next_obs"]
        self.buffers = [storage[k] for k in self.keys]

    @torch.no_grad()
    def add(self, transition):
        if len(transition) != len(self.keys):
            raise ValueError(f"Transition must have {len(self.keys)} elements, got {len(transition)}")

        num = transition[2].numel()
        data = [x.reshape(num, *buf.shape[1:]).to(self.device) for x, buf in zip(transition, self.buffers)]

        start = self.ptr
        end = start + num
        if end <= self.capacity:
            for buf, x in zip(self.buffers, data):
                buf[start:end] = x
        else:
            first = self.capacity - start
            for buf, x in zip(self.buffers, data):
                buf[start:] = x[:first]
                buf[:end - self.capacity] = x[first:]

        self.ptr = end % self.capacity
        self.size = min(self.capacity, self.size + num)

    @torch.no_grad()
    def sample(self, batch_size, device=None):
        if self.size == 0:
            raise ValueError("Replay buffer is empty")
        device = device or self.device
        indices = torch.randint(self.size, size=(batch_size,), device=self.device)
        return tuple(buf[indices].to(device) for buf in self.buffers)
