import torch

from xRL.memory.replay_buffer import create_storage


class TransitionBuffer:
    def __init__(self, obs_dim, action_dim, add_obs_dim=0, num_envs=1, nstep=3,
                 device='cuda', gamma=0.99):
        self.nstep = nstep
        self.gamma = gamma
        self.has_add = bool(add_obs_dim) and add_obs_dim > 0

        storage = create_storage((num_envs, nstep), obs_dim, action_dim, add_obs_dim, device)
        self.buf_obs = storage["obs"]
        self.buf_next_obs = storage["next_obs"]
        self.buf_action = storage["action"]
        self.buf_reward = storage["reward"]
        self.buf_terminated = torch.zeros((num_envs, nstep, 1), dtype=torch.bool, device=device)
        self.buf_episode_end = torch.zeros((num_envs, nstep, 1), dtype=torch.bool, device=device)
        if self.has_add:
            self.buf_add_obs = storage["add_obs"]
            self.buf_add_next_obs = storage["add_next_obs"]

        self.step_count = 0
        self.gamma_array = torch.tensor(
            [gamma ** i for i in range(nstep)], dtype=torch.float32, device=device
        ).view(-1, 1)

    @torch.no_grad()
    def push_sequence(self, obs, actions, rewards, next_obs, terminated, episode_end,
                      add_obs=None, next_add_obs=None):
        out = []
        for t in range(obs.shape[1]):
            self.buf_obs = _shift_fifo(self.buf_obs, obs[:, t])
            self.buf_next_obs = _shift_fifo(self.buf_next_obs, next_obs[:, t])
            self.buf_action = _shift_fifo(self.buf_action, actions[:, t])
            self.buf_reward = _shift_fifo(self.buf_reward, rewards[:, t])
            self.buf_terminated = _shift_fifo(self.buf_terminated, terminated[:, t].bool())
            self.buf_episode_end = _shift_fifo(self.buf_episode_end, episode_end[:, t].bool())
            if self.has_add:
                self.buf_add_obs = _shift_fifo(self.buf_add_obs, add_obs[:, t])
                self.buf_add_next_obs = _shift_fifo(self.buf_add_next_obs, next_add_obs[:, t])

            self.step_count += 1
            if self.step_count < self.nstep:
                continue

            reward, nstep_next_obs, discount = compute_nstep_return(
                self.buf_next_obs, self.buf_terminated, self.buf_episode_end,
                self.buf_reward, self.gamma_array, self.gamma,
            )
            transition = [self.buf_obs[:, 0], self.buf_action[:, 0], reward, nstep_next_obs, discount]
            if self.has_add:
                transition += [self.buf_add_obs[:, 0], self.buf_add_next_obs[:, 0]]
            out.append(transition)

        if not out:
            return None
        return tuple(torch.cat(items, dim=0) for items in zip(*out))


def _shift_fifo(queue, new_tensor):
    return torch.cat((queue[:, 1:], new_tensor.unsqueeze(1)), dim=1)


@torch.jit.script
def compute_nstep_return(buf_next_obs, buf_terminated, buf_episode_end, buf_reward,
                         gamma_array, gamma: float):
    episode_end = buf_episode_end.squeeze(-1)
    num_envs, nstep = episode_end.shape

    # window stops at the first episode end
    first_end = episode_end.float().argmax(dim=1)
    last = torch.where(episode_end.any(dim=1), first_end, torch.full_like(first_end, nstep - 1))
    env_ids = torch.arange(num_envs, device=episode_end.device)

    next_obs = buf_next_obs[env_ids, last]
    terminated = buf_terminated[env_ids, last].float()

    mask = torch.arange(nstep, device=episode_end.device).unsqueeze(0) <= last.unsqueeze(1)
    reward = (buf_reward * gamma_array * mask.unsqueeze(-1)).sum(1)

    # time-outs still bootstrap
    discount = gamma_array[last] * gamma * (1.0 - terminated)
    return reward, next_obs, discount
