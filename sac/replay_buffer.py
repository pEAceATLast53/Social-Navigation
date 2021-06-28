import collections, random
import torch

class ReplayBuffer():
    def __init__(self, args):
        self.buffer_a = collections.deque(maxlen = args.buffer_size)
        self.buffer_r = collections.deque(maxlen = args.buffer_size)
        self.buffer_d = collections.deque(maxlen = args.buffer_size)
        self.buffer_obs_goal = collections.deque(maxlen = args.buffer_size)
        self.buffer_obs_waypoints = collections.deque(maxlen = args.buffer_size)
        self.buffer_obs_lidar = collections.deque(maxlen = args.buffer_size)
        self.buffer_obs_goal_ = collections.deque(maxlen = args.buffer_size)
        self.buffer_obs_waypoints_ = collections.deque(maxlen = args.buffer_size)
        self.buffer_obs_lidar_ = collections.deque(maxlen = args.buffer_size)
        self.batch_size = args.batch_size
        self.device = args.device
        self.length = 0

    def store(self, a, r, d, obs_goal, obs_waypoints, obs_lidar, obs_goal_, obs_waypoints_, obs_lidar_):
        self.buffer_a.append(a)
        self.buffer_r.append(r)
        self.buffer_d.append(d)
        self.buffer_obs_goal.append(obs_goal)
        self.buffer_obs_waypoints.append(obs_waypoints)
        self.buffer_obs_lidar.append(obs_lidar)
        self.buffer_obs_goal_.append(obs_goal_)
        self.buffer_obs_waypoints_.append(obs_waypoints_)
        self.buffer_obs_lidar_.append(obs_lidar_)
        self.length = len(self.buffer_a)

    def sample(self):
        batch_idx = random.sample(range(self.length), self.batch_size)

        a_list = [self.buffer_a[idx] for idx in batch_idx]
        r_list = [self.buffer_r[idx] for idx in batch_idx]
        d_list = [self.buffer_d[idx] for idx in batch_idx]
        obs_goal_list = [self.buffer_obs_goal[idx] for idx in batch_idx]
        obs_goal_list_ = [self.buffer_obs_goal_[idx] for idx in batch_idx]
        obs_waypoints_list = [self.buffer_obs_waypoints[idx] for idx in batch_idx]
        obs_waypoints_list_ = [self.buffer_obs_waypoints_[idx] for idx in batch_idx]
        obs_lidar_list = [self.buffer_obs_lidar[idx] for idx in batch_idx]
        obs_lidar_list_ = [self.buffer_obs_lidar_[idx] for idx in batch_idx]

        return torch.stack(a_list).to(self.device).float(), torch.stack(r_list).to(self.device).float(), \
            torch.stack(d_list).to(self.device).float(), torch.stack(obs_goal_list).to(self.device).float(), \
            torch.stack(obs_goal_list_).to(self.device).float(), \
            torch.stack(obs_waypoints_list).to(self.device).float(), \
            torch.stack(obs_waypoints_list_).to(self.device).float(), \
            torch.stack(obs_lidar_list).to(self.device).float(), \
            torch.stack(obs_lidar_list_).to(self.device).float()