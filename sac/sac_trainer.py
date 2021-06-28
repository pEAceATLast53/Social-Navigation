import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import numpy as np
import os, copy

from sac.modules import Actor, Q, ObsEncoder
from sac.replay_buffer import ReplayBuffer

class SAC():
    def __init__(self, args, writer):
        #super(SAC, self).__init__()

        self.args = args
        self.device = args.device

        self.obs_encoder = ObsEncoder(args).to(self.device)

        self.action_dim = args.action_dim
        self.state_dim = self.obs_encoder.state_dim
        self.args.state_dim = self.state_dim

        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.q1, self.q2 = Q(self.state_dim, self.action_dim).to(self.device), Q(self.state_dim, self.action_dim).to(self.device)

        self.target_obs_encoder = copy.deepcopy(self.obs_encoder)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)

        self.log_alpha = torch.tensor(np.log(args.init_alpha)).to(self.device).float()
        self.log_alpha.requires_grad = True

        self.replay_buffer = ReplayBuffer(args)
        
        self.actor_optimizer = optim.Adam(list(self.obs_encoder.parameters()) + list(self.actor.parameters()), lr=args.actor_learning_rate)
        self.q1_optimizer = optim.Adam(list(self.obs_encoder.parameters()) + list(self.q1.parameters()), lr=args.critic_learning_rate)
        self.q2_optimizer = optim.Adam(list(self.obs_encoder.parameters()) + list(self.q2.parameters()), lr=args.critic_learning_rate)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=args.alpha_learning_rate)

        self.writer = writer

        self.model_save_dir = './models/' + args.model_name
        if not os.path.isdir(self.model_save_dir): os.mkdir(self.model_save_dir)

        self.critic_criterion = nn.MSELoss()

    def select_action(self, state):
        mu, log_sigma = self.actor(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def update(self):
        for _ in range(self.args.gradient_steps):
            batch_a, batch_r, batch_d, batch_obs_goal, batch_obs_goal_, batch_obs_waypoints, batch_obs_waypoints_, batch_obs_lidar, batch_obs_lidar_ = \
                self.replay_buffer.sample()

            with torch.no_grad():
               target_encoded_obs = self.target_obs_encoder(batch_obs_goal_, batch_obs_waypoints_, batch_obs_lidar_)
               a_, log_prob = self.select_action(target_encoded_obs)
               entropy = -self.log_alpha.exp() * log_prob
               q1_, q2_ = self.target_q1(target_encoded_obs, a_), self.target_q2(target_encoded_obs, a_)
               q1_q2 = torch.cat([q1_, q2_], 1)
               min_q = torch.min(q1_q2, 1, keepdim=True)[0]
               target_q_value = batch_r + (1 - batch_d) * 0.99 * (min_q + entropy)

            encoded_obs = self.obs_encoder(batch_obs_goal, batch_obs_waypoints, batch_obs_lidar)

            q1_output = self.q1(encoded_obs, batch_a)
            q2_output = self.q2(encoded_obs, batch_a)

            self.loss1 = (q1_output - target_q_value) ** 2
            self.q1_optimizer.zero_grad()
            self.loss1.mean().backward(retain_graph=True)
            self.q1_optimizer.step()

            self.loss2 = (q2_output - target_q_value) ** 2
            self.q2_optimizer.zero_grad()
            self.loss2.mean().backward(retain_graph=True)
            self.q2_optimizer.step()

            action, log_prob = self.select_action(encoded_obs)
            entropy = -self.log_alpha.exp() * log_prob

            q1_q2 = torch.cat([q1_output, q2_output], dim = 1)
            min_q = torch.min(q1_q2, 1, keepdim = True)[0]

            loss_actor = -min_q - entropy
            self.actor_optimizer.zero_grad()
            loss_actor.mean().backward()
            self.actor_optimizer.step()

            self.log_alpha_optimizer.zero_grad()
            loss_alpha = -(self.log_alpha.exp() * (log_prob + self.args.target_entropy).detach()).mean()
            loss_alpha.backward()
            self.log_alpha_optimizer.step()

    def update_targets(self):
        self.target_obs_encoder.load_state_dict(self.obs_encoder.state_dict())
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

    def save_model(self, difficulty, density):
        sub_dir = os.path.join(self.model_save_dir, difficulty + '_' + density)
        if not os.path.isdir(sub_dir): os.mkdir(sub_dir)
        torch.save(self.actor.state_dict(), os.path.join(sub_dir, 'actor.pth'))
        torch.save(self.q1.state_dict(), os.path.join(sub_dir, 'q1.pth'))
        torch.save(self.q2.state_dict(), os.path.join(sub_dir, 'q2.pth'))
        torch.save(self.obs_encoder.state_dict(), os.path.join(sub_dir, 'obs_encoder.pth'))

    def load(self):
        torch.save(self.actor.state_dict(), os.path.join(self.model_save_dir, 'actor.pth'))
        torch.save(self.q1.state_dict(), os.path.join(self.model_save_dir, 'q1.pth'))
        torch.save(self.q2.state_dict(), os.path.join(self.model_save_dir, 'q2.pth'))
        torch.save(self.obs_encoder.state_dict(), os.path.join(self.model_save_dir, 'obs_encoder.pth'))

    def save_log(self, writer, update_count):
        writer.add_scalar("Q1 Loss", self.loss1.mean().item(), update_count)
        writer.add_scalar("Q2 Loss", self.loss2.mean().item(), update_count)



        