import numpy as np
import cv2
import time
import joblib
import os
import random
import copy
import math

import env.pysocialforce as psf
from env.get_path import PathGenerator

from pytictoc import TicToc

t = TicToc()

class Env:
    def __init__(self, args):    
        self.interval = args.waypoint_interval
        self.goal_threshold = args.goal_threshold
        self.max_goal_threshold = args.max_goal_threshold

        self.radius = args.radius
        self.personal_space = args.personal_space
        self.res = args.res
        self.scale = self.res * 0.01
        self.inv_scale = 100. / self.res

        self.enable_group = args.enable_group
        self.group_size_lambda = args.group_size_lambda

        self.timestep = args.step_width
        self.episode_len = args.episode_len

        self.set_crowd_density(args.crowd_density)
        self.set_difficulty(args.difficulty)

        self.action_type = args.action_type
        self.lidar_range_pixel = int(args.lidar_range * self.inv_scale)
        self.lidar_range = args.lidar_range
        self.lidar_delta = args.lidar_delta

        self.set_map_dir(args.map_size, args.mode)
        self.graph_dir = 'data/graphs_3_30'
        self.obstacle_dir = 'data/obstacles_3'

        self.obs_goal = args.obs_goal
        self.obs_waypoints = args.obs_waypoints
        self.obs_lidar = args.obs_lidar
        self.num_wps_input = args.num_wps_input

        self.render = args.render

        self.exact_control = args.exact_control

        self.reward_func = args.reward_func

        self.path_generator = PathGenerator(self.radius, self.res)

    def set_map_dir(self, map_size, mode):
        self.map_size = map_size
        self.mode = mode
        self.map_dir = os.path.join('data/maps_3_30/', self.map_size, self.mode)

        self.ped_target_dist_min = 15.0
        self.ped_target_dist_max = 50.0
        #if map_size == 'big':
        #    self.ped_target_dist_min = 25.0
        #    self.ped_target_dist_max = 50.0

    def set_crowd_density(self, type):
        if type == 'low': self.crowd_density = 0.003
        if type == 'medium': self.crowd_density = 0.005
        if type == 'high': self.crowd_density = 0.007

    def set_difficulty(self, type):
        if type == 'easy':
            self.robot_target_dist_min = 3.0
            self.robot_target_dist_max = 6.0
        if type == 'medium':
            self.robot_target_dist_min = 6.0
            self.robot_target_dist_max = 9.0
        if type == 'difficult':
            self.robot_target_dist_min = 9.0
            self.robot_target_dist_max = 15.0

    def reset(self, map_name = None):
        if map_name is not None:
            self.map_name = map_name
            self.map_img = cv2.imread(os.path.join(self.map_dir, map_name), cv2.IMREAD_GRAYSCALE)
            if self.render: self.map_render = cv2.cvtColor(self.map_img.transpose(1,0), cv2.COLOR_GRAY2RGB)

            self.path_generator.g = joblib.load(os.path.join(self.graph_dir, self.map_name.replace('png', 'dat.gz')))

            obstacles = joblib.load(os.path.join(self.obstacle_dir, self.map_name.replace('png', 'dat.gz')))
        
            self.num_ped = max(int(len(self.path_generator.g.nodes) * self.crowd_density), 5)
            self.groups = []
            if self.enable_group:
                group_sizes = np.clip(np.random.poisson(self.group_size_lambda, self.num_ped // self.group_size_lambda), 1, 5)
                self.num_ped = 0
                last_id = 0
                for gs in group_sizes:
                    self.groups.append(range(last_id, last_id+gs))
                    last_id += gs
                    self.num_ped += gs
            else: self.groups = [[i] for i in range(self.num_ped)]
        
        else: assert self.map_name is not None

        paths_dense, _ = self.path_generator.get_random_path(num_ped = len(self.groups), target_dist_min=self.ped_target_dist_min, target_dist_max=self.ped_target_dist_max)
        self.paths = []
        for pt in paths_dense: 
            if len(pt) < self.interval + 1: self.paths.append(np.array([pt[0], pt[-1]]))
            else: self.paths.append(pt[::self.interval])

        speeds = np.clip(np.random.normal(1.34, 0.26, self.num_ped), 1.0, 1.8)
        angles = np.random.uniform(0.0, 2*np.pi, self.num_ped)
        self.next_wp_ids = [1] * len(self.groups)
        self.counts = [0] * len(self.groups)
        self.stop_times = [0] * len(self.groups)

        initial_states = []
        ped_init_points = []
        for group_id, group in enumerate(self.groups):
            for ped_id in group:
                initial_state = np.array([self.paths[group_id][0][0]*self.scale, self.paths[group_id][0][1]*self.scale, speeds[ped_id]*np.cos(angles[ped_id]), \
                    speeds[ped_id]*np.sin(angles[ped_id]), self.paths[group_id][1][0]*self.scale, self.paths[group_id][1][1]*self.scale])
                initial_states.append(initial_state)
                ped_init_points.append([self.paths[group_id][0][0], self.paths[group_id][0][1]])

        self.robot_waypoints = []
        path_dense, path_dist = self.path_generator.get_random_path_robot(personal_space=self.personal_space, ped_init_points=ped_init_points, target_dist_min=self.robot_target_dist_min, target_dist_max=self.robot_target_dist_max)
        assert self.robot_target_dist_min <= path_dist <= self.robot_target_dist_max

        for point in path_dense[::self.interval]:
            self.robot_waypoints.append([point[0]*self.scale, point[1]*self.scale])
        self.robot_next_wp_id = 1
        self.robot_pos = [self.robot_waypoints[0][0], self.robot_waypoints[0][1]]
        self.robot_vel = [0.0, 0.0]
        self.robot_orn = math.atan2(self.robot_waypoints[1][1] - self.robot_waypoints[0][1], self.robot_waypoints[1][0] - self.robot_waypoints[0][0]) \
            + np.random.uniform(-np.pi/4, np.pi/4, None)

        initial_states.append([self.robot_waypoints[0][0], self.robot_waypoints[0][1], self.robot_vel[0], self.robot_vel[1], \
            self.robot_waypoints[0][0], self.robot_waypoints[0][1]])
        initial_states = np.array(initial_states)                

        self.sim = psf.Simulator(initial_states, radius=self.radius, step_width=self.timestep, groups=self.groups + [[self.num_ped]], obstacles=obstacles, config_file = 'env/pysocialforce/config/default.toml')

        self.obstacles = np.concatenate(copy.deepcopy(self.sim.env.obstacles), 0)
        self.obstacles = list(set([tuple(self.obstacles[i,:]) for i in range(self.obstacles.shape[0])]))

        self.time_t = 0
        info = {'frame':None, 'is_goal':False}
        if self.render:
            frame = self.save_render()
            info['frame'] = frame

        self.local_obstacles = []
        self.ref_point = [self.robot_pos[0], self.robot_pos[1]]

        return self.observation(), info

    def step(self, action):
        self.waypoint_update = False
        self.set_action(action)       
        self.sim.step(1)
        states, _ = self.sim.get_states()
        if self.exact_control: self.update_robot_state_exact(action)
        else: self.update_robot_state(action)
        new_states = self.update_ped_state(states)

        new_states[-1, :6] = np.array([self.robot_pos[0], self.robot_pos[1], self.robot_vel[0], self.robot_vel[1], \
            self.robot_pos[0] + self.robot_vel[0]*self.timestep, self.robot_pos[1] + self.robot_vel[1]*self.timestep])
        self.sim.peds.update(new_states, self.groups) 
        self.time_t += 1

        info = {'frame':None, 'is_goal':False}
        if self.robot_next_wp_id == -1: info['is_goal'] = True
        if self.render:
            frame = self.save_render()
            info['frame'] = frame

        return self.observation(), self.reward(), self.done(), info

    def set_action(self, action):
        if self.action_type == 'unicycle':
            self.robot_vel[0] = action[0] * np.cos(self.robot_orn)
            self.robot_vel[1] = action[0] * np.sin(self.robot_orn)
        if self.action_type == 'holonomic':
            self.robot_vel[0] = action[0]
            self.robot_vel[1] = action[1]

        curr_states, _ = self.sim.get_states()
        curr_states[-1, :6] = np.array([self.robot_pos[0], self.robot_pos[1], self.robot_vel[0], self.robot_vel[1], \
            self.robot_pos[0] + self.robot_vel[0]*self.timestep, self.robot_pos[1] + self.robot_vel[1]*self.timestep])

        self.sim.peds.update(curr_states, self.groups)

    def update_robot_state(self, action):
        if self.action_type == 'unicycle':
            self.robot_orn += action[1] * self.timestep
            self.robot_pos[0] += self.robot_vel[0] * self.timestep
            self.robot_pos[1] += self.robot_vel[1] * self.timestep
        if self.action_type == 'holonomic':
            self.robot_orn = math.atan2(self.robot_vel[1], self.robot_vel[0])
            self.robot_pos[0] += self.robot_vel[0] * self.timestep
            self.robot_pos[1] += self.robot_vel[1] * self.timestep
        if np.linalg.norm(np.array(self.robot_pos) - np.array(self.robot_waypoints[self.robot_next_wp_id])) < self.goal_threshold:
            self.waypoint_update = True
            if self.robot_next_wp_id == len(self.robot_waypoints) -1: self.robot_next_wp_id = -1
            else: self.robot_next_wp_id += 1

    def update_robot_state_exact(self, action):
        if self.action_type == 'unicycle':
            self.robot_orn += action[1] * self.timestep
            disp_local_frame = np.array([(action[0]/action[1])*np.sin(action[1]*self.timestep), (action[0]/action[1])*(1-np.cos(action[1]*self.timestep))])
            rot_mat = np.array([[np.cos(self.robot_orn), -np.sin(self.robot_orn)], [np.sin(self.robot_orn), np.cos(self.robot_orn)]])
            disp_global_frame = np.matmul(rot_mat, disp_local_frame)
            self.robot_pos[0] += disp_global_frame[0]
            self.robot_pos[1] += disp_global_frame[1]
        if np.linalg.norm(np.array(self.robot_pos) - np.array(self.robot_waypoints[self.robot_next_wp_id])) < self.goal_threshold:
            self.waypoint_update=True
            if self.robot_next_wp_id == len(self.robot_waypoints) -1: self.robot_next_wp_id = -1
            else: self.robot_next_wp_id += 1

    def update_ped_state(self, states):
        new_states = copy.deepcopy(states)
        for group_id, group in enumerate(self.groups):
            group_state = states[group, :]
            positions = group_state[:, :2]
            center = np.mean(positions, 0)
            goal = group_state[0, 4:6]
            distances = [np.linalg.norm(positions[i, :] - goal) for i in range(len(group))]
            
            if self.counts[group_id] == 0:
                if np.linalg.norm(center - goal) < self.goal_threshold and max(distances) < self.max_goal_threshold:
                    if self.next_wp_ids[group_id] + 1 >= len(self.paths[group_id]):
                        self.counts[group_id] += 1
                        self.stop_times[group_id] = random.randint(4, 20)
                    else:
                        self.next_wp_ids[group_id] += 1
                        new_states[group, 4] = self.paths[group_id][self.next_wp_ids[group_id]][0] * self.scale
                        new_states[group, 5] = self.paths[group_id][self.next_wp_ids[group_id]][1] * self.scale
            elif self.counts[group_id] == self.stop_times[group_id]:
                pt_dense, _ = self.path_generator.get_random_path_from_src(source = (int(center[0]*self.inv_scale), int(center[1]*100/self.inv_scale)), \
                    target_dist_min=self.ped_target_dist_min, target_dist_max=self.ped_target_dist_max)
                if len(pt_dense) < self.interval + 1: self.paths[group_id] = np.array([pt_dense[0], pt_dense[-1]])
                else: self.paths[group_id] = pt_dense[::self.interval]
                self.next_wp_ids[group_id] = 1
                self.counts[group_id] = 0
                new_states[group, 4] = self.paths[group_id][1][0] * self.scale
                new_states[group, 5] = self.paths[group_id][1][1] * self.scale
            else:
                self.counts[group_id] += 1  
        return new_states

    def reward(self):
        rew = -0.001
        if self.robot_next_wp_id == -1: rew += 1
        if min(self.lidar_obstacle) < self.radius or min(self.lidar_ped) < self.radius: rew -= 1
        if self.radius <= min(self.lidar_ped) < self.radius + self.personal_space: rew += (min(self.lidar_ped) - (self.radius + self.personal_space)) / self.personal_space
        if self.reward_func == 'linear':
            if self.robot_next_wp_id > 0: 
                norm_factor = np.linalg.norm(np.array(self.robot_waypoints[self.robot_next_wp_id]) - np.array(self.robot_waypoints[self.robot_next_wp_id-1]))
                rew += (1 - np.linalg.norm(np.array(self.robot_pos) - np.array(self.robot_waypoints[self.robot_next_wp_id])) / norm_factor) * 0.001
        if self.reward_func == 'sparse':
            if self.waypoint_update: rew += 0.1
        return rew

    def done(self):
        if np.min(self.lidar_obstacle) < self.radius or np.min(self.lidar_ped) < self.radius or self.robot_next_wp_id == -1 \
            or self.episode_len == self.time_t: return 1
        return 0

    def observation(self):
        obs_dict = {}
        relative_wps = np.array(self.robot_waypoints)
        relative_wps[:,0] -= self.robot_pos[0]
        relative_wps[:,1] -= self.robot_pos[1]
        rot_mat = np.array([[np.cos(self.robot_orn), np.sin(self.robot_orn)], [-np.sin(self.robot_orn), np.cos(self.robot_orn)]])
        relative_wps = np.matmul(rot_mat, relative_wps.transpose(1,0)).transpose(1,0)

        if self.obs_goal: obs_dict['goal'] = relative_wps[-1,:]
        if self.obs_waypoints:
            if self.robot_next_wp_id == -1: obs_wps = relative_wps[-1:,:]
            else: obs_wps = relative_wps[self.robot_next_wp_id:,:]
            while obs_wps.shape[0] < self.num_wps_input:
                obs_wps = np.concatenate([obs_wps, obs_wps[-1:, :]], 0)
            obs_wps = obs_wps[:self.num_wps_input, :]
            obs_dict['waypoints'] = obs_wps.reshape((-1,))
        if self.obs_lidar:
            obs_dict['lidar'] = self.lidar()
        return obs_dict

    def lidar(self):
        self.lidar_obstacle = np.full((self.lidar_delta,), self.lidar_range, dtype=np.float64)
        self.lidar_ped = np.full((self.lidar_delta,), self.lidar_range, dtype=np.float64)

        if not self.local_obstacles or np.linalg.norm(np.array(self.robot_pos) - np.array(self.ref_point)) > 0.5:
            self.ref_point = [self.robot_pos[0], self.robot_pos[1]]
            self.local_obstacles = []
            for coord in self.obstacles:
                dh = coord[0] - self.robot_pos[0]
                dw = coord[1] - self.robot_pos[1]
                dist = np.linalg.norm(np.array([dh,dw]))
                if dist > self.lidar_range: continue
                self.local_obstacles.append(coord)
                th = math.atan2(dw, dh) - self.robot_orn
                while th < 0: th += 2*np.pi
                while th >= 2*np.pi: th -= 2*np.pi
                th_index = int((th/(2 * np.pi))*self.lidar_delta)
                if th_index == self.lidar_delta:th_index = 0
                if self.lidar_obstacle[th_index] > dist: self.lidar_obstacle[th_index] = dist
        else:
            for coord in self.local_obstacles:
                dh = coord[0] - self.robot_pos[0]
                dw = coord[1] - self.robot_pos[1]
                dist = np.linalg.norm(np.array([dh,dw]))
                if dist > self.lidar_range: continue
                th = math.atan2(dw, dh) - self.robot_orn
                while th < 0: th += 2*np.pi
                while th >= 2*np.pi: th -= 2*np.pi
                th_index = int((th/(2 * np.pi))*self.lidar_delta)
                if th_index == self.lidar_delta:th_index = 0
                if self.lidar_obstacle[th_index] > dist: self.lidar_obstacle[th_index] = dist            

        ped_states, _ = self.sim.get_states()
        ped_states = ped_states[:-1,:2]
        for i in range(ped_states.shape[0]):
            dh = ped_states[i,0] - self.robot_pos[0]
            dw = ped_states[i,1] - self.robot_pos[1]
            th = math.atan2(dw, dh) - self.robot_orn
            while th < 0: th += 2*np.pi
            while th >= 2*np.pi: th -= 2*np.pi
            th_index = int((th /(2 * np.pi))*self.lidar_delta)
            if th_index == self.lidar_delta:
                th_index = 0
            if self.lidar_ped[th_index] > np.linalg.norm(np.array([dh,dw])): self.lidar_ped[th_index] = np.linalg.norm(np.array([dh,dw]))

        self.lidar_ped -= self.radius
        self.lidar_ped = np.maximum(self.lidar_ped, np.zeros(self.lidar_delta))

        return np.minimum(self.lidar_obstacle, self.lidar_ped)

    def save_render(self):
        map_render = copy.deepcopy(self.map_render)
        states, _ = self.sim.get_states()
        for i in range(states.shape[0]-1):
            cv2.circle(map_render, (int(states[i,0] * self.inv_scale), int(states[i,1] * self.inv_scale)), int(self.radius*self.inv_scale), (0,255,0), -1)
        cv2.circle(map_render, (int(states[-1,0] * self.inv_scale), int(states[-1,1] * self.inv_scale)), int(self.radius*self.inv_scale), (0,0,255), -1)

        for wp in self.robot_waypoints:
            cv2.circle(map_render, (int(wp[0] * self.inv_scale), int(wp[1] * self.inv_scale)), int(self.radius*self.inv_scale*0.5), (0,255,255), -1)
        return map_render


