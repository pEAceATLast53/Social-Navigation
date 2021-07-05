import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from env.environment import Env
from sac.sac_trainer import SAC
from arguments import args

import cv2
import os
import torch
import collections
import random
import joblib
from tensorboardX import SummaryWriter

log_dir = './tb_logs/' + args.model_name
if not os.path.isdir(log_dir): os.mkdir(log_dir)
writer = SummaryWriter(log_dir)

if args.render:
    render_dir = './render/' + args.model_name
    if not os.path.isdir(render_dir): os.mkdir(render_dir)

buffer_save_dir = './buffer/' + args.model_name
if not os.path.isdir(buffer_save_dir): os.mkdir(buffer_save_dir)

#difficulty_curriculum
easy_to_medium = 3000
medium_to_hard = 8000
difficulty = args.difficulty

#crowd_density_curriculum
low_to_medium = 2000
medium_to_high = 10000
crowd_density = args.crowd_density

val_mean_ret = {}
for i in ['easy', 'medium', 'low']:
    for j in ['low', 'medium', 'high']:
        val_mean_ret[i+'_'+j] = -100000

env = Env(args)
trainer = SAC(args, writer)

map_list = os.listdir(env.map_dir)

map_size = args.map_size
difficulty = args.difficulty
crowd_density = args.crowd_density

map_idx = 0
update_count = 0
successes = collections.deque(maxlen=100)

episode = 1
step_count = 0

while True:
    map_name = map_list[map_idx % len(map_list)]
    map_idx += 1

    episode_return = 0
    #obs, info = env.reset(map_name)
    
    try: obs, info = env.reset(map_name)
    except AssertionError:
        print(episode, "Fail", map_name, crowd_density)
        continue
    
    obs_goal = torch.tensor(obs['goal'])
    obs_waypoints = torch.tensor(obs['waypoints'])
    obs_lidar = torch.tensor(obs['lidar'])

    frames = [info['frame']]

    while True:
        with torch.no_grad():
            encoded_obs = trainer.obs_encoder(obs_goal.to(args.device).float().unsqueeze(0), obs_waypoints.to(args.device).float().unsqueeze(0), obs_lidar.to(args.device).float().unsqueeze(0))
            action, _ = trainer.select_action(encoded_obs)   

        action_clipped = []
        if args.action_type == 'unicycle':
            action_clipped.append(max(-0.2, min(action[0,0].item(), 1.)))
            action_clipped.append(max(-0.5, min(action[0,1].item(), 0.5)))
        obs, rew, done, info = env.step(action_clipped)
        step_count += 1

        terminate = done
        is_goal = info['is_goal']
        frames.append(info['frame'])
        episode_return += rew

        obs_goal_ = torch.tensor(obs['goal'])
        obs_waypoints_ = torch.tensor(obs['waypoints'])
        obs_lidar_ = torch.tensor(obs['lidar'])
        rew = torch.tensor([rew])
        done = torch.tensor([done])

        trainer.replay_buffer.store(action, rew, done, obs_goal, obs_waypoints, obs_lidar, obs_goal_, obs_waypoints_, obs_lidar_)

        obs_goal = obs_goal_
        obs_waypoints = obs_waypoints_
        obs_lidar = obs_lidar_

        if trainer.replay_buffer.length >= args.update_after and step_count % args.update_every == 0:
            trainer.update()
            update_count += 1
            if update_count % args.target_update_interval == 0:
                trainer.update_targets()
            if update_count % args.log_save_interval == 0:
                trainer.save_log(writer, update_count)
            if update_count % (args.buffer_size // 2) == 0:
                joblib.dump(trainer.replay_buffer, os.path.join(buffer_save_dir, str(update_count)+'.dat.gz'))

        if terminate: break

    successes.append(is_goal)

    if episode % args.save_interval == 0:
        env.set_map_dir(args.map_size, 'val')
        val_map_names = random.sample(os.listdir(env.map_dir), args.num_test)
        val_returns = []
        for val_map_name in val_map_names:
            try: obs, info = env.reset(val_map_name)
            except AssertionError: continue

            obs_goal = torch.tensor(obs['goal'])
            obs_waypoints = torch.tensor(obs['waypoints'])
            obs_lidar = torch.tensor(obs['lidar'])
            val_episode_return = 0
            while True:
                with torch.no_grad():
                    encoded_obs = trainer.obs_encoder(obs_goal.to(args.device).float().unsqueeze(0), obs_waypoints.to(args.device).float().unsqueeze(0), obs_lidar.to(args.device).float().unsqueeze(0))
                    action, _ = trainer.select_action(encoded_obs)   

                action_clipped = []
                if args.action_type == 'unicycle':
                    action_clipped.append(max(-0.2, min(action[0,0].item(), 1.)))
                    action_clipped.append(max(-0.5, min(action[0,1].item(), 0.5)))
                obs, rew, done, info = env.step(action_clipped)

                val_episode_return += rew
                if done: break
                obs_goal = torch.tensor(obs['goal'])
                obs_waypoints = torch.tensor(obs['waypoints'])
                obs_lidar = torch.tensor(obs['lidar'])

            val_returns.append(val_episode_return)

        if len(val_returns) > 0:
            curr_mean = sum(val_returns) / len(val_returns)
            if val_mean_ret[difficulty+'_'+crowd_density] < curr_mean:
                val_mean_ret[difficulty+'_'+crowd_density] = curr_mean
                trainer.save_model(difficulty, crowd_density)
                print("Saved model!", difficulty, crowd_density)

        env.set_map_dir(args.map_size, 'train')

    if episode == easy_to_medium: 
        env.set_difficulty('medium')
        difficulty = 'medium'
    if episode == medium_to_hard: 
        env.set_difficulty('hard')
        difficulty = 'hard'
    if episode == low_to_medium: 
        env.set_crowd_density('medium')
        crowd_density = 'medium'
    if episode == medium_to_high: 
        env.set_crowd_density('high')
        crowd_density = 'high'

    writer.add_scalar('Episode Return', episode_return, episode)
    if len(successes) == 100:
        writer.add_scalar('Success Rate', sum(successes) / 100, episode)

    num_wp_success = env.robot_next_wp_id - 1
    if num_wp_success < 0: num_wp_success = len(env.robot_waypoints) - 1
    writer.add_scalar("# of Waypoint Success", num_wp_success, episode)

    print("Episode : ", episode, ", Return : ", episode_return, ", Success : ", is_goal, ", # of Waypoint Success : ", num_wp_success, ", Duration : ", env.time_t, ", Difficulty : ", difficulty, ", Crowd Density", crowd_density)

    if args.render:
        out = cv2.VideoWriter(os.path.join(render_dir, str(episode)+'.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 15, (frames[0].shape[0], frames[0].shape[1]))
        for aa in frames: out.write(aa)
        out.release()

    episode += 1
