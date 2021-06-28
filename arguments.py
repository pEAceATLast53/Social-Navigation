import argparse

parser = argparse.ArgumentParser("Social Navigation")
parser.add_argument('--model_name', type=str, default='local_planner')

parser.add_argument('--mode', default='train')
parser.add_argument('--map_size', default='medium_and_big')
parser.add_argument('--crowd_density', default='low')
parser.add_argument('--difficulty', default='easy')

# Environment
parser.add_argument("--res", type=int, default=3)
parser.add_argument("--radius", type=float, default=0.3)
parser.add_argument("--personal_space", type=float, default=0.1)
parser.add_argument("--goal_threshold", type=float, default=0.5)
parser.add_argument("--max_goal_threshold", type=float, default=1.5)
parser.add_argument("--enable_group", type=bool, default=False)
parser.add_argument("--group_size_lambda", type=int, default=3)
parser.add_argument("--waypoint_interval", type=int, default=4)
parser.add_argument("--action_type", type=str, default='unicycle')
parser.add_argument("--step_width", default=0.4)
parser.add_argument("--exact_control", default=False)
parser.add_argument("--reward_func", default='sparse', choices = ['sparse', 'linear'])

parser.add_argument("--render", type=bool, default=False)

parser.add_argument("--obs_goal", type=bool, default=True)
parser.add_argument("--obs_waypoints", type=bool, default=True)
parser.add_argument("--obs_lidar", type=bool, default=True)

parser.add_argument("--num_wps_input", type=int, default=1)
parser.add_argument("--lidar_delta", type=int, default=128)
parser.add_argument("--lidar_range", type=float, default=5.0)

parser.add_argument("--device", default='cuda')
parser.add_argument("--feature_dim", type=int, default=256)
parser.add_argument("--action_dim", type=int, default=2)

parser.add_argument("--episode_len", type=int, default=300)
parser.add_argument("--num_episodes", type=int, default=10000)

parser.add_argument("--save_interval", default=100)   
parser.add_argument("--log_save_interval", default=300)
parser.add_argument("--num_test", default=5)
parser.add_argument("--update_after", default=1000)
parser.add_argument("--update_every", default=50)

#RL Parameters
parser.add_argument("--buffer_size", type=int, default = 150000)    #150000
parser.add_argument("--actor_learning_rate", type=float, default=0.0003)
parser.add_argument("--critic_learning_rate", type=float, default=0.0003)
parser.add_argument("--alpha_learning_rate", type=float, default=0.0003)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)    #128
parser.add_argument("--init_alpha", type=float, default=0.2)
parser.add_argument("--target_entropy", type=float, default=-2.0)
parser.add_argument("--target_update_interval", type=int, default=300*200)

args = parser.parse_args()