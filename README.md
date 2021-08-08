# Social-Navigation
An RL trained agent navigates towards the goal while avoiding pedestrians and static obstacles in indoor environments.
## Environment
Pedestrians are implemented by the Social Force Model. I first used ORCA, but ORCA was vulnerable to the bottleneck problem, and there was no smart way to implement groups. The social force model is free of the bottleneck problem, and I can add attractive forces between pedestrians to implement groups. The social force model code is from https://github.com/yuxiang-gao/PySocialForce.

<img src="https://user-images.githubusercontent.com/86182918/124701086-e6468800-df28-11eb-8aa7-51510ea4e4de.gif" width="300" height="500">

I obtained indoor maps from http://gibsonenv.stanford.edu/database. They provide black and white maps of real indoor environments, where black areas are static obstacles and white areas are free space. env.get_path.py can take these maps, construct graphs on the traversible areas, sample an initial point and a goal for each pedestrian, and generate the waypoints between the initial and goal points by A*. With the social force model, each pedestrian heads toward its next waypoint while avoiding other pedestrians, static obstacles and also the robot. If a pedestrian reaches the goal, it stops for a random number of time steps, and then a new path is generated. 
The robot is also included in the social force model because pedestrians should perceive the robot and avoid it. However, the robot does not move by the social force model. Instead, it is trained by RL.
## Data
I did not upload the data folder yet. It consists of the black and white traversablity maps, pre-constructed graphs of traversable areas for the A*, and obstacle points.
## Soft Actor Critic
Like the pedestrians, env.get_path.py provides the robot with the initial and goal points and generates the waypoints between them by A*. The geodesic distance between the initial and goal points is determined by the assigned difficulty. Unlike the pedestrians, the robot learns to move to the next waypoint while avoiding other pedestrians and static obstacles by Soft Actor Critic. The action, observation model, network architecture and a lot of the parameters are inspired by https://arxiv.org/pdf/2010.08600.pdf.
### Action
1. Linear velocity, clipped to range [-0.5, 1.0].
2. Angular velocity, clipped to range [-0.5, 0.5].
### Observation
1. Lidar readings
2. Relative coordinate of the next waypoint
3. Relative coordinate of the goal
### Reward
I am experimenting on different kinds of rewards. Currently, the agent receives +1 if it reaches the goal, and +0.1 every time it reaches the next waypoint. The is a -0.001 penalty per time step. If it collides with a static obstacle or a pedestrian, it receives a penalty of -1 and the episode terminates. It also receives penalty when it invades a pedestrian's personal space. I am also considering setting the waypoint reward as a function of the agent's distance to the next waypoint, instead of a sparse reward of +0.1.
## Issues
Unfortunatly, the social force model is too slow for me to successfully debug the SAC code. It calculates the forces between all the obstacle points and the agent, and the forces between all pairs of pedestrians. I think I am experimenting on environments too large. Calculating forces with only nearby entities would be practical, but that still requires calculating the distances between all the entities. Maybe I should just make the obstacle points more sparse, or cut the maps and reduce the crowd size accordingly. Any ideas? 
