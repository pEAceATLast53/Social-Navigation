# Social-Navigation
An RL trained agent navigates towards the goal while avoiding pedestrians and static obstacles in indoor environments.
## Environment
Pedestrians are implemented by the Social Force Model. I first used ORCA, but ORCA was vulnerable to the bottleneck problem, and there was no smart way to implement groups. The social force model is free of the bottleneck problem, and I can add attractive forces between pedestrians to implement groups. The social force model code is from https://github.com/yuxiang-gao/PySocialForce.

<img src="https://user-images.githubusercontent.com/86182918/124701086-e6468800-df28-11eb-8aa7-51510ea4e4de.gif" width="300" height="500">

I obtained indoor maps from http://gibsonenv.stanford.edu/database. They provide black and white maps of real indoor environments, where black areas are static obstacles and white areas are free space. env.get_path.py can take these maps, construct graphs on the traversible areas, sample an initial point and a goal for each pedestrian, and generate the waypoints between the initial and goal points by A*. With the social force model, each pedestrian heads toward its next waypoint while avoiding other pedestrians, static obstacles and also the robot. If a pedestrian reaches the goal, it stops for a random number of time steps, and then a new path is generated. 
The robot is also included in the social force model because pedestrians should perceive the robot and avoid it. However, the robot does not move by the social force model. Instead, it is trained by RL.
## Soft Actor Critic
Like the pedestrians, env.get_path.py provides the robot with the initial and goal points and generates the waypoints between them by A*. The geodesic distance between the initial and goal points is determined by the assigned difficulty. Unlike the pedestrians, the robot learns to move to the next waypoint while avoiding other pedestrians and static obstacles by Soft Actor Critic.
