import torch
import torch.nn as nn

class BasicConv1D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv1D, self).__init__()
        BatchNorm = nn.BatchNorm1d
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = BatchNorm(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ObsEncoder(nn.Module):
    def __init__(self, args):
        super(ObsEncoder, self).__init__()
        self.state_dim = 0
        if args.obs_goal:
            self.goal_encoder = GoalEncoder(args)
            self.state_dim += args.feature_dim
        if args.obs_waypoints:
            self.waypoints_encoder = WayPointsEncoder(args)
            self.state_dim += args.feature_dim
        if args.obs_lidar:
            self.lidar_encoder = LidarEncoder(args)
            self.state_dim += args.feature_dim

    def forward(self, x_goal=None, x_waypoints=None, x_lidar=None):
        state = []
        if x_goal is not None:
            state.append(self.goal_encoder(x_goal))
        if x_waypoints is not None:
            state.append(self.waypoints_encoder(x_waypoints))
        if x_lidar is not None:
            state.append(self.lidar_encoder(x_lidar))
        return torch.cat(state, -1)

class GoalEncoder(nn.Module):
    def __init__(self, args):
        super(GoalEncoder, self).__init__()
        self.fc = nn.Linear(2, args.feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class WayPointsEncoder(nn.Module):
    def __init__(self, args):
        super(WayPointsEncoder, self).__init__()        
        self.fc = nn.Linear(2*args.num_wps_input, args.feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class LidarEncoder(nn.Module):
    def __init__(self, args):
        super(LidarEncoder, self).__init__()
        self.conv1_1d = BasicConv1D(1, 2, kernel_size = 3, padding = 1)
        self.conv2_1d = BasicConv1D(2, 4, kernel_size = 3, padding = 1)
        self.conv3_1d = BasicConv1D(4, 8, kernel_size = 3, padding = 1)
        self.fc = nn.Linear(args.lidar_delta // (2**3) * 8, args.feature_dim, bias = True)

        self.relu = nn.ReLU(inplace = True)
        self.maxpool_1d = nn.MaxPool1d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1_1d(x)
        x = self.maxpool_1d(x)
        x = self.conv2_1d(x)
        x = self.maxpool_1d(x)
        x = self.conv3_1d(x)
        x = self.maxpool_1d(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        return x


