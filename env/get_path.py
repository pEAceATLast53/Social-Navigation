import networkx as nx
import cv2
import numpy as np
import os
import random
import copy
import joblib

class PathGenerator:
    def __init__(self, radius, res): 
        self.map_img = None
        self.g = None
        self.obstacle_graph = None

        self.res = res
        self.scale = self.res * 0.01
        self.inv_scale = 100 / self.res

        self.radius = int(radius * self.inv_scale)

    def load_graph(self, graph_dir):
        self.g = joblib.load(graph_dir)

    def build_graph(self, map_dir):
        self.map_img = cv2.imread(map_dir, cv2.IMREAD_GRAYSCALE)
        g = nx.Graph()
        for i in range(self.map_img.shape[0] // self.radius):
            for j in range(self.map_img.shape[1] // self.radius):
                h_coord = self.radius*i
                w_coord = self.radius*j
                if self.map_img[h_coord, w_coord] == 0:
                    continue

                patch = self.map_img[max(h_coord-self.radius, 0):min(h_coord+self.radius+1, self.map_img.shape[0]), \
                    max(w_coord-self.radius,0):min(w_coord+self.radius+1, self.map_img.shape[1])]
                if patch.shape==(2*self.radius+1, 2*self.radius+1) and (patch==255).all(): g.add_node((h_coord,w_coord)) 
                else: continue

                # + radius, - radius
                patch = self.map_img[h_coord:min(h_coord+2*self.radius+1, self.map_img.shape[0]), \
                    max(w_coord-2*self.radius,0):min(w_coord+1, self.map_img.shape[1])]
                if patch.shape==(2*self.radius+1,2*self.radius+1) and (patch==255).all(): g.add_edge((h_coord, w_coord), \
                    (h_coord+self.radius,w_coord-self.radius), weight=self.radius*np.sqrt(2)) 

                # + radius, 0
                patch = self.map_img[max(h_coord, 0):min(h_coord+2*self.radius+1, self.map_img.shape[0]), \
                    max(w_coord-self.radius,0):min(w_coord+self.radius+1, self.map_img.shape[1])]
                if patch.shape==(2*self.radius+1,2*self.radius+1) and (patch==255).all(): g.add_edge((h_coord, w_coord), \
                    (h_coord+self.radius,w_coord), weight=self.radius) 

                # + radius, + radius
                patch = self.map_img[max(h_coord, 0):min(h_coord+2*self.radius+1, self.map_img.shape[0]), \
                    max(w_coord,0):min(w_coord+2*self.radius+1, self.map_img.shape[1])]
                if patch.shape==(2*self.radius+1, 2*self.radius+1) and (patch==255).all(): g.add_edge((h_coord, w_coord), \
                    (h_coord+self.radius, w_coord+self.radius), weight=self.radius*np.sqrt(2))

                # 0, +radius
                patch = self.map_img[max(h_coord-self.radius, 0):min(h_coord+self.radius+1, self.map_img.shape[0]), \
                    max(w_coord,0):min(w_coord+2*self.radius+1, self.map_img.shape[1])]
                if patch.shape==(2*self.radius+1, 2*self.radius+1) and (patch==255).all(): g.add_edge((h_coord, w_coord), \
                    (h_coord, w_coord+self.radius), weight=self.radius)

        largest_cc = max(nx.connected_components(g), key=len)
        self.g = g.subgraph(largest_cc).copy()

    def build_obstacle_graph(self, map_dir, threshold):
        self.map_img = cv2.imread(map_dir, cv2.IMREAD_GRAYSCALE)
        g = nx.Graph()
        for i in range(self.map_img.shape[0] // self.radius):
            for j in range(self.map_img.shape[1] // self.radius):
                h_coord = self.radius*i
                w_coord = self.radius*j

                patch = self.map_img[max(h_coord-self.radius, 0):min(h_coord+self.radius+1, self.map_img.shape[0]), \
                    max(w_coord-self.radius,0):min(w_coord+self.radius+1, self.map_img.shape[1])]
                num_obs = len(np.where(patch == 0)[0])
                if num_obs >= int(threshold * (2*self.radius+1) * (2*self.radius + 1)): g.add_node((h_coord,w_coord)) 
                else: continue

                # + radius, - radius
                
                patch = self.map_img[h_coord:min(h_coord+2*self.radius+1, self.map_img.shape[0]), \
                    max(w_coord-2*self.radius,0):min(w_coord+1, self.map_img.shape[1])]
                num_obs = len(np.where(patch == 0)[0])
                if num_obs >= int(threshold * (2*self.radius+1) * (2*self.radius + 1)): g.add_edge((h_coord, w_coord), \
                    (h_coord+self.radius,w_coord-self.radius), weight=self.radius*np.sqrt(2)) 
                
                # + radius, 0
                
                patch = self.map_img[max(h_coord, 0):min(h_coord+2*self.radius+1, self.map_img.shape[0]), \
                    max(w_coord-self.radius,0):min(w_coord+self.radius+1, self.map_img.shape[1])]
                num_obs = len(np.where(patch == 0)[0])
                if num_obs >= int(threshold * (2*self.radius+1) * (2*self.radius + 1)): g.add_edge((h_coord, w_coord), \
                    (h_coord+self.radius,w_coord), weight=self.radius) 
                
         
                # + radius, + radius
                
                patch = self.map_img[max(h_coord, 0):min(h_coord+2*self.radius+1, self.map_img.shape[0]), \
                    max(w_coord,0):min(w_coord+2*self.radius+1, self.map_img.shape[1])]
                num_obs = len(np.where(patch == 0)[0])
                if num_obs >= int(threshold * (2*self.radius+1) * (2*self.radius + 1)): g.add_edge((h_coord, w_coord), \
                    (h_coord+self.radius, w_coord+self.radius), weight=self.radius*np.sqrt(2))
                
 
                # 0, +radius
                
                patch = self.map_img[max(h_coord-self.radius, 0):min(h_coord+self.radius+1, self.map_img.shape[0]), \
                    max(w_coord,0):min(w_coord+2*self.radius+1, self.map_img.shape[1])]
                num_obs = len(np.where(patch == 0)[0])
                if num_obs >= int(threshold * (2*self.radius+1) * (2*self.radius + 1)): g.add_edge((h_coord, w_coord), \
                    (h_coord, w_coord+self.radius), weight=self.radius)
                

        self.obstacle_graph = g

    def render_graph(self):
        self.map_render = cv2.cvtColor(self.map_img.transpose(1,0), cv2.COLOR_GRAY2RGB)
        for n in self.g.nodes:
            cv2.circle(self.map_render, n, 4, (255,0,0), -1)
        for e in self.g.edges:
            cv2.line(self.map_render, e[0], e[1], (0,0,255))
        cv2.imwrite('graph.png', self.map_render)

    def render_path(self, paths):
        self.map_render = cv2.cvtColor(self.map_img.transpose(1,0), cv2.COLOR_GRAY2RGB)
        for n in self.g.nodes:
            cv2.circle(self.map_render, n, 2, (0,0,255), -1)
        for e in self.g.edges:
            cv2.line(self.map_render, e[0], e[1], (255,0,0))  
        for path in paths:
            for p in path:
                cv2.circle(self.map_render, tuple(p), 4, (0,255,255), -1) 
        cv2.imwrite('shortest_path.png', self.map_render)     

    def sample_node(self, num):
        assert self.g is not None
        return random.sample(self.g.nodes, num)
    
    def shortest_path(self, source, target):
        assert self.g is not None
        path = np.array(nx.astar_path(self.g, source, target, heuristic=self.l2_distance))
        geodesic_distance = np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))
        return path, geodesic_distance
    
    def is_node(self, node):
        return tuple(node) in list(self.g.nodes)

    def closest_node(self, node):
        distances = np.array([self.l2_distance(node, n) for n in list(self.g.nodes)])
        return list(self.g.nodes)[np.argmin(distances)]

    def get_random_path(self, num_ped = 1, target_dist_min = 20.0, target_dist_max = 50.0, max_trials=100):
        assert self.g is not None
        target_dist_min = target_dist_min * 100 / self.res
        target_dist_max = target_dist_max * 100 / self.res
        start_pts = self.sample_node(num_ped)
        paths = []
        dists = []
        for p in range(num_ped):
            init_pos = start_pts[p]
            for _ in range(max_trials):
                target_pos = self.sample_node(1)[0] 
                path, dist = self.shortest_path(init_pos, target_pos)
                if target_dist_min <= dist <= target_dist_max: break
            #if not target_dist_min <= dist <= target_dist_max:
            #    print("Warning: Failed to sample initial and target positions", dist * self.res * 0.01)
            paths.append(path)
            dists.append(dist)
        return paths, dists

    def get_random_path_robot(self, personal_space, ped_init_points, target_dist_min = 20.0, target_dist_max = 50.0, max_trials=1000):
        assert self.g is not None
        ped_init_points = np.array(ped_init_points)
        for _ in range(max_trials):
            init_pos = self.sample_node(1)[0]
            init_distances = [np.linalg.norm(ped_init_points[i,:] - np.array(init_pos)) for i in range(ped_init_points.shape[0])]
            if min(init_distances) < self.radius * 2 + personal_space * 100 / self.res: continue
            target_pos = self.sample_node(1)[0] 
            path, dist = self.shortest_path(init_pos, target_pos)
            dist *= self.res * 0.01
            if dist >= target_dist_min: break
        while dist > target_dist_max:
            path = path[:-1, :]
            dist = np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)) * self.res * 0.01
        return path, dist

    def get_random_path_from_src(self, source, target_dist_min = 20.0, target_dist_max = 50.0, max_trials=100):
        assert self.g is not None
        target_dist_min = target_dist_min * 100 / self.res
        target_dist_max = target_dist_max * 100 / self.res
        start_pt = self.closest_node(source)
        for _ in range(max_trials):
            target_pos = self.sample_node(1)[0]
            path, dist = self.shortest_path(start_pt, target_pos)
            if target_dist_min <= dist <= target_dist_max: break
        #if not target_dist_min <= dist <= target_dist_max:
        #    print("Warning: Failed to sample initial and target positions", dist * self.res * 0.01)
        return path, dist

    def get_path_from_src_to_target(self, source, target):
        assert self.g is not None
        start_pt = self.closest_node(source)
        target_pos = self.closest_node(target)
        path, dist = self.shortest_path(start_pt, target_pos)
        return path, dist

    def l2_distance(self, v1, v2):
        """Returns the L2 distance between vector v1 and v2."""
        return np.linalg.norm(np.array(v1) - np.array(v2))    
