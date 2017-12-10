from joblib import Parallel, delayed
import multiprocessing
import sys
import pickle
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix, distance
from sklearn.neural_network import MLPRegressor, MLPClassifier
import random

from matplotlib.patches import Circle, Arrow
from matplotlib.collections import PatchCollection

import time
from datetime import datetime

def point_to_line_dist(point, line):
    """Calculate the distance between a point and a line segment.

    To calculate the closest distance to a line segment, we first need to check
    if the point projects onto the line segment.  If it does, then we calculate
    the orthogonal distance from the point to the line.
    If the point does not project to the line segment, we calculate the 
    distance to both endpoints and take the shortest distance.

    :param point: Numpy array of form [x,y], describing the point.
    :type point: numpy.core.multiarray.ndarray
    :param line: list of endpoint arrays of form [P1, P2]
    :type line: list of numpy.core.multiarray.ndarray
    :return: The minimum distance to a point.
    :rtype: float
    """
    # unit vector
    unit_line = line[1] - line[0]
    norm_unit_line = unit_line / np.linalg.norm(unit_line)
    
    # compute the perpendicular distance to the theoretical infinite line
    segment_dist = (np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) / np.linalg.norm(unit_line))

    diff = (
        (norm_unit_line[0] * (point[0] - line[0][0])) + 
        (norm_unit_line[1] * (point[1] - line[0][1]))
    )

    x_seg = (norm_unit_line[0] * diff) + line[0][0]
    y_seg = (norm_unit_line[1] * diff) + line[0][1]

    endpoint_dist = min(
        np.linalg.norm(line[0] - point),
        np.linalg.norm(line[1] - point)
    )

    # decide if the intersection point falls on the line segment
    lp1_x = line[0][0]  # line point 1 x
    lp1_y = line[0][1]  # line point 1 y
    lp2_x = line[1][0]  # line point 2 x
    lp2_y = line[1][1]  # line point 2 y
    is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
    is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
    if is_betw_x and is_betw_y:
        return segment_dist
    else:
        # if not, then return the minimum distance to the segment endpoints
        return endpoint_dist

class World:
    
    # Max distance between edges of UAVs to count as an encounter, and start collision avoidance
    ENCOUNTER_RADIUS = 30.
    DELTA_T = 0.5
    
    def _plot_status(self):
        
        plt.clf()
        
        self.fig, self.ax = plt.subplots(1,1)
        self.ax.grid()
        self.fig.set_size_inches((20, 20))
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_xlim(0,self.GRID_LEN)
        self.ax.set_ylim(0,self.GRID_LEN)
        
        patches = []
        color_array = []
        for idx, uav in enumerate(self.uavs):
            if uav is None: continue
            color = self.uav_colors[idx]
            color_array.append(color)
            patches.append(Circle(uav.curr_loc, UAV.RADIUS, label=idx))
            patches.append(Circle(uav.curr_loc, UAV.CRASH_RADIUS+UAV.RADIUS, fill=False, ls='dotted', label=idx))
            patches.append(Circle(uav.curr_loc, World.ENCOUNTER_RADIUS+UAV.RADIUS, fill=False, ls='dashed', label=idx))
            patches.append(Circle(uav.END_LOC, UAV.APPROX_DEST_DIST, label=idx))
            uav_orientation_vec = uav.SPEED * np.array([np.cos(uav.orientation), np.sin(uav.orientation)])
            patches.append(Arrow(uav.curr_loc[0], uav.curr_loc[1], uav_orientation_vec[0], uav_orientation_vec[1]))
        if len(patches) > 0:
            p = PatchCollection(patches, alpha=0.4)
            p.set_array(np.repeat(color_array, 5))
            self.ax.add_collection(p)
            
    def __init__(self, cas_alg='mdp', grid_len=100, num_uavs=2, time_horizon=100):

        self.CAS_ALG = cas_alg
        self.GRID_LEN = grid_len
        self.NUM_UAVS = num_uavs
        self.TIME_HORIZON = time_horizon
        
        self.uavs = [UAV(idx, self) for idx in range(num_uavs)]
        
        # data struct to store x, y location of each uav in the grid world
        self.uav_coords = np.zeros((num_uavs,2))
        self.distance_mat = None
        self.encounter_mat = None
        
        self._update_uav_coords()
        self._update_encounter_mat()
        
        self.uav_colors = np.random.rand(num_uavs,)
        #self._plot_status()
                    
    def _move_uav(self, idx):

        uav = self.uavs[idx]

        assert uav is not None

        orientation = uav.move(cas_alg=self.CAS_ALG, debug=debug)

        if uav.status == UAV.CRASH:
                return -1
        elif uav.status == UAV.DONE:
                return (uav.shortest_time, uav.time)


    def run(self, debug=0, sleeptime=0.1, parallel=False):

        if parallel:
                num_cores = multiprocessing.cpu_count()
        
        crashes = np.zeros((self.TIME_HORIZON,))
        success = np.zeros((self.TIME_HORIZON,))
        success_times = []
        
        uav_move_order = np.array(list(range(len(self.uavs))))
        for self.time in range(self.TIME_HORIZON):

            if self.time % 100 == 0: print("Time: %d\r" % self.time)
            sys.stdout.flush()
            
            random.shuffle(uav_move_order)
            
            if debug>0: 
                display.clear_output(wait=True)

            if debug>0:
                self._plot_status()
                display.display(self.fig)
                
            nan_idx = []
           
            if not parallel: 
                    for idx in uav_move_order:
                        
                        uav = self.uavs[idx]
                                        
                        if uav is None: continue
                            
                        orientation = uav.move(cas_alg=self.CAS_ALG, debug=debug)
                        
                        if uav.status == UAV.CRASH:
                            if debug>1:print("CRASHED")
                            crashes[self.time] += 1
                            nan_idx.append(idx)
                        elif uav.status == UAV.DONE:
                            if debug>1:print("SUCCESS")
                            success[self.time] += 1
                            success_times.append((uav.shortest_time, uav.time))
                            nan_idx.append(idx)                
            else:

                results = Parallel(n_jobs=num_cores)(delayed(self._move_uav)(idx) for idx in uav_move_order)

                for idx, result in enumerate(results):

                        if result == -1: 
                                crashes[self.time] += 1
                                nan_idx.append(idx)     
                        elif result is not None: 
                                successes[self.time] += 1
                                success_times.append(result)
                                nan_idx.append(idx)
            
            for idx in nan_idx: 
                self.uavs[idx] = UAV(idx, self)
                
            self._update_uav_coords()
            self._update_encounter_mat()
            
            if debug>0:
                time.sleep(sleeptime)
        
        print("---Results--")
        print("Total success: ", sum(success))
        print("Total crashed: ", sum(crashes))
        
        return crashes, success, success_times
    
    def _update_uav_coords(self):
        
        for idx, uav in enumerate(self.uavs): 
            
            if uav is None: continue
                
            self.uav_coords[idx] = uav.curr_loc
            
    def _update_encounter_mat(self):
        
        # num_uav * num_uav matrix of pairwise euclidean distances
        self.distance_mat = distance_matrix(self.uav_coords, self.uav_coords)
        
        self.distance_mat -= 2 * UAV.RADIUS
        
        # contains true if encounter (i.e distance is less than encounter radius)
        self.encounter_mat = self.distance_mat < World.ENCOUNTER_RADIUS
        
        # cant encounter ourselves
        np.fill_diagonal(self.encounter_mat, False)
    
    # senses and returns closest uav
    def sense(self, uav_idx):
        
        # array of num_uavs which extacts row corresponding to uav_idx
        uav_encounters = self.encounter_mat[uav_idx, :]
        
        # return true uav indices (corresponding to encounter)
        encounter_uav_idxs = np.where(uav_encounters)[0]
        
        encounter_uav_locs = [self.uavs[idx].curr_loc for idx in encounter_uav_idxs]
        
        if len(encounter_uav_idxs) == 0: 
            return None
        
        encounter_dists = self.distance_mat[uav_idx, :]
        
        # idx of closest encountering uav to us
        encounter_uav_idx = np.argsort(encounter_dists)[1]
        encounter_uav = self.uavs[encounter_uav_idx]
        encounter_dist = encounter_dists[encounter_uav_idx]
        
        # these are part of the relevant q learning state variables
        sensor_data = [encounter_uav_locs, encounter_dist, encounter_uav.SPEED, encounter_uav.orientation]
        
        return sensor_data


class UAV:
    
    RADIUS = 1.
    # Max distance between edges of UAVs to count as a crash
    CRASH_RADIUS = 0.

    MIN_SPEED = 5.
    MAX_SPEED = 15.
    
    ACTIONS = [np.deg2rad(deg) for deg in range(-20, 25, 5)]
    
    CRASH_REWARD = -100.
    ESCAPED_REWARD = 100.
    BASE_ENCOUNTER_REWARD = -100
    
    
    GAMMA = 0.95
    
    MAX_EXPLORATION_PROB = 0.5
    EXPLORATION_DECAY = 0.95
    MIN_EXPLORATION_PROB = 0.01
    
    APPROX_DEST_DIST = 0.1
    
    NAV="NAV"
    DONE="DONE"
    CRASH="CRASH"
        
    def __init__(self, idx, world):
        
        self.IDX = idx
        
        self.WORLD = world
        
        # x, y in 0 to grid_len
        self.START_LOC = world.GRID_LEN * np.random.random_sample(2)
        self.END_LOC = world.GRID_LEN * np.random.random_sample(2)
        
        self.curr_loc = self.START_LOC
        
        # random number between MIN_SPEED (5) and MAX_SPEED (10)
        self.SPEED = max(UAV.MIN_SPEED, UAV.MAX_SPEED * np.random.random_sample(1)[0])
        
        # whether in encounter or not, in order to trigger the mdp
        self.in_encounter = False
        
        self.orientation = self._navigation_angle()
        
        # state variables used by mdp
        self.previous_state = self.state = None
        self.previous_action = None
        
        self.status = UAV.NAV

        self.time = 0
        self.shortest_time = np.ceil(np.linalg.norm(self.END_LOC-self.START_LOC)/(self.SPEED))

    
    # build state variables, after normalizing to lie between 0 to 1
    def _build_state(self, sensor_data):
        
        self_pos_var = self.curr_loc/self.WORLD.GRID_LEN
        self_speed_var = np.array([(self.SPEED-UAV.MIN_SPEED)/(UAV.MAX_SPEED-UAV.MIN_SPEED)])
        self_dir_var = np.array([self.orientation/(2*np.pi)])
        
        encounter_center, encounter_dist, encounter_speed, encounter_dir = sensor_data
        encounter_dist_var = np.array([encounter_dist/self.WORLD.ENCOUNTER_RADIUS])
        encounter_speed_var = np.array([(encounter_speed-UAV.MIN_SPEED)/(UAV.MAX_SPEED-UAV.MIN_SPEED)])
        encounter_dir_var = np.array([encounter_dir/(2*np.pi)])
        
        # this is the mdp state at any given time
        state = np.concatenate((self_pos_var, self_speed_var, self_dir_var, encounter_dist_var, encounter_speed_var, encounter_dir_var))
            
        return state
            
    def move(self, cas_alg='mdp', debug=0):
        
        if cas_alg=='mdp':
            return self.move_mdpcas(debug=debug)
        elif cas_alg=='random':
            return self.move_randomcas(debug=debug)
    
    def move_randomcas(self, debug=0):
        
        if debug>1: 
            print("\n---Navigating %d---" % self.IDX)
        
        nav_angle = self._navigation_angle()
        
        sensor_data = self.WORLD.sense(self.IDX)
        
        if sensor_data:
            
            if debug>1:
                print("Encounter")
            
            encounter_dist = sensor_data[1]
            if encounter_dist <= UAV.CRASH_RADIUS: 
                self.status = UAV.CRASH
                return
            
            action_idx = np.random.randint(0, len(UAV.ACTIONS))
            avoidance_update = UAV.ACTIONS[action_idx]
            
            updated_nav_angle = nav_angle + avoidance_update
        
            if debug>1:
                print("Nav angle for %d updated from %.2f to %.2f" % (self.IDX, nav_angle, updated_nav_angle))

            if not self._update_loc(updated_nav_angle, sensor_data):
                return
            return updated_nav_angle
        
        else:
            
            if debug>1:
                print("No encounter")
                print("Navigating to %.2f" % nav_angle)
                
            self._update_loc(nav_angle)
            return nav_angle
    
    # uav is thinking about where to move
    # and then when update_loc is called
    def move_mdpcas(self, debug=0):
        
        if debug>1: print("\n---Navigating %d---" % self.IDX)
        
        self.previous_state = self.state
        were_in_encounter = self.in_encounter
        
        # first found shortest path direction
        nav_angle = self._navigation_angle()
        
        # returns closest encounter uav's distance, speed and orientation
        # or None if no encounter
        sensor_data = self.WORLD.sense(self.IDX)
                
        if sensor_data is not None:
            self.in_encounter = True    
        else:
            sensor_data = [np.array([0, 0]), 0, 0, 0]
            self.in_encounter = False
        self.state = self._build_state(sensor_data)
        
        encounter_dist = sensor_data[1]
        
        # no encounter
        if not were_in_encounter and not self.in_encounter:
            
            if debug>1:
                print("No encounter")
                print("Navigating to %.2f" % nav_angle)
                                    
            self._update_loc(nav_angle)
            return nav_angle
        
        # escaped encounter
        elif were_in_encounter and not self.in_encounter:
            
            reward = self._reward(encounter_dist=encounter_dist)
            self._update_q(reward)
            
            if debug>1:
                print("Escaped encounter")
                print("Received reward of %.2f" % reward)
                print("Navigating to %.2f" % nav_angle)
            
            self._update_loc(nav_angle)
            return nav_angle
        
        # new encounter
        elif not were_in_encounter and self.in_encounter:
            
            if debug>1:
                print("New encounter")
                
            # crash
            if encounter_dist <= UAV.CRASH_RADIUS: 
                self.status = UAV.CRASH
                return
            
            avoidance_update = self._mdpcas()
            updated_nav_angle = nav_angle + avoidance_update
        
            if debug>1:        
                print("Nav angle for %d updated from %.2f to %.2f" % (self.IDX, nav_angle, updated_nav_angle))

            if not self._update_loc(updated_nav_angle, sensor_data):
                return
            return updated_nav_angle
            
        # continuing encounter
        elif were_in_encounter and self.in_encounter:
            
            reward = self._reward(encounter_dist=encounter_dist)
            self._update_q(reward)
            
            if debug>1:
                print("Continuing encounter")            
                print("Received reward of %.2f" % reward)
            
            # crash
            if encounter_dist <= UAV.CRASH_RADIUS: 
                self.status = UAV.CRASH
                return
            
            avoidance_update = self._mdpcas()
            updated_nav_angle = nav_angle + avoidance_update
            
            if debug>1:
                print("Nav angle for %d updated from %.2f to %.2f" % (self.IDX, nav_angle, updated_nav_angle))

            if not self._update_loc(updated_nav_angle, sensor_data):
                return
            return updated_nav_angle
        
    # update current location in direction of nav_angle with move length corresponding to speed
    def _update_loc(self, nav_angle, sensor_data=None):
                
        self.orientation = nav_angle

        dist_to_dest = distance.euclidean(self.curr_loc, self.END_LOC)
        
        move_len = min(dist_to_dest, self.WORLD.DELTA_T * self.SPEED)
        delta = move_len * np.array([np.cos(nav_angle), np.sin(nav_angle)])
        
        potential_loc = self.curr_loc + delta
        potential_loc = np.clip(potential_loc, UAV.RADIUS, self.WORLD.GRID_LEN-UAV.RADIUS)
        
        # make sure you don't ghost through another uav
        if sensor_data:
            encounter_locs, encounter_dist, encounter_speed, encounter_dir = sensor_data
            for encounter_loc in encounter_locs:
                dist = point_to_line_dist(encounter_loc, [self.curr_loc, potential_loc])
                if dist <= 2 * UAV.RADIUS:
                    self.status = UAV.CRASH
                    return False
        
        self.curr_loc = potential_loc
        
        dist_to_dest = distance.euclidean(self.curr_loc, self.END_LOC)
        
        if dist_to_dest < UAV.APPROX_DEST_DIST:
            self.status = UAV.DONE

        self.time += 1   
 
        return True
        
    def _update_q(self, reward):

        global Q
                
        previous_action_var = np.array([(self.previous_action-min(UAV.ACTIONS))/(max(UAV.ACTIONS)-min(UAV.ACTIONS))])
        feat = np.concatenate((self.previous_state, previous_action_var)).reshape(1, -1)
        
        if reward == UAV.ESCAPED_REWARD:
            _, future_value = self._best_action()
            label = np.array([reward + UAV.GAMMA * future_value])
        else:
            label = np.array([reward])
                
        Q.partial_fit(feat, label)
        
    def _mdpcas(self, alg='mdp'):
        
        # best policy
        best_action_idx, _ = self._best_action()

        # randomize policy with some probability for exploration purposes
        action_idx = self._exploration_policy(best_action_idx)
        action = UAV.ACTIONS[action_idx]

        self.previous_action = action

        return action
    
    # epsilon greedy exploration policy
    def _exploration_policy(self, best_action_idx):
        
        chance = np.random.random()
        
        exploration_prob = max(UAV.MAX_EXPLORATION_PROB * (UAV.EXPLORATION_DECAY**self.WORLD.time), UAV.MIN_EXPLORATION_PROB)
        
        if chance < exploration_prob:
            return np.random.randint(0, len(UAV.ACTIONS))
        else:
            return best_action_idx
    
    # loops over all possible actions, and picks the one with the highest estimated q value given current state 
    def _best_action(self):
        global Q
        
        try:
            best_value = -np.inf
            best_action_idx = None
            for action_idx, action in enumerate(UAV.ACTIONS):
                action_var = np.array([(action-min(UAV.ACTIONS))/(max(UAV.ACTIONS)-min(UAV.ACTIONS))])
                feat = np.concatenate((self.state, action_var)).reshape(1, -1)
                value = Q.predict(feat)
                if value > best_value:
                    best_value = value
                    best_action_idx = action_idx
        except Exception as e:
            best_value = np.random.random()
            best_action_idx = np.random.randint(0, len(UAV.ACTIONS))
                            
        return best_action_idx, best_value
    
    def _reward(self, encounter_dist):
        if encounter_dist > self.WORLD.ENCOUNTER_RADIUS:
            reward = UAV.ESCAPED_REWARD
        elif encounter_dist <= UAV.CRASH_RADIUS:
            reward = UAV.CRASH_REWARD
        else:
            # inverse proportionality to encounter dist
            reward = UAV.BASE_ENCOUNTER_REWARD / encounter_dist
        return reward
        
    # finds angle of shortest path between curr loc and end loc
    def _navigation_angle(self):
        dir_vec = self.END_LOC - self.curr_loc   
        angle = np.arctan2(dir_vec[1], dir_vec[0])
        angle = angle if angle>0 else 2*np.pi+angle
        return angle


# In[5]:

version_num = int((datetime.utcnow()- datetime(1970, 1, 1)).total_seconds())

time_horizon = 10**4 #5
run_time = time_horizon * World.DELTA_T
grid_len = 10**4 #4
num_uavs = 10**2 #3
debug = 0

parallel = False

Q = None

for grid_len_exp in range(1, 4):

        grid_len = 10**grid_len_exp

        print("Running experiment with: %d uavs, %d grid_len, %d timesteps" % (num_uavs, grid_len, time_horizon))
        
        Q = MLPRegressor((50, 20))

        # ### MDP CAS Training

        world = World(cas_alg='mdp', grid_len=grid_len, num_uavs=num_uavs, time_horizon=time_horizon)
        # crashes, success = world.run(cas_alg='mdp', debug=1, sleeptime=0.001)
        crashes, success, success_times = world.run(debug=0, parallel=parallel)
        pickle.dump([crashes, success, success_times], open('mdp_train_%d_%d' % (grid_len_exp, version_num), 'wb'))
        #plt.hist(success, bins=range(0, time_horizon))
        #plt.show()
        #plt.hist(crashes,  bins=range(0, time_horizon))
        #plt.show()


        # ### MDP CAS Testing

        world = World(cas_alg='mdp', grid_len=grid_len, num_uavs=num_uavs, time_horizon=time_horizon)
        # crashes, success = world.run(cas_alg='mdp', debug=1, sleeptime=0.001)
        crashes, success, success_times = world.run(debug=0, parallel=parallel)
        pickle.dump([crashes, success, success_times], open('mdp_test_%d_%d' % (grid_len_exp, version_num), 'wb'))

        #plt.hist(success, bins=range(0, time_horizon))
        #plt.show()
        #plt.hist(crashes,  bins=range(0, time_horizon))
        #plt.show()


        # ### Random CAS Testing

        world = World(cas_alg='random', grid_len=grid_len, num_uavs=num_uavs, time_horizon=time_horizon)
        # crashes, success = world.run(debug=2, sleeptime=5, cas_alg='random')
        crashes, success, success_times = world.run(debug=0, parallel=parallel)
        pickle.dump([crashes, success, success_times], open('random_test_%d_%d' % (grid_len_exp, version_num), 'wb'))

        #plt.hist(success, bins=range(0, time_horizon, int(time_horizon/1000)))
        #plt.show()
        #plt.hist(crashes, bins=range(0, time_horizon, int(time_horizon/1000)))
        #plt.show()

