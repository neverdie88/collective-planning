from baselines.spaces.box import Box
from baselines.spaces.product import Product
import numpy as np
import re
import gym
from os.path import dirname, abspath
from baselines.special import probScale1D, categorical_sample, probScale
from scipy.spatial import distance
from gym.utils import seeding
import csv

class TemporalPolicePatrollingGame(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, N, time_shift = 0):
        d = dirname(dirname(abspath(__file__)))
        self.incident_file_num = 31
        self.stateNum = 24
        self.h = 100
        self.time_shift = time_shift
        time_range = [[0,7], [8,19],[20,24]]
        time_shift_h = np.array(time_range)*60
        self.period = time_shift_h[self.time_shift]
        allocation_file = (d + '/data/duty_frcs_F_1M2D_5thTeam.csv')
        csv_data = np.genfromtxt(allocation_file, delimiter=',')
        self.fixed_allocation = []# np.zeros(self.stateNum)
        start_hour = time_range[self.time_shift][0]
        end_hour = time_range[self.time_shift][1]
        #F_sector_centroid_distance_with_distance_and_cost
        for i in range(len(csv_data)):
            row = csv_data[i]
            if row[1]>=start_hour and row[2]<=end_hour:
                self.fixed_allocation.append(int(row[0]))
                #self.fixed_allocation[int(row[0])] += 1
        #travel_time[urgent][zone1][zone2][time_mean][time_var]
        self.travel_time = np.ones((3, self.stateNum, self.stateNum,2))#*100
        travel_time_file = (d + '/data/F_sector_centroid_distance_with_distance_and_cost.csv')#travel_time_google
        csv_data = np.genfromtxt(travel_time_file, delimiter=',')
        for i in range(1,len(csv_data)):
            row = csv_data[i]
            loc1 = int(row[0])
            loc2 = int(row[3])
            time = float(row[7])/60 + 5
            #non-urgent incident travel time
            self.travel_time[0, loc1, loc2, 0] = time
            #urgent incident travel time
            self.travel_time[1, loc1, loc2, 0] = time#/1.2
            #re-allocation travel time
            self.travel_time[2, loc1, loc2, 0] = time

        self.incident_set = []
        #"sector", "start_time", "is_urgent", "demand", "engagement_time"

        for i in range(1, self.incident_file_num+1):
            input_file =d + '/data/SyntheticMonthx1000/'+ str(i) + 'L.csv'
            daily_incidents = []
            csv_data = np.genfromtxt(input_file, delimiter=',', dtype = int)
            for i in range(len(csv_data)):
                row = csv_data[i]
                if (row[1]>=self.period[0])&(row[1]<=self.period[1]):
                    daily_incidents.append(row)
           # time_daily_incidents = np.array(daily_incidents)[:,1]
            #delta_time  = np.roll(time_daily_incidents, -1, axis=0) - time_daily_incidents
            self.incident_set.append(daily_incidents)



        self.t = 0
        self.start_time = self.period[0]
        # self.initialDistribution = np.ones(self.stateNum, dtype=float)/self.stateNum
        self.N = len(self.fixed_allocation)
        #randomly extract an incident scenario
        self.incidents = self.incident_set[np.random.randint(7)]
        self.max_waiting_time = 1000
        self.max_obs_value = -1000
        self.delay_cost_coeff = -1
        self.lose_passenger_cost = -100
        self.adjacent_list = []
        self.initialDistribution = np.ones(self.stateNum, dtype=float)/self.stateNum
        for i in range(self.stateNum):
            adjacent_nodes = []
            for j in range(self.stateNum):
                if self.travel_time[0, i, j, 0] <10.0:
                    adjacent_nodes.append(j)
            self.adjacent_list.append(adjacent_nodes)
        # adjacent_temp = np.argpartition(self.travel_time[0, 7, :, 0], 5)[:5]
        # self.adjacent_list[7] = list(adjacent_temp)
        self.scenario_index = 0
        self.urgentResponse = 10
        self.nonurgentResponse = 20
        self.time_interval = 3
        self.C = 5
        self.discretized_travel_times = np.zeros((self.stateNum, self.stateNum), dtype=int)
        for loc1 in range(self.stateNum):
            for loc2 in range(self.stateNum):
                if loc1!=loc2:
                    discretized_time = int(self.travel_time[2, loc1, loc2, 0]/self.C)
                    self.discretized_travel_times[loc1, loc2] = discretized_time
        a = [i for i in range(24)]
        a[0] = [0,1,3,4,6]
        a[1] = [1,0,3]
        a[2] = [2,3,4,5,6]
        a[3] = [3,0,1,2,4]
        a[4] = [4,0,2,3,5,6]
        a[5] = [5,2,4,6,21]
        a[6] = [6,0,7,4,5,3]
        a[7] = [7,6,8,9]
        a[8] = [8,7,9,10,11]
        a[9] = [9,7,8,10,11]
        a[10] = [10,8,9,11,13,14]
        a[11] = [11,8,9,10,12,13,14]
        a[12] = [12,11,13,15,18]
        a[13] = [13,10,11,12,14,15,17,18]
        a[14] = [14,10,11,13,17,18,20,21]
        a[15] = [15,12,16,18]
        a[16] = [16,15,17,18,19,20]
        a[17] = [17,13,14,16,18,20,21,19]
        a[18] = [18,12,13,14,15,16,17]
        a[19] = [19,16,17,20,22,23]
        a[20] = [20,14,16,17,19,21,22,23]
        a[21] = [21,5,14,17,20,23]
        a[22] = [22,19,20,23]
        a[23] = [23,19,20,21,22]
        self.adjacent_list = a

    def generate_demand(self):
        incident = self.incidents[self.t]
        #map origin and dest to the node in the map
        loc = incident[0]
        # dest = incident[0]
        start_time = incident[1]
        is_urgent = incident[2]
        demand = incident[3]
        engagement_time = incident[4]
        return loc, start_time, is_urgent, demand, engagement_time

    def reset(self):
        self.t = 0
        self.clock_time = 0
        self.start_time = self.period[0]
        # randomly extract an incident scenario
        self.incidents = self.incident_set[self.scenario_index]#np.array(self.fixed_allocation)#np.random.randint(7)#np.random.randint(self.incident_file_num)
        # print('scenario_index:'+ str(self.scenario_index))
        self.scenario_index += 1
        self.scenario_index = self.scenario_index%self.incident_file_num
        #agent location
        self.agent_locs = self.fixed_allocation#np.array([categorical_sample(self.initialDistribution, np.random) for _ in range(self.N)])#
        #how many time step ahead agent would be free
        self.assignment_done_time = []
        self.assignment_done_loc = np.array([], dtype=int)
        # summary of free agent

        self.S = np.zeros((self.stateNum, self.C+1))  # (self.C+1))
        for i in range(self.N):
                self.S[self.agent_locs[i],-1] += 1 #/ max(self.agent_status[i], 1.0)
        self.obs = np.array(self.S)#np.zeros((self.N, self.stateNum))
        self.state = self.S[:,-1]
        self.state_count = np.tile(self.S[:,0, np.newaxis], [1, self.stateNum]).flatten()
        return self.obs,  self.state_count#, self.assignment_validity





    @property
    def observation_space(self):
        return (Box(low=-self.max_obs_value, high=self.max_obs_value, shape=(self.stateNum, self.C+1)))

    @property
    def action_space(self):
        components = []
        for i in range(self.stateNum):
            components.append(Box(low=0, high=self.N, shape=(self.stateNum)))
        return Product(components)

    def _step(self, prob):
        self.t += 1
        prob = probScale(np.asarray(prob, dtype=float))
        a = np.asarray(
            [np.random.multinomial(self.state[i], prob[i]) for i in range(prob.shape[0])])
        tempDests = np.zeros(self.stateNum)
        for i in range(self.stateNum):
            tempDests[i] += a[i, i]
            for j in range(self.stateNum):
                if i != j:
                    for k in range(a[i, j]):
                        self.assignment_done_loc = np.append(self.assignment_done_loc, j)
                        self.assignment_done_time = np.append(self.assignment_done_time, self.travel_time[2, i, j, 0])




        # generate the new demand
        loc, arrival_time, is_urgent, self.demand, engagement_time = self.generate_demand()
        time_past = arrival_time - self.clock_time
        self.clock_time = arrival_time
        # update agent status at the arrival of new demand
        self.assignment_done_time = np.maximum(0, np.array(self.assignment_done_time) - time_past)
        done_list = []
        for i in range(len(self.assignment_done_time)):
            if self.assignment_done_time[i]==0:
                done_list.append(i)
                tempDests[self.assignment_done_loc[i]] += 1

        #purge the on-the-fly list
        self.assignment_done_time = np.delete(self.assignment_done_time, done_list)
        self.assignment_done_loc = np.delete(self.assignment_done_loc, done_list)

        if np.sum(tempDests)+len(self.assignment_done_time)<len(self.fixed_allocation):
            print('error')

        #search through the available polices
        back_to_base = 0
        best_available_loc = 0
        best_available_val = 10000000
        for i in range(len(tempDests)):
            if tempDests[i]>0:
                time_to_pickup = self.travel_time[is_urgent, i, loc, 0]
                if time_to_pickup<best_available_val:
                    best_available_loc = i
                    best_available_val = time_to_pickup
                    back_to_base = time_to_pickup*2

        #search through the soon available polices
        best_future_loc = -1
        best_future_val = best_available_val
        if np.min(tempDests)<0:
            print('error')
        for i in range(len(self.assignment_done_time)):
            time_to_pickup = self.travel_time[is_urgent, self.assignment_done_loc[i], loc, 0] + self.assignment_done_time[i]
            if time_to_pickup<best_future_val:
                best_future_loc = i
                best_future_val = time_to_pickup
                back_to_base =time_to_pickup + self.travel_time[is_urgent, self.assignment_done_loc[i], loc, 0]


        if best_future_loc >=0:
            #self.assignment_done_loc[best_future_loc] = loc
            self.assignment_done_time[best_future_loc] += back_to_base + engagement_time
            if is_urgent:
                delay_time = max(0.0, best_future_val-self.urgentResponse)
                reward = -10*(float(best_future_val>self.urgentResponse))#+delay_time) #self.delay_cost_coeff*best_future_val
            else:
                delay_time = max(0.0, best_future_val-self.nonurgentResponse)
                reward = -10*(float(best_future_val>self.nonurgentResponse))#+delay_time)
        else:
            self.assignment_done_loc = np.append(self.assignment_done_loc, best_available_loc)
            self.assignment_done_time = np.append(self.assignment_done_time, back_to_base + engagement_time)
            tempDests[best_available_loc]-=1
            if is_urgent:
                delay_time = max(0.0, best_future_val - self.urgentResponse)
                reward = -10*(float(best_future_val>self.urgentResponse))#+delay_time)#self.delay_cost_coeff * best_available_val
            else:
                delay_time = max(0.0, best_future_val - self.nonurgentResponse)
                reward = -10 * (float(best_future_val > self.nonurgentResponse))# + delay_time)

        if np.min(tempDests)<0:
            print('error')
        self.state = tempDests
        self.S = np.zeros((self.stateNum,self.C+1))# (self.C+1))
        self.S[:,-1] = tempDests
        for i in range(len(self.assignment_done_loc)):
            avail_time = int(self.assignment_done_time[i] / self.time_interval)
            if avail_time<self.C:
                self.S[self.assignment_done_loc[i], avail_time] += 1
        self.obs = np.array(self.S)

        done = False
        if self.t >= len(self.incidents)-1:
            done = True
        return self.obs, a, reward, done, {'state': self.S, 'count': [], 'rewards': [], 'delay':delay_time}#a

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def _render(self, mode="human", close=False):
        pass
