from baselines.spaces.box import Box
from baselines.spaces.product import Product
import numpy as np
import re
import gym
from os.path import dirname, abspath
from baselines.special import probScale

class CGMRealTaxi(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, N, variance, satisfied_percentage, penalty_weight):
        self.satisfied_percentage = satisfied_percentage
        self.penalty_weight = penalty_weight
        self.h = 48
        d = dirname(dirname(abspath(__file__))) + '/data'
        print(d)
        allZoneFile = (d + '/all_zones.csv')
        with open(allZoneFile, "r") as f:
            content = f.read().splitlines()

        zones = list(map(int, content))
        costMatrix = np.zeros((len(zones), len(zones)))

        costFile = d +'/cost_all_zones.csv'
        inputFile = open(costFile, 'r')
        for i in range(len(zones)):
            line = inputFile.readline()
            words = re.split('\s+', line)
            for w in range(len(words) - 1):
                costMatrix[i, w] = float(words[w])

        inputFile.close()

        fileIndex = 1001
        descriptFile = (d+ '/%d.log' % fileIndex)
        inputFile = open(descriptFile, 'r')
        H = 0
        scenarioNum = 12
        goodZone = []
        for k in range(20):
            line = inputFile.readline()
            if 'Flow period' in line:
                words = re.split('\s+', line)
                H = int(words[5]) + 1
            if 'Good Zones (WD)' in line:
                words = re.split('\s+', line)
                zoneNum = len(words) - 5
                goodZone.append(list(map(int, words[4:len(words) - 1])))

        inputFile.close()

        stateIndex = list(goodZone[0])
        for k in range(11):
            fileIndex += 1
            descriptFile = (d + '/%d.log' % fileIndex)
            inputFile = open(descriptFile, 'r')
            for j in range(20):
                line = inputFile.readline()
                if 'Flow period' in line:
                    words = re.split('\s+', line)
                    H = int(words[5]) + 1
                if 'Good Zones (WD)' in line:
                    words = re.split('\s+', line)
                    zoneNum = len(words) - 5
                    goodZone.append(list(map(int, words[4:len(words) - 1])))
                    for index in goodZone[k + 1]:
                        if index not in stateIndex:
                            stateIndex.append(index)
            inputFile.close()

        stateIndex.sort()

        badZoneIndices = []
        for i in zones:
            if i not in stateIndex:
                badZoneIndices.append(zones.index(i))

        self.stateNum = len(stateIndex)
        self.actionNum = self.stateNum# +1
        stateNum = len(stateIndex)

        flows = np.zeros((scenarioNum, H, stateNum, stateNum), int)
        for scenario in range(12):
            fileIndex = 1001 + scenario
            flowFile = (d+ '/%d-flow-wd.csv' % fileIndex)
            inputFile = open(flowFile, 'r')
            zoneNum = len(goodZone[scenario])
            for i in range(H):
                line = inputFile.readline()
                for j in range(zoneNum):
                    line = inputFile.readline()
                    words = re.split('\s+', line)
                    for w in range(len(words) - 1):
                        flows[scenario, i, stateIndex.index(goodZone[scenario][j]), stateIndex.index(
                            goodZone[scenario][w])] = int(words[w])
            inputFile.close()

        inputFile = open(d + '/adjectiveZone.csv', 'r')
        self.adjacentMatrix = np.ones((87,9), int)*-1

        for i in range(87):
            line = inputFile.readline()
            words = re.split(',', line)
            for j in range(len(words) - 1):
                adjacentZone = int(words[j])
                if adjacentZone in stateIndex:
                    self.adjacentMatrix[i, j] = stateIndex.index(adjacentZone)

        inputFile.close()
        self.adjacentMatrix = np.delete(self.adjacentMatrix, badZoneIndices,0)

        self.adjacentList = []
        for i in range(self.stateNum):
            adjacentLocs = []
            for j in range(9):
                if(self.adjacentMatrix[i][j]>=0):
                    adjacentLocs.append(self.adjacentMatrix[i][j])
            self.adjacentList.append(adjacentLocs)


        costMatrix1 = np.zeros((self.stateNum, self.stateNum))
        for i in range(self.stateNum):
            for j in range(self.stateNum):
                costMatrix1[i, j] = costMatrix[zones.index(stateIndex[i]), zones.index(stateIndex[j])]

        self.costMatrix = costMatrix1



        self.N = N
        # P = np.zeros((self.stateNum, self.stateNum, self.stateNum), float)
        # for a in range(0, self.stateNum):
        #     for s in range(0, self.stateNum):
        #         P[a, s, a] = 1

        # self.initialDistribution = np.zeros(self.stateNum, float) / self.stateNum
        # self.initialDistribution[0] = 0.5
        # self.initialDistribution[self.stateNum - 1] = 0.5

        self.t = 0
        self.averagePayments = np.zeros((H, self.stateNum, self.stateNum))
        inputFile = open(d+'/averageReward.csv', 'r')
        for t in range(H):
            line = inputFile.readline()
            for s in range(self.stateNum):
                line = inputFile.readline().rstrip()
                words = re.split(',', line)
                self.averagePayments[t, s] = np.array(list(map(float, words[:words.__len__() - 1])))
        inputFile.close()
        self.averagePayments = self.averagePayments - self.costMatrix[np.newaxis, :, :]
        meanPayment = np.mean(self.averagePayments)
        self.averagePayments = self.averagePayments/meanPayment
        print('mean Payment:' + str(meanPayment))
        self.averagePayments = np.maximum(self.averagePayments,0)
        self.costMatrix = np.maximum(self.costMatrix, 1)
        self.rewardMagnitude = max(np.max(self.costMatrix), np.max(self.averagePayments))
        maxCost = np.max(costMatrix)
        self.flows = flows

        self.initialDistribution = np.sum(self.flows[:,0], axis=(0,2))
        self.initialDistribution = self.initialDistribution/ np.sum(self.initialDistribution)

        self.scenarioNum = scenarioNum
        #self.roamingCost = 3.0
        self.S = np.zeros(self.stateNum, int)
        self.S[:self.stateNum] = np.random.multinomial(self.N, self.initialDistribution)
        self.D = self.flows[np.random.randint(self.scenarioNum)]#np.random.randint(self.scenarioNum)
        self.totalD = np.sum(self.D, axis=2)
        self._state = np.dstack((self.S[:self.stateNum], self.totalD[0]))[0]
        self.high = self.N#max(self.N, self.maxD)
        self.low = 0
        del costMatrix
        del content
        self.variance = variance
        self.means = np.ones((self.h, self.stateNum))
        self.covs = np.ones(self.stateNum) * self.variance
        if variance>0.0:
            inputFile = open(d + '/%.1f.txt' % variance, 'r')
            line = inputFile.readline()
            words = re.split('\s+', line)
            self.covs = np.array(list(map(float, words)))
        average_demands = np.mean(np.sum(flows, axis=3), axis=0)
        high_demand_zones = np.array(average_demands > 250, dtype=int)
        high_demand_zones = (np.sum(high_demand_zones, axis=0) > 0)
        self.high_demand_zones = high_demand_zones
        self.selected_zones_multipliers = np.array(high_demand_zones, dtype=float)
        self.penalty_weights = self.penalty_weight*np.array(high_demand_zones, dtype=float)
        self.zero_costs = np.zeros((self.stateNum, self.stateNum))

        self.averagePassengerFlows = np.array(flows, float).sum(axis=0) / scenarioNum  # .sum(axis=0)/12
        self.averagePassengerLocation = self.averagePassengerFlows.sum(axis=2)
        self.averageTotalD = self.averagePassengerFlows.sum(axis=2)

        # self.zeroIndices = (averagePassengerLocation == 0)
        # self.averagePassengerLocation[zeroIndices] = 1
        # flowDistribution = averagePassengerFlows / averagePassengerLocation[:, :, np.newaxis]
        # # featureNormalizors = averagePassengerLocation*1.3
        # averagePassengerLocation[zeroIndices] = 0
        # numPassengerMax = np.max(averagePassengerLocation)
        # averagePassengerNum = np.mean(averagePassengerLocation)
        # taxiMax = float(N)
        # Rmax = np.max(averagePayments)
        # epsilon = 1
        # roamingCost = 1.0
        #
        #

    def reset(self):
        self.t = 0
        self.S = np.zeros(self.stateNum, int)
        self.S[:self.stateNum] = np.random.multinomial(self.N, self.initialDistribution)
        self.state = self.S
        self.D = self.flows[np.random.randint(self.scenarioNum)]#
        underlying = np.random.normal(self.means, self.covs[np.newaxis, :])
        underlying[underlying > 2] = 2
        underlying[underlying < 0] = 0
        self.totalD = np.sum(self.D, axis=2)  #
        self.realizedFlow = self.D.astype(float) / np.maximum(self.totalD[:, :, np.newaxis], 0.0001)
        self.totalD = np.round(self.totalD * underlying).astype(int)
        self._state = np.dstack((self.S[:self.stateNum], self.totalD[0]))[0]
        obs = np.zeros(self.h + self.stateNum*2)
        obs[self.h:self.stateNum + self.h] = self.S
        obs[self.stateNum + self.h:] = self.totalD[self.t]
        obs /= float(self.N)
        obs[self.t] = 1
        self.state_count = np.tile(self.state[:, np.newaxis], [1, self.actionNum]).flatten()
        return obs, self.state_count

    def scenario_generating(self, sample_num, future_step):
        demands = []
        passenger_flow_ratios = []
        trip_pays = []
        for sample in range(sample_num):
            D = self.flows[np.random.randint(self.scenarioNum)]  #
            underlying = np.random.normal(self.means, self.covs[np.newaxis, :])
            underlying[underlying > 2] = 2
            underlying[underlying < 0] = 0
            totalD = np.sum(self.D, axis=2)
            realizedFlow = D.astype(float) / np.maximum(totalD[:, :, np.newaxis], 0.0001)
            totalD = np.round(totalD * underlying).astype(int)
            payment = np.sum(realizedFlow*self.averagePayments, axis=2)
            demands.append(totalD[self.t:self.t+future_step+1])
            passenger_flow_ratios.append(realizedFlow[self.t:self.t+future_step+1])
            trip_pays.append(payment[self.t:self.t+future_step+1])
        return demands, passenger_flow_ratios, trip_pays

    def real_demand_generating(self):
        demands = []
        passenger_flow_ratios = []
        trip_pays = []
        future_step = 0
        for sample in range(1):
            realizedFlow = self.realizedFlow#D.astype(float) / np.maximum(totalD[:, :, np.newaxis], 0.0001)
            payment = np.sum(realizedFlow*self.averagePayments, axis=2)
            demands.append(self.totalD[self.t:self.t+future_step+1])
            passenger_flow_ratios.append(realizedFlow[self.t:self.t+future_step+1])
            trip_pays.append(payment[self.t:self.t+future_step +1])
        return demands, passenger_flow_ratios, trip_pays




    @property
    def observation_space(self):
        return Box(low=self.low, high=self.high, shape=(self.h + self.stateNum*2))

    @property
    def action_space(self):
        components = []
        for i in range(self.stateNum):
            components.append(Box(low=0, high=self.high, shape=(self.actionNum)))
        return Product(components)


    def step(self, prob):
        prob = probScale(np.asarray(prob, dtype=float))
        a = np.asarray(
            [np.random.multinomial(self.S[i], prob[i]) for i in range(prob.shape[0])])
        temp1 = np.zeros((self.stateNum, self.stateNum, self.stateNum), float)
        tempAction = a
        tempDests = np.sum(tempAction, axis=0).astype(float)
        totalD = self.totalD[self.t]
        servedPassengerTotal = np.minimum(self.totalD[self.t], tempDests).astype(int)
        passengerFlows = np.asarray(
            [np.random.multinomial(servedPassengerTotal[i], self.realizedFlow[self.t, i]) for i in range(a.shape[0])],
            float)
        pickUpProbs = servedPassengerTotal / np.maximum(tempDests, 0.0001)
        tempNoPick = (1 - pickUpProbs[np.newaxis, :]) * tempAction[:, :]
        passengerFlows = passengerFlows / np.maximum(servedPassengerTotal, 0.0001)[:, np.newaxis]
        statePayments = np.sum(
            self.averagePayments[self.t] * passengerFlows, axis=1)
        shortage = self.selected_zones_multipliers*np.minimum(tempDests - self.totalD[self.t] * self.satisfied_percentage,
                             0)
        missing = self.selected_zones_multipliers*np.minimum(tempDests - self.totalD[self.t],
                             0)
        ratio = np.mean(np.maximum(servedPassengerTotal[self.high_demand_zones],1e-8) / np.maximum(totalD[self.high_demand_zones], 1e-8))
        action_rewards = pickUpProbs[np.newaxis, :] * statePayments[np.newaxis, :] - self.zero_costs
        penalty = self.penalty_weights*shortage
        trip_pay = statePayments * servedPassengerTotal
        local_trip_pay = trip_pay[self.high_demand_zones]
        immediateReward = penalty + trip_pay
        # taxis without passenger
        for s in range(self.stateNum):  # immediateReward[:, s] = tempPayment[:,s]
            temp1[:, s, s] = tempNoPick[:, s]
        # taxis follow passenger flow
        temp1 += tempAction[:, :, np.newaxis] * passengerFlows[np.newaxis, :, :] * pickUpProbs[np.newaxis, :,
                                                                                   np.newaxis]
        self.S = np.array(np.rint(np.sum(temp1, axis=(0, 1))), int)
        self.state = self.S
        reward = np.sum(immediateReward)  # * temp1.sum(axis=2)
        # print('revenue:' + str(reward - self.penalty_weight * np.sum(
        #     np.minimum(tempDests - self.totalD[self.t] * self.satisfied_percentage,
        #                0))))
        self.t += 1
        # self._state = np.dstack((self.S[:self.stateNum], np.zeros(self.stateNum)))[0]
        SAS = temp1  # remainingPassenger = d
        obs = np.zeros(self.h + self.stateNum * 2)
        if self.t < 48:
            obs[self.h:self.stateNum + self.h] = self.S
            obs[self.stateNum + self.h:] = self.totalD[self.t]
            obs /= float(self.N)
            obs[self.t] = 1
        return obs, a, reward, False, {'state': self.S, 'count': SAS, 'ratio':ratio, 'shortage':shortage, 'missing':missing,  'penalty': penalty, 'trip_pay': trip_pay, 'action_rewards':action_rewards, 'local_trip_pay':local_trip_pay}#

    def _close(self):
        self.state = None
