import numpy as np
from os.path import dirname, abspath
from baselines.spaces.product import Product
import gym
from baselines.spaces.box import Box
from baselines.special import probScale
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
import pygame

from gym import spaces, utils

TileMargin = 4

BLACK = (0, 0, 0)                             #some color definitions
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255,255,0)
MAGENTA = (255,0,255)

class PatrollingGame(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, N, initialDistribution, victimNum, edge1 = 9, edge2 = 9, delay_penalty = -1, rescue_reward = 1):
        self.display = False
        self.stateNum = edge1*edge2
        self.actionNum = 5
        self.N = N
        self.victimNum = victimNum
        self.remaining = 0
        # self.N = 1
        self.delay_penalty = delay_penalty
        self.rescue_reward = rescue_reward

        stateIndex = np.array(range(self.stateNum)).reshape(edge1, edge2)
        actionIndex = np.array([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]])
        self.transitedState = np.tile(range(self.stateNum), (self.actionNum, 1))
        self.incomeFlows = [[] for i in range(self.stateNum)]
        self.decodeState = {}
        for s1 in range(edge1):
            for s2 in range(edge2):
                self.decodeState.update({stateIndex[s1, s2]:[s1, s2]})

        self.distances = np.zeros((self.stateNum, self.stateNum))
        for s1 in range(self.stateNum):
            for s2 in range(self.stateNum):
                coord1 = self.decodeState[s1]
                coord2 = self.decodeState[s2]
                self.distances[s1, s2] = abs(coord1[0] - coord2[0]) + abs((coord1[1] - coord2[1]))

        for a in range(0, self.actionNum):
            for s1 in range(edge1):
                for s2 in range(edge2):
                    if ((s1 + actionIndex[a, 0]) in range(edge1)) and ((s2 + actionIndex[a, 1]) in range((edge2))):
                        self.transitedState[a, stateIndex[s1, s2]] = stateIndex[
                            s1 + actionIndex[a, 0], s2 + actionIndex[a, 1]]
                        self.incomeFlows[stateIndex[
                            s1 + actionIndex[a, 0], s2 + actionIndex[a, 1]]].append([stateIndex[s1, s2], a])
                    else:
                        self.incomeFlows[stateIndex[s1, s2]].append([stateIndex[s1, s2], a])


        self.initialDistribution = initialDistribution
        self.target_distribution = np.ones(self.stateNum, dtype=float)/float(self.stateNum)

        self.low = 0
        self.high = self.N
        tempA = np.zeros((self.stateNum, 2)).astype(int)
        tempA[:, 1] = self.actionNum - 1
        # self.shape = (self.stateNum, self.actionNum)
        self.goalState = self.stateNum - 1
        self.edge1 = edge1
        self.edge2 = edge2
        self.neighbors = {}
        for s1 in range(self.edge1):
            for s2 in range(self.edge2):
                s = stateIndex[s1, s2]
                tempNeighbor = []
                for a in range(0, self.actionNum):
                    if ((s1 + actionIndex[a, 0]) in range(self.edge1)) and (
                                (s2 + actionIndex[a, 1]) in range((self.edge2))):
                        tempNeighbor.append(stateIndex[s1 + actionIndex[a, 0], s2 + actionIndex[a, 1]])
                self.neighbors.update({s: tempNeighbor})
        if self.display:
            nR = 5
            nC = 5
            self.X = nR
            self.Y = nC
            self.locsNum = self.X * self.Y
            self.fps = 60
            self.clock = pygame.time.Clock()
            pygame.init()
            clock = pygame.time.Clock()

            self.TileWidth = 50  # width/float(self.X)  # pixel sizes for grid squares
            self.TileHeight = 50  # height/float(self.Y)

            width = self.TileWidth * self.X
            height = self.TileHeight * self.Y
            self.screen_dim = (width, height)
            self.radius = int(self.TileWidth / 3)
            self.resolution = 1
            pygame.display.set_caption('Grid simulation')
            self.screen = pygame.display.set_mode(self.screen_dim, 0, 32)
            self.font = pygame.font.SysFont('Arial', int(self.TileWidth / 4))
            self.stateIndex = stateIndex

    def reset(self):
        self.t = 0
        self.state = np.random.multinomial(self.N, self.initialDistribution)
        self.targets = np.random.multinomial(self.victimNum, self.target_distribution)
        obs = np.zeros((3, self.stateNum))  # agent loc, target loc, coverage area
        obs[0] = self.state/ float(self.N)
        coverage_area = np.zeros(self.stateNum)
        for s in range(self.stateNum):
            if self.targets[s] > 0:
                for sp in self.neighbors[s]:
                    coverage_area[sp] = 1.0
        obs[1] = coverage_area
        obs[2] = self.targets
        obs = obs.flatten()


        self.state_count = np.tile(self.state[:, np.newaxis], [1, self.actionNum]).flatten()
        if self.display:
            self._render()
        return obs, self.state_count


    @property
    def _state(self):
        return self.state

    @property
    def observation_space(self):
        return Box(low=self.low, high=self.high, shape=(self.stateNum*3))

    @property
    def action_space(self):
        components = []
        for i in range(self.stateNum):
            components.append(Box(low=0, high=self.high, shape=(self.actionNum)))
        return Product(components)

    def step(self, prob):
        # prob = self.action_space.unflatten(prob)
        prob = probScale(np.asarray(prob, dtype=float))
        a = np.asarray(
            [np.random.multinomial(self.state[i], prob[i]) for i in range(prob.shape[0])])
        SAS = np.zeros((self.stateNum, self.actionNum, self.stateNum))
        self.S = np.zeros(self.stateNum)
        for i in range(self.stateNum):
            for j in range(self.actionNum):
                self.S[self.transitedState[j, i]] += a[i, j]
                SAS[i, j, self.transitedState[j, i]] = a[i,j]



        rescued_victims = np.minimum(self.S, self.targets)
        #rescued_victims = np.array(np.minimum(np.rint(self.S / 3), self.targets), dtype=int)
        self.targets =  self.targets - rescued_victims
        #penalty = self.targets*self.delay_penalty
        rescues = rescued_victims*self.rescue_reward
        rewards = rescues #+ penalty
        state_rewards = rewards/np.maximum(1.0, self.S)
        action_rewards = np.zeros((self.stateNum, self.actionNum))
        for i in range(self.stateNum):
            for j in range(self.actionNum):
                action_rewards[i, j] = state_rewards[self.transitedState[j, i]]


        reward = np.sum(rewards)
        self.state = self.S
        total_target_num = np.sum(self.targets)
        if total_target_num < self.victimNum:
            self.targets += np.random.multinomial(self.victimNum - total_target_num, self.target_distribution)
        obs = np.zeros((3, self.stateNum))#agent loc, target loc, coverage area
        obs[0] = self.state/ float(self.N)
        coverage_area = np.zeros(self.stateNum)
        for s in range(self.stateNum):
            if self.targets[s] > 0:
                for sp in self.neighbors[s]:
                    coverage_area[sp] = 1.0
        obs[1] = coverage_area
        obs[2] = self.targets
        obs = obs.flatten()

        if self.display:
            self._render()
        self.t += 1
        return obs.flatten(), a, reward, False, {'state': self.state, 'count': SAS,
                                                  'rescues':rescues, 'rewards':action_rewards}  # 'penalty': penalty,self.state_count,

    def _close(self):
        self.state = None


    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def _render(self, mode='human', close=False):
        if close:
            return
        self.screen.fill(WHITE)

        for Row in range(self.Y+1):  # Drawing grid
            pygame.draw.line(self.screen, BLACK,(0,Row*self.TileHeight),(self.screen_dim[0],Row*self.TileHeight), self.resolution)
        for Column in range(self.X + 1):
            pygame.draw.line(self.screen, BLACK, (Column * self.TileHeight,0),
                             (Column * self.TileHeight, self.screen_dim[1]), self.resolution)

        self.clock.tick(self.fps)

        for x in range(self.edge1):#row
            for y in range(self.edge2):#column
                s = self.stateIndex[x, y]
                # if self.state[s]>0:
                #     pygame.draw.rect(self.screen, BLACK, (
                #         y * self.TileWidth, x * self.TileHeight, (y + 1) * self.TileWidth,
                #         (x + 1) * self.TileHeight))
                if self.targets[s]>0:
                    pygame.draw.circle(self.screen, YELLOW, (y * self.TileWidth + int(self.TileWidth / 2),
                                                             x * self.TileHeight + int(self.TileHeight / 2)),
                                       int(self.TileWidth / 3.0))
                    self.screen.blit(self.font.render('state ' +str(s) + ' time ' +str(self.t), True, BLACK), (
                        y * self.TileWidth + int(self.TileWidth / 2),
                        x * self.TileHeight + int(self.TileHeight / 2)))
                if self.state[s]>0:
                    self.screen.blit(self.font.render(str(self.state[s]), True, (255, 0, 0)), (
                    y * self.TileWidth + int(self.TileWidth / 4),
                    x * self.TileHeight + int(self.TileHeight / 3)))



        pygame.display.update()

