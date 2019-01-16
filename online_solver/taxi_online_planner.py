#!/usr/bin/python

# Copyright 2017, Gurobi Optimization, Inc.

# This example formulates and solves the following simple MIP model:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#  x, y, z binary

from gurobipy import *
import numpy as np

class TaxiPlannner(object):
    def __init__(self, state_num, next_state_num, N, not_adjacent_list):
        self.state_num = state_num
        self.next_state_num = next_state_num
        self.N = N
        self.not_adjacent_list = not_adjacent_list

    def solve(self, state_count, cost_rate, trip_weights, demand, passenger_flow_ratio, future_trip_weights, future_UB):
        # Create a new model
        m = Model("Demand")


        # action count variable
        n_a = m.addVars(self.state_num,self.next_state_num,lb=0, ub=self.N, vtype=GRB.CONTINUOUS)

        for i in range(self.state_num):
            m.addConstrs(n_a[i, j] == 0 for j in self.not_adjacent_list[i])

        #incoming state count
        n_s = m.addVars(self.next_state_num,lb=0, ub=self.N, vtype=GRB.CONTINUOUS)
        served_demand = m.addVars(self.next_state_num,lb=0, ub=self.N, vtype=GRB.CONTINUOUS)
        m.addConstrs(n_a.sum(i, '*') == state_count[i] for i in range(self.state_num))
        m.addConstrs(n_a.sum('*', j) == n_s[j] for j in range(self.next_state_num))

        #for j in range(self.next_state_num):
        #    m.addGenConstrMin(served_demand[j], [demand[j], n_s[j]])
        m.addConstrs(served_demand[j] <=  demand[j] for j in range(self.next_state_num))
        m.addConstrs(served_demand[j] <= n_s[j] for j in range(self.next_state_num))

        customer_flows = m.addVars(self.next_state_num,self.next_state_num,lb=0, ub=self.N, vtype=GRB.CONTINUOUS)
        m.addConstrs(customer_flows[i,j] == served_demand[i]*passenger_flow_ratio[i,j] for i in range(self.next_state_num) for j in range(self.next_state_num))

        #next state count
        n_p = m.addVars(self.next_state_num, lb=0, ub=self.N, vtype=GRB.CONTINUOUS)
        m.addConstrs(n_p[i] == n_s[i] - served_demand[i] +customer_flows.sum(i, '*') for i in range(self.next_state_num))

        #next state rewards
        R_p = m.addVars(self.next_state_num, lb=0, ub=future_UB, vtype=GRB.CONTINUOUS)
        m.addConstrs(R_p[i]<=n_p[i]*future_trip_weights[i] for i in range(self.next_state_num))

        # Set objective
        cost = quicksum(n_a[i,j]*cost_rate[i,j] for i in range(self.state_num) for j in range(self.next_state_num))
        immediate_reward = quicksum(served_demand[j]*trip_weights[j] for j in range(self.next_state_num))
        next_state_reward = quicksum(R_p)
        m.setObjective(immediate_reward + cost + next_state_reward, GRB.MAXIMIZE)
        #
        # # Add constraint: x + 2 y + 3 z <= 4
        # m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
        #
        # # Add constraint: x + y >= 1
        # m.addConstr(x + y >= 1, "c1")
        m.setParam("OutputFlag", 0)

        m.optimize()

        # for v in m.getVars():
        #     print('%s %g' % (v.varName, v.x))

       # print('Obj: %g' % m.objVal)
        sol = np.array([[n_a[i,j].X for j in range(self.next_state_num)] for i in range(self.state_num)])
        return sol

def main():
    state_num = 81
    N = 8000.0
    next_state_num = 81
    testing_solver = TaxiPlannner(81, 81, 8000.0)
    state_count = np.array(np.random.random(state_num) * N, dtype=int)
    cost = np.random.random((state_num, next_state_num)) * 10
    demand = np.random.random(state_num) * 800
    passenger_flow_ratio = np.random.random((next_state_num, next_state_num))
    passenger_flow_ratio = passenger_flow_ratio / np.sum(passenger_flow_ratio, axis=1)[:, np.newaxis]
    trip_weights = np.random.random(next_state_num)
    future_trip_weights = np.random.random(next_state_num) * 2
    future_UB = np.random.random(next_state_num) * 300
    testing_solver.solve(state_count, cost, trip_weights, demand, passenger_flow_ratio, future_trip_weights, future_UB)

if __name__ == '__main__':
    main()
# except GurobiError as e:
#     print('Error code ' + str(e.errno) + ": " + str(e))

# except AttributeError:
#     print('Encountered an attribute error')
