import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from baselines.common.minout import minout
from tensorflow.python.framework import ops

class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

class TaxiCentralizedCollectiveCriticWithCostAndBiasVPN(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=True,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, satisfied_percentage = 0.1):
        super(TaxiCentralizedCollectiveCriticWithCostAndBiasVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.N = 8000.0
        self.satisfied_percentage = satisfied_percentage

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            obs = obs*self.N
            action_count = action_count*self.N

            output = []
            inputLayer = obs
            time_period = obs[:, :self.H]#*self.N


            next_state_count = []
            normalizedDenseCounter = 0

            state_count = obs[:, self.H: self.H + self.zoneNum ]
            demand_count = obs[:, self.H + self.zoneNum:]
            normalizedDenseCounter +=1
            #payment
            immediateRewards = []
            future_vals = []
            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)

            served_demands = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                stateReward = tf.minimum(x - demand_count[:,i]*self.satisfied_percentage, 0.0)
                output.append(stateReward)
                immediateRewards.append(stateReward)
                served_demands.append(served_demand)

            immediateRewards = tf.stack(immediateRewards, axis=1)

            # predict passenger flow
            served_demands = tf.stack(served_demands, axis=1)


            initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)

            V = tf.layers.dense(time_period,int(state_count.get_shape()[1])* self.zoneNum, kernel_initializer=initializer)
            V = tf.reshape(V, [-1, int(state_count.get_shape()[1]), self.zoneNum])
            # V = tf.get_variable('V_' + str(normalizedDenseCounter), [int(state_count.get_shape()[1]), self.zoneNum],
            #                     tf.float32, initializer,
            #                     trainable=True)
            V_norm = tf.nn.softmax(V, dim=2)
            customer_flow = tf.matmul(tf.reshape(served_demands, [-1, 1, self.zoneNum]), V_norm)[:,0,:]
            # predict the next upper bound
            d = tf.layers.dense(time_period, self.zoneNum, kernel_initializer=tf.random_uniform_initializer(minval=100, maxval=self.N), name= "d", use_bias=False)

            # predict the next threshold
            TH = tf.layers.dense(time_period, self.zoneNum,
                                kernel_initializer=tf.random_uniform_initializer(minval=100, maxval=self.N), name="TH",
                                use_bias=False)

            # d = tf.get_variable("d", [self.zoneNum], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
            # predict the next payment
            future_trip_weights = tf.layers.dense(time_period, len(self.nb_actions), kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3), use_bias=False)
            # future_trip_weights = tf.get_variable('rd', [len(self.nb_actions)], tf.float32,
            #                                       tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
            #                                       trainable=True)

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                #future reward
                next_x = x - served_demands[:,i] + customer_flow[:,i]
                #second piecewise
                future_val = tf.multiply(tf.minimum(next_x - TH[:,i], 0), future_trip_weights[:,i])
            #    future_val = tf.minimum(p, d[:,i])
                next_state_count.append(next_x)
                # future_val = tf.multiply(future_served_demand, future_trip_weights[:,i])
                future_vals.append(future_val)
                output.append(future_val)

            future_vals = tf.stack(future_vals, axis=1)

            output = tf.stack(output, axis=1)

            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'V_norm':V_norm, 'customer_flow':customer_flow, 'immediateRewards':immediateRewards, 'd':d, 'future_trip_weights':future_trip_weights, 'trip_weights':trip_weights, 'next_vals':future_vals, 'next_state_count':next_state_count}



class TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndBiasVPN(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=True,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, satisfied_percentage = 0.1):
        super(TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndBiasVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.N = 8000.0
        self.satisfied_percentage = satisfied_percentage

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            obs = obs*self.N
            action_count = action_count*self.N

            output = []
            inputLayer = obs
            time_period = obs[:, :self.H]#*self.N


            next_state_count = []
            normalizedDenseCounter = 0

            state_count = obs[:, self.H: self.H + self.zoneNum ]
            demand_count = obs[:, self.H + self.zoneNum:]
            normalizedDenseCounter +=1
            #payment
            immediateRewards = []
            future_vals = []
            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)

            served_demands = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                stateReward = tf.minimum(x - demand_count[:,i]*self.satisfied_percentage, 0.0) + tf.multiply(served_demand,trip_weights[i])
                output.append(stateReward)
                immediateRewards.append(stateReward)
                served_demands.append(served_demand)

            immediateRewards = tf.stack(immediateRewards, axis=1)

            # predict passenger flow
            served_demands = tf.stack(served_demands, axis=1)


            initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)

            V = tf.layers.dense(time_period,int(state_count.get_shape()[1])* self.zoneNum, kernel_initializer=initializer)
            V = tf.reshape(V, [-1, int(state_count.get_shape()[1]), self.zoneNum])
            # V = tf.get_variable('V_' + str(normalizedDenseCounter), [int(state_count.get_shape()[1]), self.zoneNum],
            #                     tf.float32, initializer,
            #                     trainable=True)
            V_norm = tf.nn.softmax(V, dim=2)
            customer_flow = tf.matmul(tf.reshape(served_demands, [-1, 1, self.zoneNum]), V_norm)[:,0,:]#tf.matmul(served_demands, V_norm)


            # predict the next threshold for penalty
            TH = tf.layers.dense(time_period, self.zoneNum,
                                kernel_initializer=tf.random_uniform_initializer(minval=100, maxval=self.N), name="TH",
                                use_bias=False)

            # d = tf.get_variable("d", [self.zoneNum], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
            # predict the next payment
            future_trip_penalty = tf.layers.dense(time_period, len(self.nb_actions), kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3), use_bias=False)

            future_trip_weights= tf.layers.dense(time_period, len(self.nb_actions),
                              kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3), use_bias=False)

            # predict the next upper bound
            d = tf.layers.dense(time_period, self.zoneNum,
                                kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=self.N), name="d",
                                use_bias=False)

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                #future reward
                next_x = x - served_demands[:,i] + customer_flow[:,i]
                #second piecewise
                future_val = tf.multiply(tf.minimum(next_x - TH[:,i], 0), future_trip_penalty[:,i]) + tf.multiply(next_x, future_trip_weights[:,i])
                future_val = tf.minimum(future_val, d[:,i])
                next_state_count.append(next_x)
                # future_val = tf.multiply(future_served_demand, future_trip_weights[:,i])
                future_vals.append(future_val)
                output.append(future_val)

            future_vals = tf.stack(future_vals, axis=1)

            output = tf.stack(output, axis=1)

            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'V_norm':V_norm, 'customer_flow':customer_flow, 'immediateRewards':immediateRewards, 'd':d, 'future_trip_weights':future_trip_weights, 'trip_weights':trip_weights, 'next_vals':future_vals, 'next_state_count':next_state_count}


class TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndBiasVPNSeparatePR(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=True,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, satisfied_percentage = 0.1):
        super(TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndBiasVPNSeparatePR, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.N = 8000.0
        self.satisfied_percentage = satisfied_percentage

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            obs = obs*self.N
            action_count = action_count*self.N

            output = []
            inputLayer = obs
            time_period = obs[:, :self.H]#*self.N


            next_state_count = []
            normalizedDenseCounter = 0

            state_count = obs[:, self.H: self.H + self.zoneNum ]
            demand_count = obs[:, self.H + self.zoneNum:]
            normalizedDenseCounter +=1
            #payment
            immediateRevenues = []

            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)

            served_demands = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                revenue = tf.multiply(served_demand,trip_weights[i])
                stateReward = tf.minimum(x - demand_count[:,i]*self.satisfied_percentage, 0.0) + revenue
                output.append(stateReward)
                immediateRevenues.append(revenue)
                served_demands.append(served_demand)

            immediateRevenues = tf.stack(immediateRevenues, axis=1)

            # predict passenger flow
            served_demands = tf.stack(served_demands, axis=1)


            initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)

            V = tf.layers.dense(time_period,int(state_count.get_shape()[1])* self.zoneNum, kernel_initializer=initializer)
            V = tf.reshape(V, [-1, int(state_count.get_shape()[1]), self.zoneNum])
            # V = tf.get_variable('V_' + str(normalizedDenseCounter), [int(state_count.get_shape()[1]), self.zoneNum],
            #                     tf.float32, initializer,
            #                     trainable=True)
            V_norm = tf.nn.softmax(V, dim=2)
            customer_flow = tf.matmul(tf.reshape(served_demands, [-1, 1, self.zoneNum]), V_norm)[:,0,:]#tf.matmul(served_demands, V_norm)


            # predict the next threshold for penalty
            TH = tf.layers.dense(time_period, self.zoneNum,
                                kernel_initializer=tf.random_uniform_initializer(minval=100, maxval=self.N), name="TH",
                                use_bias=False)

            # d = tf.get_variable("d", [self.zoneNum], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
            # predict the next payment
            future_trip_penalty = tf.layers.dense(time_period, len(self.nb_actions), kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3), use_bias=False)

            future_trip_weights= tf.layers.dense(time_period, len(self.nb_actions),
                              kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3), use_bias=False)

            # predict the next upper bound
            d = tf.layers.dense(time_period, self.zoneNum,
                                kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=self.N), name="d",
                                use_bias=False)

            future_revenues = []
            future_penalty = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                #future reward
                next_x = x - served_demands[:,i] + customer_flow[:,i]
                #second piecewise
                penalty = tf.multiply(tf.minimum(next_x - TH[:,i], 0), future_trip_penalty[:,i])
                revenue = tf.minimum(tf.multiply(next_x, future_trip_weights[:,i]), d[:,i])
                future_revenues.append(revenue)
                future_penalty.append(penalty)
                future_val = penalty + revenue
                next_state_count.append(next_x)
                # future_val = tf.multiply(future_served_demand, future_trip_weights[:,i])
                output.append(future_val)

            future_revenues = tf.stack(future_revenues, axis=1)
            future_penalty = tf.stack(future_penalty, axis=1)

            output = tf.stack(output, axis=1)

            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val': x, 'V_norm': V_norm, 'customer_flow': customer_flow,
                'immediateRevenues': immediateRevenues, 'd': d, 'future_trip_weights': future_trip_weights,
                'trip_weights': trip_weights, 'future_revenues': future_revenues, 'future_penalty': future_penalty,
                'next_state_count': next_state_count}




class TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndBiasDenseVPN(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=True,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, satisfied_percentage = 0.1):
        super(TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndBiasDenseVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.N = 8000.0
        self.satisfied_percentage = satisfied_percentage

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            obs = obs*self.N
            action_count = action_count*self.N

            output = []
            inputLayer = obs
            time_period = obs[:, :self.H]#*self.N


            next_state_count = []
            normalizedDenseCounter = 0

            state_count = obs[:, self.H: self.H + self.zoneNum ]
            demand_count = obs[:, self.H + self.zoneNum:]
            normalizedDenseCounter +=1
            #payment
            immediateRevenues = []

            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)

            served_demands = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                revenue = tf.multiply(served_demand,trip_weights[i])
                stateReward = tf.minimum(x - demand_count[:,i]*self.satisfied_percentage, 0.0) + revenue
                output.append(stateReward)
                immediateRevenues.append(revenue)
                served_demands.append(served_demand)

            immediateRevenues = tf.stack(immediateRevenues, axis=1)

            # predict passenger flow
            served_demands = tf.stack(served_demands, axis=1)


            initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)

            V = tf.layers.dense(time_period,int(state_count.get_shape()[1])* self.zoneNum, kernel_initializer=initializer)
            V = tf.reshape(V, [-1, int(state_count.get_shape()[1]), self.zoneNum])
            # V = tf.get_variable('V_' + str(normalizedDenseCounter), [int(state_count.get_shape()[1]), self.zoneNum],
            #                     tf.float32, initializer,
            #                     trainable=True)
            V_norm = tf.nn.softmax(V, dim=2)
            customer_flow = tf.matmul(tf.reshape(served_demands, [-1, 1, self.zoneNum]), V_norm)[:,0,:]#tf.matmul(served_demands, V_norm)


            # predict the next threshold for penalty
            TH = tf.layers.dense(time_period, self.zoneNum,
                                kernel_initializer=tf.random_uniform_initializer(minval=100, maxval=self.N), name="TH",
                                use_bias=False)

            # d = tf.get_variable("d", [self.zoneNum], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
            # predict the next payment
            future_trip_penalty = tf.layers.dense(time_period, len(self.nb_actions), kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3), use_bias=False)

            future_trip_weights= tf.layers.dense(time_period, len(self.nb_actions),
                              kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3), use_bias=False)

            future_trip_penalty_bias = tf.layers.dense(time_period, len(self.nb_actions),
                                                       kernel_initializer=tf.random_uniform_initializer(minval=0,
                                                                                                        maxval=3e-3),
                                                       use_bias=False)

            future_trip_bias = tf.layers.dense(time_period, len(self.nb_actions),
                                               kernel_initializer=tf.random_uniform_initializer(minval=0,
                                                                                                maxval=3e-3),
                                               use_bias=False)

            # predict the next upper bound
            d = tf.layers.dense(time_period, self.zoneNum,
                                kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=self.N), name="d",
                                use_bias=False)

            # next_local_obs = []
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                # future reward
                next_x = x - served_demands[:, i] + customer_flow[:, i]
                next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)

            output = tf.stack(output, axis=1)

            influent_flows = tf.layers.dense(next_state_count, self.zoneNum, kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3))
            future_revenues = tf.minimum(tf.multiply(influent_flows, future_trip_weights) + future_trip_penalty_bias, d)
            # future_penalty = tf.layers.dense(next_state_count, self.zoneNum, kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3))
            future_penalty = tf.multiply(tf.minimum(influent_flows - TH, 0), future_trip_penalty) + future_trip_bias
            output = tf.concat([output, future_revenues, future_penalty], axis=1)



            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'V_norm':V_norm, 'customer_flow':customer_flow, 'immediateRevenues':immediateRevenues, 'd':d, 'future_trip_weights':future_trip_weights, 'trip_weights':trip_weights, 'future_revenues':future_revenues, 'future_penalty':future_penalty, 'next_state_count':next_state_count}

class TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndBiasNonLinADenseVPN(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, satisfied_percentage = 0.1):
        super(TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndBiasNonLinADenseVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.N = 8000.0
        self.satisfied_percentage = satisfied_percentage

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            obs = obs*self.N
            action_count = action_count*self.N

            output = []
            inputLayer = obs
            time_period = obs[:, :self.H]#*self.N


            next_state_count = []
            normalizedDenseCounter = 0

            state_count = obs[:, self.H: self.H + self.zoneNum ]
            demand_count = obs[:, self.H + self.zoneNum:]
            normalizedDenseCounter +=1
            #payment
            immediateRevenues = []

            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)

            served_demands = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                revenue = tf.multiply(served_demand,trip_weights[i])
                stateReward = tf.minimum(x - demand_count[:,i]*self.satisfied_percentage, 0.0) + revenue
                output.append(stateReward)
                immediateRevenues.append(revenue)
                served_demands.append(served_demand)

            immediateRevenues = tf.stack(immediateRevenues, axis=1)

            # predict passenger flow
            served_demands = tf.stack(served_demands, axis=1)


            initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)

            V = tf.layers.dense(time_period,int(state_count.get_shape()[1])* self.zoneNum, kernel_initializer=initializer)
            V = tf.reshape(V, [-1, int(state_count.get_shape()[1]), self.zoneNum])
            # V = tf.get_variable('V_' + str(normalizedDenseCounter), [int(state_count.get_shape()[1]), self.zoneNum],
            #                     tf.float32, initializer,
            #                     trainable=True)
            V_norm = tf.nn.softmax(V, dim=2)
            customer_flow = tf.matmul(tf.reshape(served_demands, [-1, 1, self.zoneNum]), V_norm)[:,0,:]#tf.matmul(served_demands, V_norm)



            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                # future reward
                next_x = x - served_demands[:, i] + customer_flow[:, i]
                next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)

            output = tf.stack(output, axis=1)

            future_revenues = []
            future_penalty = []

            future_trip_penalty = tf.layers.dense(time_period, len(self.nb_actions),
                                                  kernel_initializer=tf.random_uniform_initializer(minval=0,
                                                                                                   maxval=3e-3),
                                                  use_bias=False)



            future_trip_weights = tf.layers.dense(time_period, len(self.nb_actions),
                                                  kernel_initializer=tf.random_uniform_initializer(minval=0,
                                                                                                   maxval=3e-3),
                                                  use_bias=False)

            future_trip_penalty_bias = tf.layers.dense(time_period, len(self.nb_actions),
                                                       kernel_initializer=tf.random_uniform_initializer(minval=0,
                                                                                                        maxval=3e-3),
                                                       use_bias=False)

            future_trip_bias = tf.layers.dense(time_period, len(self.nb_actions),
                                                  kernel_initializer=tf.random_uniform_initializer(minval=0,
                                                                                                   maxval=3e-3),
                                                  use_bias=False)



            for i in range(len(self.nb_actions)):
                x = next_state_count
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    # y.append(x)
                x = tf.layers.dense(x, 2)
                future_revenues.append(tf.multiply(x[:,0], future_trip_weights[:,i]) + future_trip_bias[:,i])
                future_penalty.append(tf.multiply(x[:,1], future_trip_penalty[:,i]) + future_trip_penalty_bias[:,i])



            future_revenues = tf.stack(future_revenues, axis=1)
            future_penalty = tf.stack(future_penalty, axis=1)


            output = tf.concat([output, future_revenues, future_penalty], axis=1)



            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'V_norm':V_norm, 'customer_flow':customer_flow, 'immediateRevenues':immediateRevenues, 'trip_weights':trip_weights, 'future_revenues':future_revenues, 'future_penalty':future_penalty, 'next_state_count':next_state_count}



#========================================================================================================================
class TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndDenseVPN(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=True,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, satisfied_percentage = 0.1):
        super(TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndDenseVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.N = 8000.0
        self.satisfied_percentage = satisfied_percentage

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            obs = obs*self.N
            action_count = action_count*self.N

            output = []
            inputLayer = obs
            time_period = obs[:, :self.H]#*self.N


            next_state_count = []
            normalizedDenseCounter = 0

            state_count = obs[:, self.H: self.H + self.zoneNum ]
            demand_count = obs[:, self.H + self.zoneNum:]
            normalizedDenseCounter +=1
            #payment
            immediateRevenues = []

            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)

            served_demands = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                revenue = tf.multiply(served_demand,trip_weights[i])
                stateReward = tf.minimum(x - demand_count[:,i]*self.satisfied_percentage, 0.0) + revenue
                output.append(stateReward)
                immediateRevenues.append(revenue)
                served_demands.append(served_demand)

            immediateRevenues = tf.stack(immediateRevenues, axis=1)

            # predict passenger flow
            served_demands = tf.stack(served_demands, axis=1)


            initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)

            V = tf.layers.dense(time_period,int(state_count.get_shape()[1])* self.zoneNum, kernel_initializer=initializer)
            V = tf.reshape(V, [-1, int(state_count.get_shape()[1]), self.zoneNum])
            # V = tf.get_variable('V_' + str(normalizedDenseCounter), [int(state_count.get_shape()[1]), self.zoneNum],
            #                     tf.float32, initializer,
            #                     trainable=True)
            V_norm = tf.nn.softmax(V, dim=2)
            customer_flow = tf.matmul(tf.reshape(served_demands, [-1, 1, self.zoneNum]), V_norm)[:,0,:]#tf.matmul(served_demands, V_norm)


            # predict the next threshold for penalty
            TH = tf.layers.dense(time_period, self.zoneNum,
                                kernel_initializer=tf.random_uniform_initializer(minval=100, maxval=self.N), name="TH",
                                use_bias=False)

            # d = tf.get_variable("d", [self.zoneNum], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
            # predict the next payment
            future_trip_penalty = tf.layers.dense(time_period, len(self.nb_actions), kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3), use_bias=False)

            future_trip_weights= tf.layers.dense(time_period, len(self.nb_actions),
                              kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3), use_bias=False)


            # predict the next upper bound
            d = tf.layers.dense(time_period, self.zoneNum,
                                kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=self.N), name="d",
                                use_bias=False)

            # next_local_obs = []
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                # future reward
                next_x = x - served_demands[:, i] + customer_flow[:, i]
                next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)

            output = tf.stack(output, axis=1)

            influent_flows = tf.layers.dense(next_state_count, self.zoneNum, kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3))
            future_revenues = tf.minimum(tf.multiply(influent_flows, future_trip_weights), d)
            # future_penalty = tf.layers.dense(next_state_count, self.zoneNum, kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3))
            future_penalty = tf.multiply(tf.minimum(influent_flows - TH, 0), future_trip_penalty)
            output = tf.concat([output, future_revenues, future_penalty], axis=1)



            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'V_norm':V_norm, 'customer_flow':customer_flow, 'immediateRevenues':immediateRevenues, 'd':d, 'future_trip_weights':future_trip_weights, 'trip_weights':trip_weights, 'future_revenues':future_revenues, 'future_penalty':future_penalty, 'next_state_count':next_state_count}

class TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndNonLinADenseVPN(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, satisfied_percentage = 0.1):
        super(TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndNonLinADenseVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.N = 8000.0
        self.satisfied_percentage = satisfied_percentage

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            obs = obs*self.N
            action_count = action_count*self.N

            output = []
            inputLayer = obs
            time_period = obs[:, :self.H]#*self.N


            next_state_count = []
            normalizedDenseCounter = 0

            state_count = obs[:, self.H: self.H + self.zoneNum ]
            demand_count = obs[:, self.H + self.zoneNum:]
            normalizedDenseCounter +=1
            #payment
            immediateRevenues = []

            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)

            served_demands = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                revenue = tf.multiply(served_demand,trip_weights[i])
                stateReward = tf.minimum(x - demand_count[:,i]*self.satisfied_percentage, 0.0) + revenue
                output.append(stateReward)
                immediateRevenues.append(revenue)
                served_demands.append(served_demand)

            immediateRevenues = tf.stack(immediateRevenues, axis=1)

            # predict passenger flow
            served_demands = tf.stack(served_demands, axis=1)


            initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)

            V = tf.layers.dense(time_period,int(state_count.get_shape()[1])* self.zoneNum, kernel_initializer=initializer)
            V = tf.reshape(V, [-1, int(state_count.get_shape()[1]), self.zoneNum])
            # V = tf.get_variable('V_' + str(normalizedDenseCounter), [int(state_count.get_shape()[1]), self.zoneNum],
            #                     tf.float32, initializer,
            #                     trainable=True)
            V_norm = tf.nn.softmax(V, dim=2)
            customer_flow = tf.matmul(tf.reshape(served_demands, [-1, 1, self.zoneNum]), V_norm)[:,0,:]#tf.matmul(served_demands, V_norm)



            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                # future reward
                next_x = x - served_demands[:, i] + customer_flow[:, i]
                next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)

            output = tf.stack(output, axis=1)

            future_revenues = []
            future_penalty = []

            future_trip_penalty = tf.layers.dense(time_period, len(self.nb_actions),
                                                  kernel_initializer=tf.random_uniform_initializer(minval=0,
                                                                                                   maxval=3e-3),
                                                  use_bias=False)



            future_trip_weights = tf.layers.dense(time_period, len(self.nb_actions),
                                                  kernel_initializer=tf.random_uniform_initializer(minval=0,
                                                                                                   maxval=3e-3),
                                                  use_bias=False)




            next_state_count = tf.concat([next_state_count, time_period], axis=1)
            for i in range(len(self.nb_actions)):
                x = next_state_count
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    # y.append(x)
                x = tf.layers.dense(x, 1)[:,0]
                future_revenues.append(tf.multiply(x, future_trip_weights[:,i]))

            for i in range(len(self.nb_actions)):
                x = next_state_count
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    # y.append(x)
                x = tf.layers.dense(x, 1)[:,0]
                future_penalty.append(tf.multiply(x, future_trip_penalty[:,i]))



            future_revenues = tf.stack(future_revenues, axis=1)
            future_penalty = tf.stack(future_penalty, axis=1)


            output = tf.concat([output, future_revenues, future_penalty], axis=1)



            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'V_norm':V_norm, 'customer_flow':customer_flow, 'immediateRevenues':immediateRevenues, 'future_trip_penalty': future_trip_penalty, 'future_trip_weights':future_trip_weights, 'trip_weights':trip_weights, 'future_revenues':future_revenues, 'future_penalty':future_penalty, 'next_state_count':next_state_count}

class TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndNonLinADenseVPNNoTime(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, satisfied_percentage = 0.1):
        super(TaxiCentralizedPenaltyandRewardCollectiveCriticWithCostAndNonLinADenseVPNNoTime, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.N = 8000.0
        self.satisfied_percentage = satisfied_percentage

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            obs = obs*self.N
            action_count = action_count*self.N

            output = []
            inputLayer = obs
            time_period = obs[:, :self.H]#*self.N


            next_state_count = []
            normalizedDenseCounter = 0

            state_count = obs[:, self.H: self.H + self.zoneNum ]
            demand_count = obs[:, self.H + self.zoneNum:]
            normalizedDenseCounter +=1
            #payment
            immediateRevenues = []

            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)

            served_demands = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                revenue = tf.multiply(served_demand,trip_weights[i])
                stateReward = tf.minimum(x - demand_count[:,i]*self.satisfied_percentage, 0.0) + revenue
                output.append(stateReward)
                immediateRevenues.append(revenue)
                served_demands.append(served_demand)

            immediateRevenues = tf.stack(immediateRevenues, axis=1)

            # predict passenger flow
            served_demands = tf.stack(served_demands, axis=1)


            initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)

            V = tf.layers.dense(time_period,int(state_count.get_shape()[1])* self.zoneNum, kernel_initializer=initializer)
            V = tf.reshape(V, [-1, int(state_count.get_shape()[1]), self.zoneNum])
            # V = tf.get_variable('V_' + str(normalizedDenseCounter), [int(state_count.get_shape()[1]), self.zoneNum],
            #                     tf.float32, initializer,
            #                     trainable=True)
            V_norm = tf.nn.softmax(V, dim=2)
            customer_flow = tf.matmul(tf.reshape(served_demands, [-1, 1, self.zoneNum]), V_norm)[:,0,:]#tf.matmul(served_demands, V_norm)



            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                # future reward
                next_x = x - served_demands[:, i] + customer_flow[:, i]
                next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)

            output = tf.stack(output, axis=1)

            future_revenues = []
            future_penalty = []


            next_state_count = tf.concat([next_state_count, time_period], axis=1)
            for i in range(len(self.nb_actions)):
                x = next_state_count
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    # y.append(x)
                x = tf.layers.dense(x, 1)[:,0]
                future_revenues.append(x)

            for i in range(len(self.nb_actions)):
                x = next_state_count
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    # y.append(x)
                x = tf.layers.dense(x, 1)[:,0]
                future_penalty.append(x)



            future_revenues = tf.stack(future_revenues, axis=1)
            future_penalty = tf.stack(future_penalty, axis=1)


            output = tf.concat([output, future_revenues, future_penalty], axis=1)



            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'V_norm':V_norm, 'customer_flow':customer_flow, 'immediateRevenues':immediateRevenues, 'trip_weights':trip_weights, 'future_revenues':future_revenues, 'future_penalty':future_penalty, 'next_state_count':next_state_count}

class TaxiCentralizedPenaltyandRewardDeepVPN(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, satisfied_percentage = 0.1, penalty_weight = 1.0, penalty_zones = [], adjacent_list = []):
        super(TaxiCentralizedPenaltyandRewardDeepVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.N = 8000.0
        self.satisfied_percentage = satisfied_percentage
        self.penalty_weight = penalty_weight
        self.penalty_zones = penalty_zones
        self.adjacent_list = adjacent_list

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            obs = obs*self.N
            zero_val = obs[:,0]*0.0
            action_count = action_count#*self.N

            output = []
            inputLayer = obs
            time_period = obs[:, :self.H]#*self.N
            time_period_decoded = tf.cast(self.H - 1 - tf.argmax(time_period, axis=1), dtype=tf.float32)


            next_state_count = []
            normalizedDenseCounter = 0

            state_count = obs[:, self.H: self.H + self.zoneNum ]
            demand_count = obs[:, self.H + self.zoneNum:]
            normalizedDenseCounter +=1
            #payment
            immediateRevenues = []

            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)

            served_demands = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                revenue = tf.multiply(served_demand,trip_weights[i])
                if self.penalty_zones[i]:
                    stateReward = self.penalty_weight*tf.minimum(x - demand_count[:,i]*self.satisfied_percentage, 0.0) + revenue
                else:
                    stateReward = revenue
                output.append(stateReward)
                immediateRevenues.append(revenue)
                served_demands.append(served_demand)

            immediateRevenues = tf.stack(immediateRevenues, axis=1)

            # predict passenger flow
            served_demands = tf.stack(served_demands, axis=1)


            initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)

            V = tf.layers.dense(time_period,int(state_count.get_shape()[1])* self.zoneNum, kernel_initializer=initializer)
            V = tf.reshape(V, [-1, int(state_count.get_shape()[1]), self.zoneNum])
            # V = tf.get_variable('V_' + str(normalizedDenseCounter), [int(state_count.get_shape()[1]), self.zoneNum],
            #                     tf.float32, initializer,
            #                     trainable=True)
            V_norm = tf.nn.softmax(V, dim=2)
            customer_flow = tf.matmul(tf.reshape(served_demands, [-1, 1, self.zoneNum]), V_norm)[:,0,:]#tf.matmul(served_demands, V_norm)



            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                # future reward
                next_x = x - served_demands[:, i] + customer_flow[:, i]
                next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)

            output = tf.stack(output, axis=1)

            future_revenues = []
            future_penalty = []


            next_state_count = tf.concat([next_state_count, time_period], axis=1)
            future_revenue_bound = tf.get_variable("future_revenue_bound", [self.zoneNum], trainable=True,
                                                   initializer=tf.random_uniform_initializer(minval=0, maxval=10))
            future_penalty_bound = tf.get_variable("future_penalty_bound", [self.zoneNum], trainable=True,
                                                   initializer=tf.random_uniform_initializer(minval=-10, maxval=0))
            for i in range(len(self.nb_actions)):
                local_obs = [next_state_count[:, j] for j in self.adjacent_list[i]]
                local_obs = tf.stack(local_obs, axis=1)
                local_obs = tf.concat([local_obs, time_period], axis=1)
                x = local_obs
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    # y.append(x)
                x = tf.minimum(tf.layers.dense(x, 1)[:,0],future_revenue_bound[i])*time_period_decoded
                future_revenues.append(x)
                if self.penalty_zones[i]:
                    x = local_obs
                    for hidden_size in self.hidden_sizes:
                        x = tf.layers.dense(x,
                                            hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                        if self.layer_norm:
                            x = tc.layers.layer_norm(x, center=True, scale=True)
                        if self.batch_norm:
                            x = tc.layers.batch_norm(x)
                        x = self.hidden_nonlinearity(x)
                        # y.append(x)
                    x = tf.minimum(tf.layers.dense(x, 1)[:,0], future_penalty_bound[i])*time_period_decoded*self.penalty_weight
                    future_penalty.append(x)
                else:
                    future_penalty.append(zero_val)



            future_revenues = tf.stack(future_revenues, axis=1)
            future_penalty = tf.stack(future_penalty, axis=1)


            output = tf.concat([output, future_revenues, future_penalty], axis=1)



            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'V_norm':V_norm, 'customer_flow':customer_flow, 'immediateRevenues':immediateRevenues, 'trip_weights':trip_weights, 'future_revenues':future_revenues, 'future_penalty':future_penalty, 'next_state_count':next_state_count}



class TaxiCentralizedInitializationDeepVPN(Model):
    def __init__(self, zone_number, name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(TaxiCentralizedInitializationDeepVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.zone_number = zone_number
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.N = 8000.0

    def __call__(self, zone_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            zone_count = zone_count*self.N



            future_revenues = []
            future_penalty = []

            for i in range(self.zone_number):
                x = zone_count
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    # y.append(x)
                x = tf.layers.dense(x, 1)[:,0]
                future_revenues.append(x)

            for i in range(len(self.nb_actions)):
                x = zone_count
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    # y.append(x)
                x = tf.layers.dense(x, 1)[:,0]
                future_penalty.append(x)



            future_revenues = tf.stack(future_revenues, axis=1)
            future_penalty = tf.stack(future_penalty, axis=1)


            output = tf.concat([future_revenues, future_penalty], axis=1)
            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x,  'future_revenues':future_revenues, 'future_penalty':future_penalty}


class GridDeepVPN(Model):
    def __init__(self, nb_actions, inflows, adjacent_list, name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (32, 32), hidden_nonlinearity = tf.nn.relu, stateNum = 81, N = 20):
        super(GridDeepVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.stateNum = stateNum
        self.N = 1
        self.inflows = inflows
        self.adjacent_list = adjacent_list

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
           # obs = obs*self.N
            action_count = action_count#*self.N

            output = []


            normalizedDenseCounter = 0

            state_count = obs[:, : self.stateNum ]*self.N
            demand_count = obs[:, self.stateNum:self.stateNum*2]
            normalizedDenseCounter +=1
            #payment

            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.ones_initializer(),
                                trainable=True)

            next_state_count = []

            for i in range(self.stateNum):
                temp = []
                for sa in self.inflows[i]:
                    temp.append(action_count[:,sa[0], sa[1]])
                temp = tf.stack(temp, axis=1)
                next_state_count.append(tf.reduce_sum(temp, axis=1))

            next_state_count = tf.stack(next_state_count, axis=1)

            next_coverages = []
            for i in range(self.stateNum):
                cover = tf.stack([next_state_count[:,j] for j in self.adjacent_list[i]], axis=1)
                cover = tf.minimum(tf.reduce_sum(cover, axis=1), 1.0)
                next_coverages.append(cover)

            next_coverages = tf.stack(next_coverages, axis=1)

            served_demands = []
            for i in range(self.stateNum):
                x = next_state_count[:,i]
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                served_demands.append(served_demand) #= tf.multiply(served_demand,trip_weights[i])

            served_demands = tf.stack(served_demands, axis=1)


            #remaining_demands = demand_count - served_demands
            remaining_demands = obs[:,self.stateNum:self.stateNum*2]*tf.minimum(1.0, next_state_count)
            #remaining_demands = tf.reduce_min(remaining_demands, axis=1)

            immediate_reward = tf.reduce_sum(served_demands, axis=1)#tf.layers.dense(served_demands, 1)[:,0]#
            output.append(immediate_reward)


            x = next_state_count#tf.concat([next_state_count, next_coverages, remaining_demands], axis=1)#next_state_count#
            for hidden_size in self.hidden_sizes:
                x = tf.layers.dense(x,
                                    hidden_size, use_bias = False)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                if self.batch_norm:
                    x = tc.layers.batch_norm(x)
                x = self.hidden_nonlinearity(x)
            x = tf.layers.dense(x, 1)[:,0]
            #x = tf.stack([remaining_demands,x], axis=1)
            next_return = x#tf.layers.dense(x, 1)[:,0]

            output.append(next_return)
            output = tf.stack(output, axis=1)
            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'immediate_reward':immediate_reward, 'trip_weights':trip_weights, 'next_return':next_return, 'next_state_count':next_state_count}


class DecGridDeepVPN(Model):
    def __init__(self, nb_actions, inflows, adjacent_list, name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (32, 32), hidden_nonlinearity = tf.nn.relu, stateNum = 81, N = 20):
        super(DecGridDeepVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.stateNum = stateNum
        self.N = N
        self.inflows = inflows
        self.adjacent_list = adjacent_list

    def __call__(self, obs, action_count, local_actions, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
           # obs = obs*self.N
            action_count = action_count#*self.N

            output = []


            normalizedDenseCounter = 0

            state_count = obs[:, : self.stateNum ]*self.N
            demand_count = obs[:, self.stateNum:self.stateNum*2]
            normalizedDenseCounter +=1
            #payment

            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.ones_initializer(),
                                trainable=True)

            next_state_count = []

            for i in range(self.stateNum):
                temp = []
                for sa in self.inflows[i]:
                    temp.append(action_count[:,sa[0], sa[1]])
                temp = tf.stack(temp, axis=1)
                next_state_count.append(tf.reduce_sum(temp, axis=1))

            next_state_count = tf.stack(next_state_count, axis=1)

            next_coverages = []
            for i in range(self.stateNum):
                cover = tf.stack([next_state_count[:,j] for j in self.adjacent_list[i]], axis=1)
                cover = tf.minimum(tf.reduce_sum(cover, axis=1), 1.0)
                next_coverages.append(cover)

            next_coverages = tf.stack(next_coverages, axis=1)

            served_demands = []
            for i in range(self.stateNum):
                x = next_state_count[:,i]
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                served_demands.append(served_demand) #= tf.multiply(served_demand,trip_weights[i])

            served_demands = tf.stack(served_demands, axis=1)


            #remaining_demands = demand_count - served_demands
            remaining_demands = obs[:,self.stateNum:self.stateNum*2]*tf.minimum(1.0, next_state_count)
            #remaining_demands = tf.reduce_min(remaining_demands, axis=1)

            immediate_reward = tf.reduce_sum(served_demands, axis=1, keep_dims=True)#tf.layers.dense(served_demands, 1)[:,0]#
            immediate_reward = tf.tile((immediate_reward), (1, self.N))
            output.append(immediate_reward)

            next_return = []
            i = 0
            local_state = tf.one_hot(tf.cast(obs[:, i + self.stateNum * 2],tf.int32), self.stateNum)
            local_info = tf.concat([next_state_count, local_state, local_actions[:,i], tf.tile(tf.reshape(tf.one_hot(i, self.N), [1, self.N]), (tf.shape(obs)[0],1))], axis=1)
            x = local_info#tf.concat([next_state_count, next_coverages, remaining_demands], axis=1)#next_state_count#
            hidden = 0
            for hidden_size in self.hidden_sizes:
                x = tf.layers.dense(x,
                                    hidden_size, use_bias = False, name='hidden_'+str(hidden))  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True, scope='ln_'+str(hidden))
                if self.batch_norm:
                    x = tc.layers.batch_norm(x)
                x = self.hidden_nonlinearity(x)
                hidden += 1
            x = tf.layers.dense(x, 1, name='hidden_'+str(hidden))[:,0]
            #x = tf.stack([remaining_demands,x], axis=1)
            next_return.append(x)#tf.layers.dense(x, 1)[:,0]

            for i in range(1, self.N):
                local_state = tf.one_hot(tf.cast(obs[:, i + self.stateNum * 2], tf.int32), self.stateNum)
                local_info = tf.concat([next_state_count, local_state, local_actions[:, i],
                                        tf.tile(tf.reshape(tf.one_hot(i, self.N), [1, self.N]), (tf.shape(obs)[0], 1))], axis=1)
                x = local_info  # tf.concat([next_state_count, next_coverages, remaining_demands], axis=1)#next_state_count#
                hidden = 0
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size, use_bias=False,
                                        name='hidden_' +str(hidden), reuse=True)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True, scope='ln_'+str(hidden), reuse=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    hidden += 1
                x = tf.layers.dense(x, 1, name='hidden_' +str(hidden), reuse=True)[:, 0]
                # x = tf.stack([remaining_demands,x], axis=1)
                next_return.append(x)  # tf.layers.dense(x, 1)[:,0]

            next_return = tf.stack(next_return, axis=1)

            output.append(next_return)
            output = tf.stack(output, axis=1)
            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'immediate_reward':immediate_reward, 'trip_weights':trip_weights, 'next_return':next_return, 'next_state_count':next_state_count}




class GridDecDeepVPN(Model):
    def __init__(self, nb_actions, inflows, adjacent_list, name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (32, 32), hidden_nonlinearity = tf.nn.relu, stateNum = 81, N = 20):
        super(GridDecDeepVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.stateNum = stateNum
        self.N = 1
        self.inflows = inflows
        self.adjacent_list = adjacent_list

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
           # obs = obs*self.N
            action_count = action_count#*self.N

            output = []


            normalizedDenseCounter = 0

            # state_count = obs[:, : self.stateNum ]*self.N
            demand_count = obs[:, self.stateNum*2:]
            normalizedDenseCounter +=1
            #payment

            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.ones_initializer(),
                                trainable=True)

            next_state_count = []

            for i in range(self.stateNum):
                temp = []
                for sa in self.inflows[i]:
                    temp.append(action_count[:,sa[0], sa[1]])
                temp = tf.stack(temp, axis=1)
                next_state_count.append(tf.reduce_sum(temp, axis=1))

            next_state_count = tf.stack(next_state_count, axis=1)

            next_coverages = []
            for i in range(self.stateNum):
                cover = tf.stack([next_state_count[:,j] for j in self.adjacent_list[i]], axis=1)
                cover = tf.minimum(tf.reduce_sum(cover, axis=1), 1.0)
                next_coverages.append(cover)

            next_coverages = tf.stack(next_coverages, axis=1)

            served_demands = []
            for i in range(len(self.nb_actions)):
                x = next_state_count[:,i]
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                served_demands.append(served_demand) #= tf.multiply(served_demand,trip_weights[i])

            served_demands = tf.stack(served_demands, axis=1)


            #remaining_demands = demand_count - served_demands
            remaining_demands = obs[:,self.stateNum*2:]*tf.minimum(1.0, next_state_count)
            #remaining_demands = tf.reduce_min(remaining_demands, axis=1)

            immediate_reward = tf.reduce_sum(served_demands, axis=1)#tf.layers.dense(served_demands, 1)[:,0]#
            output.append(immediate_reward)


            x = next_state_count#tf.concat([next_state_count, next_coverages, remaining_demands], axis=1)#next_state_count#
            for hidden_size in self.hidden_sizes:
                x = tf.layers.dense(x,
                                    hidden_size, use_bias = False)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                if self.batch_norm:
                    x = tc.layers.batch_norm(x)
                x = self.hidden_nonlinearity(x)
            x = tf.layers.dense(x, 1)[:,0]
            #x = tf.stack([remaining_demands,x], axis=1)
            next_return = x#tf.layers.dense(x, 1)[:,0]

            output.append(next_return)
            output = tf.stack(output, axis=1)
            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'immediate_reward':immediate_reward, 'trip_weights':trip_weights, 'next_return':next_return, 'next_state_count':next_state_count}


class StatelessCritic(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=False, batch_norm = False,  hidden_sizes = (32, 32), hidden_nonlinearity = tf.nn.relu, N = 20):
        super(StatelessCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
       # self.N = 1

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = action_count#tf.concat([next_state_count, next_coverages, remaining_demands], axis=1)#next_state_count#
            for hidden_size in self.hidden_sizes:
                x = tf.layers.dense(x,
                                    hidden_size, use_bias = False)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                if self.batch_norm:
                    x = tc.layers.batch_norm(x)
                x = self.hidden_nonlinearity(x)
            output = tf.layers.dense(x, 1)[:,0]
        return {'symbolic_val':output}


class GuassianSqueezeCritic(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=False, batch_norm = False,  hidden_sizes = (32, 32), hidden_nonlinearity = tf.nn.relu, N = 20, mu = 400.0, sigma = 200.0):
        super(GuassianSqueezeCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.N = N
        self.mu = mu
        self.sigma = sigma

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = tf.reduce_sum(action_count*tf.range(self.nb_actions, dtype=tf.float32), axis=1)
            output = x*tf.exp(-tf.square(x-self.mu)/self.sigma**2)
            sum_val = tf.square(tf.reduce_sum(action_count*tf.range(self.nb_actions, dtype=tf.float32), axis=1)-self.mu)
            #output = x
        return {'symbolic_val':output, 'test_val':sum_val}


def normalizedDense(x, num_units, nonlinearity=None, initializer = tf.random_normal_initializer(0, 0.05), counter = 0):
    ''' fully connected layer '''
    initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)
    V = tf.get_variable('V_' +str(counter), [int(x.get_shape()[1]),num_units], tf.float32, initializer, trainable=True)
    # with ops.name_scope(None, "softmax_normalize", [V]) as name:
    #     V = ops.convert_to_tensor(V, name="x")
    V_norm = tf.nn.softmax(V, dim=1)
    return tf.matmul(x, V_norm)