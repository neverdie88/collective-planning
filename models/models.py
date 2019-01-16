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

# class Actor(Model):
#     def __init__(self, nb_actions, name='actor', layer_norm=True):
#         super(Actor, self).__init__(name=name)
#         self.nb_actions = nb_actions
#         self.layer_norm = layer_norm
#
#     def __call__(self, obs, reuse=False):
#         with tf.variable_scope(self.name) as scope:
#             if reuse:
#                 scope.reuse_variables()
#
#             x = obs
#             # x = tf.layers.dense(x, 18)
#             # if self.layer_norm:
#             #     x = tc.layers.layer_norm(x, center=True, scale=True)
#             # x = tf.nn.relu(x)
#             #
#             # x = tf.layers.dense(x, 18)
#             # if self.layer_norm:
#             #     x = tc.layers.layer_norm(x, center=True, scale=True)
#             # inputLayer = tf.nn.relu(x)
#             inputLayer = x
#
#             # output = tf.nn.sigmoid(x)*(maxArray-minArray) + min
#             output = []
#             for i in range(len(self.nb_actions)):
#                 x = tf.layers.dense(inputLayer, self.nb_actions[i], kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
#                 x = tf.nn.softmax(x)
#                 output.append(x)
#             x = tf.concat(output, axis=1)
#             #x = tf.reshape(x, [-1])
#         return x


class Actor(Model):
    def __init__(self,  ob_space, ac_space, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.ac_space = ac_space
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()



            # output = tf.nn.sigmoid(x)*(maxArray-minArray) + min

            x = tf.layers.dense(obs, self.ac_space.n, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.softmax(x)

            #x = tf.reshape(x, [-1])
        return x

class Critic(Model):
    def __init__(self,  ob_space, ac_space, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.ac_space = ac_space
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            inputLayer = x

            x = tf.layers.dense(inputLayer, self.ac_space.n, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            # x = tf.reduce_sum(x*action, axis=1)
        return x

class CollectiveActor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(CollectiveActor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # x = obs
            # x = tf.layers.dense(x, 18)
            # if self.layer_norm:
            #     x = tc.layers.layer_norm(x, center=True, scale=True)
            # x = tf.nn.relu(x)
            #
            # x = tf.layers.dense(x, 18)
            # if self.layer_norm:
            #     x = tc.layers.layer_norm(x, center=True, scale=True)
            # inputLayer = tf.nn.relu(x)
            # inputLayer = x

            # output = tf.nn.sigmoid(x)*(maxArray-minArray) + min
            output = []
            for i in range(len(self.nb_actions)):
                inputLayer = obs[:,i]
                x = tf.layers.dense(inputLayer, self.nb_actions[i], kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.concat(output, axis=1)
            #x = tf.reshape(x, [-1])
        return x


class GridCollectiveCritic(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=True,  relu_output = True, num_channels = 10, num_maxout_units = 2):
        super(GridCollectiveCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.num_channels = num_channels
        self.num_maxout_units = num_maxout_units

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # if self.relu_output:
            output = []
            for i in range(len(self.nb_actions)):
                x = action_count[:,i]
                x = tf.layers.dense(x, self.num_channels*self.nb_actions[i])
                x = minout(x, self.num_maxout_units*self.nb_actions[i])
                output.append(x)
            output = tf.concat(output, axis=1)
            x = tf.layers.dense(output, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

class GridCollectiveActor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True, num_channels = 10, num_maxout_units = 2):
        super(GridCollectiveActor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.num_maxout_units = num_maxout_units
        self.num_channels = num_channels
        # self.hidden_nonlinearity = hidden_nonlinearity

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            output = []
            for i in range(len(self.nb_actions)):
                inputLayer = obs[:,i]
                x = tf.layers.dense(inputLayer, self.num_channels, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = minout(x, self.num_maxout_units)
                x = tf.layers.dense(x, self.nb_actions[i],
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
        return x

class TaxiCollectiveActorPseudoFlow(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=None):
        super(TaxiCollectiveActorPseudoFlow, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = None
        if layer_norm== 'layer_norm':
            self.layer_norm = tc.layers.layer_norm
        if layer_norm == 'batch_norm':
            self.layer_norm = tc.layers.batch_norm
        self.zoneNum = 81
        self.H = 48

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            normalizedDenseCounter = 0
            normalizedDenseCounter += 1

            state_count = obs[:,self.H:self.H + self.zoneNum]
            demand_count = obs[:,self.H + self.zoneNum:]
            initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)
            V = tf.get_variable('V_' + str(normalizedDenseCounter), [int(state_count.get_shape()[1]), self.zoneNum], tf.float32, initializer,
                                trainable=True)
            V_norm = tf.nn.softmax(V, dim=1)
            flows = tf.matmul(state_count, V_norm)

            # flows = normalizedDense(obs[:, self.H:self.H + self.zoneNum], self.zoneNum, counter=normalizedDenseCounter)

            for i in range(len(self.nb_actions)):
                #counterpart: the flow from other
                x = flows - tf.matmul(tf.expand_dims(state_count[:,i], axis=1),tf.expand_dims(V_norm[i, :],0))
                x = demand_count - x
                x = tf.layers.dense(x, 18, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm!= None:
                    x = self.layer_norm(x)
                x = tf.nn.relu(x)
                x = tf.layers.dense(x, self.nb_actions[i],
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
        return x

class TaxiCollectiveActorPseudoFlowMaxout(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=None, num_pieces = 3, num_maxout_units = 18):
        super(TaxiCollectiveActorPseudoFlowMaxout, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = None
        if layer_norm== 'layer_norm':
            self.layer_norm = tc.layers.layer_norm
        if layer_norm == 'batch_norm':
            self.layer_norm = tc.layers.batch_norm
        self.zoneNum = 81
        self.H = 48
        self.num_maxout_units = num_maxout_units
        self.num_pieces = num_pieces

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            normalizedDenseCounter = 0
            normalizedDenseCounter += 1

            state_count = obs[:,self.H:self.H + self.zoneNum]
            demand_count = obs[:,self.H + self.zoneNum:]
            initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)#tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)
            V = tf.get_variable('V_' + str(normalizedDenseCounter), [int(state_count.get_shape()[1]), self.zoneNum], tf.float32, initializer,
                                trainable=True)
            V_norm = tf.nn.softmax(V, dim=1)
            flows = tf.matmul(state_count, V_norm)

            # flows = normalizedDense(obs[:, self.H:self.H + self.zoneNum], self.zoneNum, counter=normalizedDenseCounter)

            for i in range(len(self.nb_actions)):
                #counterpart: the flow from other
                x = flows - tf.matmul(tf.expand_dims(state_count[:,i], axis=1),tf.expand_dims(V_norm[i, :],0))
                x = demand_count - x
                x = tf.layers.dense(x, self.num_pieces*self.num_maxout_units, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = minout(x, self.num_maxout_units)
                if self.layer_norm!= None:
                    x = self.layer_norm(x)
                x = tf.layers.dense(x, self.num_pieces * self.num_maxout_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = minout(x, self.num_maxout_units)
                if self.layer_norm != None:
                    x = self.layer_norm(x)
                x = tf.layers.dense(x, self.nb_actions[i],
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
            print('initialized actor network')
        return x


class TaxiCollectiveActorDense(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=None):
        super(TaxiCollectiveActorDense, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = None
        if layer_norm== 'layer_norm':
            self.layer_norm = tc.layers.layer_norm
        if layer_norm == 'batch_norm':
            self.layer_norm = tc.layers.batch_norm
        self.zoneNum = 81
        self.H = 48

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            for i in range(len(self.nb_actions)):
                inputLayer = obs#[:,:self.H]
                x = tf.layers.dense(inputLayer, 18, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm!= None:
                    x = self.layer_norm(x)
                x = tf.nn.relu(x)
                x = tf.layers.dense(x, 18,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm!= None:
                    x = self.layer_norm(x)
                x = tf.nn.relu(x)
                x = tf.layers.dense(x, self.nb_actions[i],
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
            #x = tf.reshape(x, [-1])
        return x


# class TaxiBasicCollectiveActor(Model):
#     def __init__(self, nb_actions, name='actor', layer_norm=True):
#         super(TaxiBasicCollectiveActor, self).__init__(name=name)
#         self.nb_actions = nb_actions
#         self.layer_norm = layer_norm
#         self.zoneNum = 81
#         self.H = 48
#
#     def __call__(self, obs, reuse=False):
#         with tf.variable_scope(self.name) as scope:
#             if reuse:
#                 scope.reuse_variables()
#
#             inputLayer = obs
#             output = []
#             for i in range(len(self.nb_actions)):
#                 inputLayer = obs#[:,:self.H]
#                 x = tf.layers.dense(inputLayer, self.nb_actions[i], kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
#                 x = tf.nn.softmax(x)
#                 output.append(x)
#             x = tf.stack(output, axis=1)
#
#
#             immediateRewards = []
#             for i in range(len(self.nb_actions)):
#                 x = tf.reduce_sum(action_count[:,:,i], axis=1)
#                 # next_state_count.append(x - obs[:, self.H + self.zoneNum +i] + customer_flow[:,i])
#                 immediateRewards.append(tf.reduce_min(tf.stack([x, obs[:, self.H + self.zoneNum +i]], axis=1), axis=1))
#             immediateRewards = tf.stack(immediateRewards, axis=1)
#             output.append(immediateRewards)
#             #cost
#             flatten_action_count = tf.reshape(action_count, shape=[-1, (action_count.shape[1]*action_count.shape[2]).value])
#             output.append(flatten_action_count)
#
#             #future reward
#             # next_state_count = tf.stack(next_state_count, axis=1)
#             # x = normalizedDense(next_state_count, self.zoneNum, counter = normalizedDenseCounter)
#             # normalizedDenseCounter += 1
#             # d = tf.get_variable("d", [self.zoneNum], trainable=True,
#             #                     initializer=tf.random_uniform_initializer(minval=0, maxval=300))
#             # x = tf.minimum(x, d)
#             # output.append(x)
#             output = tf.concat(output, axis=1)
#             x = tf.layers.dense(output, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
#         return x

class TaxiCollectiveActor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(TaxiCollectiveActor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.zoneNum = 81
        self.H = 48

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # x = obs
            # x = tf.layers.dense(x, 18)
            # if self.layer_norm:
            #     x = tc.layers.layer_norm(x, center=True, scale=True)
            # x = tf.nn.relu(x)
            #
            # x = tf.layers.dense(x, 18)
            # if self.layer_norm:
            #     x = tc.layers.layer_norm(x, center=True, scale=True)
            # inputLayer = tf.nn.relu(x)
            # inputLayer = x

            # output = tf.nn.sigmoid(x)*(maxArray-minArray) + min
            output = []
            for i in range(len(self.nb_actions)):
                inputLayer = obs#[:,:self.H]
                x = tf.layers.dense(inputLayer, self.nb_actions[i], kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
            #x = tf.reshape(x, [-1])
        return x

class TaxiCollectiveCritic(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=True,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(TaxiCollectiveCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputLayer = obs
            output = []
            next_state_count = []
            normalizedDenseCounter = 0
            customer_flow = normalizedDense(obs[:, self.H + self.zoneNum:], self.zoneNum,
                                            counter=normalizedDenseCounter)
            normalizedDenseCounter +=1
            #payment
            immediateRewards = []
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                next_state_count.append(x - obs[:, self.H + self.zoneNum +i] + customer_flow[:,i])
                immediateRewards.append(tf.reduce_min(tf.stack([x, obs[:, self.H + self.zoneNum +i]], axis=1), axis=1))
            immediateRewards = tf.stack(immediateRewards, axis=1)
            output.append(immediateRewards)
            #cost
            flatten_action_count = tf.reshape(action_count, shape=[-1, (action_count.shape[1]*action_count.shape[2]).value])
            output.append(flatten_action_count)

            #future reward
            next_state_count = tf.stack(next_state_count, axis=1)
            x = normalizedDense(next_state_count, self.zoneNum, counter = normalizedDenseCounter)
            normalizedDenseCounter += 1
            d = tf.get_variable("d", [self.zoneNum], trainable=True,
                                initializer=tf.random_uniform_initializer(minval=0, maxval=300))
            x = tf.minimum(x, d)
            output.append(x)
            output = tf.concat(output, axis=1)
            x = tf.layers.dense(output, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

class TaxiCollectiveTestCritic(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=True,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(TaxiCollectiveTestCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputLayer = obs
            output = []
            next_state_count = []
            normalizedDenseCounter = 0
            customer_flow = normalizedDense(obs[:, self.H+self.zoneNum:], self.zoneNum, counter = normalizedDenseCounter)
            normalizedDenseCounter +=1
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                next_state_count.append(x - obs[:, self.H + self.zoneNum +i] + customer_flow[:,i])


            #future reward
            next_state_count = tf.stack(next_state_count, axis=1)
            x = normalizedDense(next_state_count, self.zoneNum, counter = normalizedDenseCounter)
            normalizedDenseCounter += 1
            #
            # x = tf.minimum(x, d)
            # output.append(x)
            # output = tf.concat(output, axis=1)
            # x = tf.layers.dense(output, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

class TaxiBasicCollectiveCritic(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=True,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(TaxiBasicCollectiveCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputLayer = obs
            output = []
            # next_state_count = []
            # normalizedDenseCounter = 0
            # customer_flow = normalizedDense(obs[:, self.H:], self.zoneNum, counter = normalizedDenseCounter)
            # normalizedDenseCounter +=1
            #payment
            immediateRewards = []
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                # next_state_count.append(x - obs[:, self.H + self.zoneNum +i] + customer_flow[:,i])
                immediateRewards.append(tf.reduce_min(tf.stack([x, obs[:, self.H + self.zoneNum +i]], axis=1), axis=1))
            immediateRewards = tf.stack(immediateRewards, axis=1)
            output.append(immediateRewards)
            #cost
            flatten_action_count = tf.reshape(action_count, shape=[-1, (action_count.shape[1]*action_count.shape[2]).value])
            output.append(flatten_action_count)

            #future reward
            # next_state_count = tf.stack(next_state_count, axis=1)
            # x = normalizedDense(next_state_count, self.zoneNum, counter = normalizedDenseCounter)
            # normalizedDenseCounter += 1
            # d = tf.get_variable("d", [self.zoneNum], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=0, maxval=300))
            # x = tf.minimum(x, d)
            # output.append(x)
            output = tf.concat(output, axis=1)
            x = tf.layers.dense(output, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
        return x

class TaxiCollectiveCritic(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=True,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(TaxiCollectiveCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputLayer = obs
            output = []
            next_state_count = []
            normalizedDenseCounter = 0
            output.append(obs[:, :self.H])
            customer_flow = normalizedDense(obs[:, self.H+ self.zoneNum:], self.zoneNum, counter = normalizedDenseCounter)
            normalizedDenseCounter +=1
            #payment
            immediateRewards = []
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                next_state_count.append(x - obs[:, self.H + self.zoneNum +i] + customer_flow[:,i])
                immediateRewards.append(tf.reduce_min(tf.stack([x, obs[:, self.H + self.zoneNum +i]], axis=1), axis=1))
            immediateRewards = tf.stack(immediateRewards, axis=1)
            output.append(immediateRewards)
            #cost
            flatten_action_count = tf.reshape(action_count, shape=[-1, (action_count.shape[1]*action_count.shape[2]).value])
            output.append(flatten_action_count)

            #future reward
            next_state_count = tf.stack(next_state_count, axis=1)
            x = normalizedDense(next_state_count, self.zoneNum, counter = normalizedDenseCounter)
            normalizedDenseCounter += 1
            d = tf.get_variable("d", [self.zoneNum], trainable=True,
                                initializer=tf.random_uniform_initializer(minval=0, maxval=300))
            x = tf.minimum(x, d)
            output.append(x)
            output = tf.concat(output, axis=1)
            x = tf.layers.dense(output, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

class CollectiveCritic(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=True,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(CollectiveCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputLayer = obs


            if self.relu_output:
                w = tf.get_variable("w", [np.sum(self.nb_actions)], trainable=True, initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                b = tf.get_variable("b", [np.sum(self.nb_actions)], trainable=True,
                                    initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.multiply(action_count, w) + b
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, np.sum(self.nb_actions))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                for idx, hidden_size in enumerate(self.hidden_sizes):#self.nb_actions[i]
                    x = tf.layers.dense(x, hidden_size,
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                    x = self.hidden_nonlinearity(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x


class CollectiveStateCritic(Model):
    def __init__(self, nb_actions, name='collective_state_critic', layer_norm=True, relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(CollectiveStateCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        self.actionNum = np.sum(nb_actions)
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity

    def __call__(self, obs, reuse=False):#
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # x = obs
            # x = tf.layers.dense(x, 18)
            # if self.layer_norm:
            #     x = tc.layers.layer_norm(x, center=True, scale=True)
            # x = tf.nn.relu(x)
            #
            # x = tf.layers.dense(x, 18)
            # if self.layer_norm:
            #     x = tc.layers.layer_norm(x, center=True, scale=True)
            # inputLayer = tf.nn.relu(x)
            #
            # output = []
            # for i in range(len(self.nb_actions)):
            #     x = tf.layers.dense(inputLayer, self.nb_actions[i],
            #                         kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            #     x = tf.nn.relu(x)
            #     output.append(x)
            # x = tf.concat(output, axis=1)
            # x = tf.multiply(x, tf.multiply(state_count, action))

            inputLayer = obs#tf.multiply(state_count, action)

            output = []
            # if len(self.hidden_sizes)>0:
            for i in range(len(self.nb_actions)):
                x = inputLayer
                for idx, hidden_size in enumerate(self.hidden_sizes):#self.nb_actions[i]
                    x = tf.layers.dense(x, hidden_size,
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                    x = self.hidden_nonlinearity(x)
                x = tf.layers.dense(x, self.nb_actions[i],
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = self.hidden_nonlinearity(x)
                output.append(x)
            x = tf.concat(output, axis=1)
        return x

class CollectiveStateActionCritic(Model):
    def __init__(self, nb_actions, name='collective_state_action_critic', layer_norm=True, relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(CollectiveStateActionCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        self.actionNum = np.sum(nb_actions)
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity

    def __call__(self, obs, action_count, reuse=False):#
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # x = obs
            # x = tf.layers.dense(x, 18)
            # if self.layer_norm:
            #     x = tc.layers.layer_norm(x, center=True, scale=True)
            # x = tf.nn.relu(x)
            #
            # x = tf.layers.dense(x, 18)
            # if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            # inputLayer = tf.nn.relu(x)
            #
            # output = []
            # for i in range(len(self.nb_actions)):
            #     x = tf.layers.dense(inputLayer, self.nb_actions[i],
            #                         kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            #     x = tf.nn.relu(x)
            #     output.append(x)
            # x = tf.concat(output, axis=1)
            # x = tf.multiply(x, tf.multiply(state_count, action))

            inputLayer = obs#tf.multiply(state_count, action)

            output = []
            # if len(self.hidden_sizes)>0:
            for i in range(len(self.nb_actions)):
                x = inputLayer
                for idx, hidden_size in enumerate(self.hidden_sizes):#self.nb_actions[i]
                    x = tf.layers.dense(x, hidden_size,
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                    x = self.hidden_nonlinearity(x)
                x = tf.layers.dense(x, self.actionNum,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = x * action_count
                x = self.hidden_nonlinearity(x)
                x = tf.layers.dense(x, self.nb_actions[i],
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = self.hidden_nonlinearity(x)
                output.append(x)
            x = tf.concat(output, axis=1)
        return x

class Critic1(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic1, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars






def normalizedDense(x, num_units, nonlinearity=None, initializer = tf.random_normal_initializer(0, 0.05), counter = 0):
    ''' fully connected layer '''
    initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)
    V = tf.get_variable('V_' +str(counter), [int(x.get_shape()[1]),num_units], tf.float32, initializer, trainable=True)
    # with ops.name_scope(None, "softmax_normalize", [V]) as name:
    #     V = ops.convert_to_tensor(V, name="x")
    V_norm = tf.nn.softmax(V, dim=1)
    return tf.matmul(x, V_norm)



#######################################################################
#police patrol neural network


class CollectiveDecActorPoliceAttention(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=False, batch_norm=False, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, C = 0, adjacent_list=[]):
        super(CollectiveDecActorPoliceAttention, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        self.stateNum = len(adjacent_list)
        self.C = C

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            # local_obs_joint = []
            # w = tf.get_variable("temporal_weight", [self.C], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            # w = tf.nn.sigmoid(w)
            # w = tf.get_variable("temporal_weight", [1], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=0, maxval=1))
            # w = tf.nn.sigmoid(-w * tf.range(self.C, dtype=tf.float32))
            w = tf.get_variable("temporal_weight", [1], trainable=True,
                                initializer=tf.random_uniform_initializer(minval=0, maxval=1))
            w = tf.exp(-w * tf.range(self.C, dtype=tf.float32))
            state_obs = tf.multiply(obs[:,:,:self.C], w)
            state_obs = tf.reduce_sum(state_obs, axis=2)
            for i in range(len(self.nb_actions)):
                local_obs1 = [state_obs[:,j] for j in self.adjacent_list[i]]
                local_obs1 = tf.stack(local_obs1, axis=1)                # local_obs_joint.append(local_obs)
                local_obs2 = [obs[:, j, self.C] for j in self.adjacent_list[i]]
                local_obs2 = tf.stack(local_obs2, axis=1)
                x = tf.concat([local_obs1, local_obs2], axis=1)
                adjacent_num = len(self.adjacent_list[i])
                # local_output = tf.zeros([None, adjacent_num], tf.float32)
                local_output = [obs[:,0,0]*0.0]*self.stateNum
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    # y.append(x)
                x = tf.layers.dense(x, adjacent_num)
                x = tf.nn.softmax(x)
                for j in range(len(self.adjacent_list[i])):
                    local_output[self.adjacent_list[i][j]] = x[:,j]
                local_output = tf.stack(local_output, axis=1)
                output.append(local_output)
            x = tf.stack(output, axis=1)
            # x = tf.reshape(x, [-1])
        return x#{'action':x, 'y':y, 'local_obs_joint':local_obs_joint}


class DecActorPoliceAttention(Model):
    def __init__(self, nb_actions, name='dec_actor', layer_norm=False, batch_norm=False, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, C = 0, adjacent_list=[], N = 1):
        super(DecActorPoliceAttention, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = np.zeros((len(adjacent_list), len(adjacent_list)), dtype=np.float32)
        for i in range(len(adjacent_list)):
            for j in adjacent_list[i]:
                self.adjacent_list[i, j] = 1
        self.stateNum = len(adjacent_list)
        self.C = C
        self.N = N

    def __call__(self, obs, local_states, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            # local_obs_joint = []
            # w = tf.get_variable("temporal_weight", [self.C], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            # w = tf.nn.sigmoid(w)
            # w = tf.get_variable("temporal_weight", [1], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=0, maxval=1))
            # w = tf.nn.sigmoid(-w * tf.range(self.C, dtype=tf.float32))
            w = tf.get_variable("temporal_weight", [1], trainable=True,
                                initializer=tf.random_uniform_initializer(minval=0, maxval=1))
            w = tf.exp(-w * tf.range(self.C, dtype=tf.float32))
            state_obs = tf.multiply(obs[:,:,:self.C], w)
            state_obs = tf.reduce_sum(state_obs, axis=2)
            joint_obs_masks = []
            for i in range(self.N):
                local_state = local_states[:, i]#tf.one_hot(tf.cast(local_states[:, i], tf.int32), self.stateNum)
                obs_masks = tf.matmul(local_state, self.adjacent_list)
                obs_masks = tf.where(tf.reduce_max(obs_masks, axis=1)>0, obs_masks, obs_masks + 1.0)
                joint_obs_masks.append(obs_masks)

                local_obs1 = state_obs*obs_masks
                local_obs2 = obs[:, :, self.C]*obs_masks
                x = tf.concat([local_obs1, local_obs2, local_state], axis=1)
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    # y.append(x)
                x = tf.layers.dense(x, self.stateNum)
                local_output = masked_softmax(x, obs_masks)
                output.append(local_output)
            x = tf.stack(output, axis=1)
            # x = tf.reshape(x, [-1])
        return x, joint_obs_masks

class CollectiveDecCriticPoliceAttention(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=False, batch_norm=False, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, C = 0, adjacent_list=[]):
        super(CollectiveDecActorPoliceAttention, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        self.stateNum = len(adjacent_list)
        self.C = C

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            # local_obs_joint = []
            # w = tf.get_variable("temporal_weight", [self.C], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            # w = tf.nn.sigmoid(w)
            # w = tf.get_variable("temporal_weight", [1], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=0, maxval=1))
            # w = tf.nn.sigmoid(-w * tf.range(self.C, dtype=tf.float32))
            w = tf.get_variable("temporal_weight", [1], trainable=True,
                                initializer=tf.random_uniform_initializer(minval=0, maxval=1))
            w = tf.exp(-w * tf.range(self.C, dtype=tf.float32))
            state_obs = tf.multiply(obs[:,:,:self.C], w)
            state_obs = tf.reduce_sum(state_obs, axis=2)
            for i in range(len(self.nb_actions)):
                local_obs1 = [state_obs[:,j] for j in self.adjacent_list[i]]
                local_obs1 = tf.stack(local_obs1, axis=1)                # local_obs_joint.append(local_obs)
                local_obs2 = [obs[:, j, self.C] for j in self.adjacent_list[i]]
                local_obs2 = tf.stack(local_obs2, axis=1)
                x = tf.concat([local_obs1, local_obs2], axis=1)
                adjacent_num = len(self.adjacent_list[i])
                # local_output = tf.zeros([None, adjacent_num], tf.float32)
                local_output = [obs[:,0,0]*0.0]*self.stateNum
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    # y.append(x)
                x = tf.layers.dense(x, adjacent_num)
                for j in range(len(self.adjacent_list[i])):
                    local_output[self.adjacent_list[i][j]] = x[:,j]
                local_output = tf.stack(local_output, axis=1)
                output.append(local_output)
            x = tf.stack(output, axis=1)
            # x = tf.reshape(x, [-1])
        return x#{'action':x, 'y':y, 'local_obs_joint':local_obs_joint}


class CollectiveCenCriticPolice(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, adjacent_list = [], N = 10):
        super(CollectiveCenCriticPolice, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = len(nb_actions)
        self.N = N
        self.adjacent_list = adjacent_list

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            obs = obs#*self.N
            action_count = action_count#*self.N

            next_state_count = []
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                # future reward
                next_x = x
                next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)

            x = tf.concat([next_state_count, obs[:,self.zoneNum:]], axis=1)
            for hidden_size in self.hidden_sizes:
                x = tf.layers.dense(x,
                                    hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                if self.batch_norm:
                    x = tc.layers.batch_norm(x)
                x = self.hidden_nonlinearity(x)
            output = tf.layers.dense(x, 1)[:,0]
        return {'symbolic_val':output, 'next_state_count':next_state_count}

class CollectiveCenCriticPoliceFull(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, C = 0, adjacent_list = [],discretized_travel_times = [], N = 10):
        super(CollectiveCenCriticPoliceFull, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = len(nb_actions)
        self.N = N
        self.adjacent_list = adjacent_list
        self.C = C
        self.discretized_travel_times = discretized_travel_times

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            obs = obs#*self.N
            action_count = action_count#*self.N

            next_state_count = []
            # w = tf.get_variable("temporal_weight", [self.C], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            # w = tf.nn.sigmoid(w)
            # w = tf.get_variable("temporal_weight", [1], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=0, maxval=1))
            # w = tf.nn.sigmoid(-w * tf.range(self.C, dtype=tf.float32))
            w = tf.get_variable("temporal_weight", [1], trainable=True,
                                initializer=tf.random_uniform_initializer(minval=0, maxval=1))
            w = tf.exp(-w * tf.range(self.C, dtype=tf.float32))
            for i in range(len(self.nb_actions)):
                temp = []
                for c in range(self.C):
                    temp.append([obs[:, i, c]])
                for j in self.adjacent_list[i]:
                    temp[self.discretized_travel_times[j, i]].append(action_count[:, j, i])
                for c in range(self.C):
                    temp[c] = tf.reduce_sum(tf.stack(temp[c], axis=1), axis=1)
                temp = tf.stack(temp, axis=1)
                next_count = tf.reduce_sum(tf.multiply(temp, w), axis=1)
                next_state_count.append(next_count)
                # x = [action_count[:, :, i], obs[:,self.zoneNum+i]]#tf.reduce_sum(action_count[:, :, i], axis=1)
                # # future reward
                # x = tf.concat([action_count[:, :, i], tf.reshape(obs[:,self.zoneNum+i], [-1,1])], axis=1)
                # next_x = tf.layers.dense(x, 1)[:,0]
                # next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)

            x = next_state_count
            for hidden_size in self.hidden_sizes:
                x = tf.layers.dense(x,
                                    hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                if self.batch_norm:
                    x = tc.layers.batch_norm(x)
                x = self.hidden_nonlinearity(x)
            output = tf.layers.dense(x, 1)[:,0]
        return {'symbolic_val':output, 'next_state_count':next_state_count}


# tf.reshape(action_count, [-1, 24 * 24])
#
# x = tf.concat([next_state_count, obs[:, self.zoneNum:]], axis=1)
# for hidden_size in self.hidden_sizes:
#     x = tf.layers.dense(x,
#                         hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
#     if self.layer_norm:
#         x = tc.layers.layer_norm(x, center=True, scale=True)
#     if self.batch_norm:
#         x = tc.layers.batch_norm(x)
#     x = self.hidden_nonlinearity(x)

class CollectiveCenCriticPoliceFullVPN(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, adjacent_list = [], C = 0, discretized_travel_times = [], N = 10):
        super(CollectiveCenCriticPoliceFullVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = len(nb_actions)
        self.C = C
        self.N = N
        self.adjacent_list = adjacent_list
        self.discretized_travel_times = discretized_travel_times

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            state_count = []#action_count*0.0#*self.N



            next_state_count = []
            #temporal attention mechanism
            # w = tf.get_variable("temporal_weight", [self.C], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            # w = tf.nn.sigmoid(w)
            # w = tf.get_variable("temporal_weight", [1], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=0, maxval=1))
            # w = tf.nn.sigmoid(-w * tf.range(self.C, dtype=tf.float32))
            w = tf.get_variable("temporal_weight", [1], trainable=True,
                                initializer=tf.random_uniform_initializer(minval=0, maxval=1))
            w = tf.exp(-w * tf.range(self.C, dtype=tf.float32))


            for i in range(len(self.nb_actions)):
                temp = []
                for c in range(self.C):
                    temp.append([obs[:,i, c]])
                for j in self.adjacent_list[i]:
                    temp[self.discretized_travel_times[j, i]].append(action_count[:,j, i])
                for c in range(self.C):
                    temp[c] = tf.reduce_sum(tf.stack(temp[c], axis=1), axis=1)
                temp = tf.stack(temp, axis=1)
                next_count = tf.reduce_sum(tf.multiply(temp, w), axis=1)
                next_state_count.append(next_count)
                # x = [action_count[:, :, i], obs[:,self.zoneNum+i]]#tf.reduce_sum(action_count[:, :, i], axis=1)
                # # future reward
                # x = tf.concat([action_count[:, :, i], tf.reshape(obs[:,self.zoneNum+i], [-1,1])], axis=1)
                # next_x = tf.layers.dense(x, 1)[:,0]
                # next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)

            x = next_state_count
            for hidden_size in self.hidden_sizes:
                x = tf.layers.dense(x,
                                    hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                if self.batch_norm:
                    x = tc.layers.batch_norm(x)
                x = self.hidden_nonlinearity(x)
            immediate_reward = tf.layers.dense(x, 1)[:,0]

            x = next_state_count
            for hidden_size in self.hidden_sizes:
                x = tf.layers.dense(x,
                                    hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                if self.batch_norm:
                    x = tc.layers.batch_norm(x)
                x = self.hidden_nonlinearity(x)
            next_return = tf.layers.dense(x, 1)[:, 0]

            output = [immediate_reward, next_return]
            output = tf.stack(output, axis=1)
            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'immediate_reward':immediate_reward, 'next_return':next_return,  'next_state_count':next_state_count}



class DecCenCriticPoliceFullVPN(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, adjacent_list = [], C = 0, discretized_travel_times = [], N = 10):
        super(DecCenCriticPoliceFullVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = len(nb_actions)
        self.C = C
        self.N = N
        self.adjacent_list = adjacent_list
        self.discretized_travel_times = discretized_travel_times

    def __call__(self, obs, action_count, local_states, local_actions, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            state_count = []#action_count*0.0#*self.N



            next_state_count = []
            #temporal attention mechanism
            # w = tf.get_variable("temporal_weight", [self.C], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            # w = tf.nn.sigmoid(w)
            # w = tf.get_variable("temporal_weight", [1], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=0, maxval=1))
            # w = tf.nn.sigmoid(-w * tf.range(self.C, dtype=tf.float32))
            w = tf.get_variable("temporal_weight", [1], trainable=True,
                                initializer=tf.random_uniform_initializer(minval=0, maxval=1))
            w = tf.exp(-w * tf.range(self.C, dtype=tf.float32))


            for i in range(len(self.nb_actions)):
                temp = []
                for c in range(self.C):
                    temp.append([obs[:,i, c]])
                for j in self.adjacent_list[i]:
                    temp[self.discretized_travel_times[j, i]].append(action_count[:,j, i])
                for c in range(self.C):
                    temp[c] = tf.reduce_sum(tf.stack(temp[c], axis=1), axis=1)
                temp = tf.stack(temp, axis=1)
                next_count = tf.reduce_sum(tf.multiply(temp, w), axis=1)
                next_state_count.append(next_count)
                # x = [action_count[:, :, i], obs[:,self.zoneNum+i]]#tf.reduce_sum(action_count[:, :, i], axis=1)
                # # future reward
                # x = tf.concat([action_count[:, :, i], tf.reshape(obs[:,self.zoneNum+i], [-1,1])], axis=1)
                # next_x = tf.layers.dense(x, 1)[:,0]
                # next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)
            immediate_rewards = []
            next_returns = []
            i = 0
            local_info = tf.concat([next_state_count, local_states[:, i], local_actions[:,i], tf.tile(tf.reshape(tf.one_hot(i, self.N), [1, self.N]), (tf.shape(obs)[0],1))], axis=1)
            x = local_info#tf.concat([next_state_count, next_coverages, remaining_demands], axis=1)#next_state_count#
            hidden = 0
            for hidden_size in self.hidden_sizes:
                x = tf.layers.dense(x,
                                    hidden_size, name='hidden_'+str(hidden))  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True, scope='ln_'+str(hidden))
                if self.batch_norm:
                    x = tc.layers.batch_norm(x)
                x = self.hidden_nonlinearity(x)
                hidden += 1
            immediate_reward = tf.layers.dense(x, 1, name='hidden_'+str(hidden))[:,0]
            immediate_rewards.append(immediate_reward)
            hidden += 1

            x = local_info
            for hidden_size in self.hidden_sizes:
                x = tf.layers.dense(x,
                                    hidden_size, name='hidden_'+str(hidden))   # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True, scope='ln_'+str(hidden))
                if self.batch_norm:
                    x = tc.layers.batch_norm(x)
                x = self.hidden_nonlinearity(x)
                hidden += 1
            next_return = tf.layers.dense(x, 1, name='hidden_'+str(hidden))[:, 0]
            next_returns.append(next_return)

            for i in range(1, self.N):
                local_info = tf.concat([next_state_count, local_states[:, i], local_actions[:, i],
                                        tf.tile(tf.reshape(tf.one_hot(i, self.N), [1, self.N]), (tf.shape(obs)[0], 1))],
                                       axis=1)
                x = local_info  # tf.concat([next_state_count, next_coverages, remaining_demands], axis=1)#next_state_count#
                hidden = 0
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size, name='hidden_' + str(
                            hidden), reuse=True)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True, scope='ln_' + str(hidden), reuse=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    hidden += 1
                immediate_reward = tf.layers.dense(x, 1, name='hidden_' + str(hidden), reuse=True)[:, 0]
                immediate_rewards.append(immediate_reward)
                hidden += 1

                x = local_info
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size, name='hidden_' + str(
                            hidden), reuse=True)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True, scope='ln_' + str(hidden), reuse=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    hidden += 1
                next_return = tf.layers.dense(x, 1, name='hidden_' + str(hidden), reuse=True)[:, 0]
                next_returns.append(next_return)

            next_returns = tf.stack(next_returns, axis=1)
            immediate_rewards = tf.stack(immediate_rewards, axis=1)
            output = [immediate_rewards, next_returns]
            output = tf.stack(output, axis=1)
            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'immediate_reward':immediate_rewards, 'next_return':next_returns,  'next_state_count':next_state_count}


class CollectiveCenCriticPoliceNeighboring(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=False, batch_norm=False, relu_output=True,
                 hidden_sizes=(), hidden_nonlinearity=tf.nn.relu, adjacent_list=[], N=10):
        super(CollectiveCenCriticPoliceNeighboring, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.nb_actions = nb_actions
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = len(nb_actions)
        self.N = N
        self.adjacent_list = adjacent_list

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            obs = obs  # *self.N
            action_count = action_count  # *self.N

            next_state_count = []
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                # future reward
                next_x = x
                next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)
            future_count = obs[:, self.zoneNum:]
            x = tf.concat([next_state_count, obs[:, self.zoneNum:]], axis=1)
            output = []
            for i in range(len(self.nb_actions)):
                local_obs = [next_state_count[:, j] for j in self.adjacent_list[i]]
                local_obs.extend([future_count[:,j] for j in self.adjacent_list[i]])
                local_obs = tf.stack(local_obs, axis=1)  # local_obs_joint.append(local_obs)
                x = tf.layers.dense(local_obs, 1)[:, 0]
                x = self.hidden_nonlinearity(x)
                output.append(x)

            output = tf.stack(output, axis=1)
            x = output
            for hidden_size in self.hidden_sizes:
                x = tf.layers.dense(x,
                                    hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                if self.batch_norm:
                    x = tc.layers.batch_norm(x)
                x = self.hidden_nonlinearity(x)
            output = tf.layers.dense(x, 1)[:, 0]
        return {'symbolic_val': output, 'next_state_count': next_state_count}
