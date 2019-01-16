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


class CollectiveDecActorGridPatrolling(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[] , stateNum = 0, N = 1):
        super(CollectiveDecActorGridPatrolling, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        self.stateNum = stateNum
        self.N = float(N)
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            local_obs_joint = []
            #visiable_obs = tf.concat([obs[:, :self.stateNum*2], obs[:, self.stateNum:self.stateNum*2]*self.N*obs[:, self.stateNum*2:]], axis=1)
            for i in range(len(self.nb_actions)):
                local_obs = [obs[:,j] for j in self.adjacent_list[i]]
                local_obs.extend([obs[:, self.stateNum + j]*obs[:, 2*self.stateNum + j] for j in self.adjacent_list[i]])
                local_obs = tf.stack(local_obs, axis=1)
                local_obs_joint.append(local_obs)
                #local_obs = visiable_obs#obs[:, :self.stateNum*2]
                x = local_obs
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                x = tf.layers.dense(x, self.nb_actions[i])
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
            # local_obs_joint = tf.stack(local_obs_joint, axis=1)
        return x#{'output':x, 'local_obs':local_obs_joint}



class CollectiveDecSharedActorGrid(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[] , stateNum = 0, N = 1):
        super(CollectiveDecSharedActorGrid, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        self.stateNum = stateNum
        self.N = float(N)
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            i = 0
            Mask = np.zeros(len(self.nb_actions) * 2)
            for j in self.adjacent_list[i]:
                Mask[j] = 1
                Mask[j + len(self.nb_actions)] = 1
            Mask = tf.Variable(Mask, dtype=tf.float32)
            local_obs = tf.multiply(obs, Mask)
            # local_obs_joint.append(local_obs)
            x = local_obs
            hidden = 0
            loc_weights = []
            # output_biases = []
            for hidden_size in self.hidden_sizes:
                loc_weight = tf.get_variable('loc_weight_' + str(hidden), [hidden_size], tf.float32, trainable=True)
                loc_weights.append(loc_weight)
                x = tf.layers.dense(x,
                                    hidden_size, name='hidden_layer_' + str(
                        hidden))  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                x = tf.add(x, loc_weight)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                if self.batch_norm:
                    x = tc.layers.batch_norm(x)
                x = self.hidden_nonlinearity(x)
                hidden += 1
            output_bias = tf.get_variable('output_bias_' + str(i), [self.nb_actions[i]], tf.float32, trainable=True)
            # output_biases.append(output_bias)
            x = tf.layers.dense(x, self.nb_actions[i])
            x = tf.add(x, output_bias)
            x = tf.nn.softmax(x)
            output.append(x)

            for i in range(1, len(self.nb_actions)):
                Mask = np.zeros(len(self.nb_actions) * 2)
                for j in self.adjacent_list[i]:
                    Mask[j] = 1
                    Mask[j + len(self.nb_actions)] = 1
                Mask = tf.Variable(Mask, dtype=tf.float32)
                local_obs = tf.multiply(obs, Mask)
                # local_obs_joint.append(local_obs)
                x = local_obs
                hidden = 0
                for hidden_size in self.hidden_sizes:

                    x = tf.layers.dense(x,
                                        hidden_size, name='hidden_layer_' + str(
                            hidden),
                                        reuse=True)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    x = tf.add(x, loc_weights[hidden])
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    hidden += 1
                output_bias = tf.get_variable('output_bias_' + str(i), [self.nb_actions[i]], tf.float32, trainable=True)
                x = tf.layers.dense(x, self.nb_actions[i])
                x = tf.add(x, output_bias)
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
            # local_obs_joint = tf.stack(local_obs_joint, axis=1)
        return x

class CollectiveDecActorGrid(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[] , stateNum = 0, N = 1):
        super(CollectiveDecActorGrid, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        self.stateNum = stateNum
        self.N = float(N)
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            local_obs_joint = []
            #visiable_obs = tf.concat([obs[:, :self.stateNum*2], obs[:, self.stateNum:self.stateNum*2]*self.N*obs[:, self.stateNum*2:]], axis=1)
            for i in range(len(self.nb_actions)):
                local_obs = [obs[:,j] for j in self.adjacent_list[i]]
                local_obs.extend([obs[:, self.stateNum + j]*obs[:, self.stateNum + j] for j in self.adjacent_list[i]])
                local_obs = tf.stack(local_obs, axis=1)
                local_obs_joint.append(local_obs)
                #local_obs = visiable_obs#obs[:, :self.stateNum*2]
                x = local_obs
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                x = tf.layers.dense(x, self.nb_actions[i])
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
            # local_obs_joint = tf.stack(local_obs_joint, axis=1)
        return x#{'output':x, 'local_obs':local_obs_joint}

class CollectiveDecActorGridNObs(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[] , stateNum = 0, N = 1):
        super(CollectiveDecActorGridNObs, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        self.stateNum = stateNum
        self.N = float(N)
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            local_obs_joint = []
            #visiable_obs = tf.concat([obs[:, :self.stateNum*2], obs[:, self.stateNum:self.stateNum*2]*self.N*obs[:, self.stateNum*2:]], axis=1)
            for i in range(len(self.nb_actions)):
                local_obs = [obs[:,j] for j in self.adjacent_list[i]]
                local_obs = tf.stack(local_obs, axis=1)
                local_obs_joint.append(local_obs)
                #local_obs = visiable_obs#obs[:, :self.stateNum*2]
                x = local_obs
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                x = tf.layers.dense(x, self.nb_actions[i])
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
            # local_obs_joint = tf.stack(local_obs_joint, axis=1)
        return x#{'output':x, 'local_obs':local_obs_joint}



class CollectiveDecActorGrid0Obs(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[] , stateNum = 0, N = 1):
        super(CollectiveDecActorGrid0Obs, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        self.stateNum = stateNum
        self.N = float(N)
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            local_obs_joint = []
            #visiable_obs = tf.concat([obs[:, :self.stateNum*2], obs[:, self.stateNum:self.stateNum*2]*self.N*obs[:, self.stateNum*2:]], axis=1)
            for i in range(len(self.nb_actions)):
                x = tf.get_variable('a_'+str(i),[self.nb_actions[i]], tf.float32,
                                tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                    trainable=True)
                x = tf.nn.softmax(x)
                x = tf.reshape(x, shape=[1, self.nb_actions[i]])
                output.append(x)
            x = tf.stack(output, axis=1)
        return x#{'output':x, 'local_obs':local_obs_joint}


class CollectiveDecCriticGrid0Obs(Model):
    def __init__(self, nb_actions, name='dec_collective_critic', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[] , stateNum = 0, N = 1):
        super(CollectiveDecCriticGrid0Obs, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        self.stateNum = stateNum
        self.N = float(N)
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            local_obs_joint = []
            #visiable_obs = tf.concat([obs[:, :self.stateNum*2], obs[:, self.stateNum:self.stateNum*2]*self.N*obs[:, self.stateNum*2:]], axis=1)
            for i in range(len(self.nb_actions)):
                x = tf.get_variable('a_'+str(i),[self.nb_actions[i]], tf.float32,
                                tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                    trainable=True)
                x = tf.reshape(x, shape=[1, self.nb_actions[i]])
                output.append(x)
            x = tf.stack(output, axis=1)
        return x#{'output':x, 'local_obs':local_obs_joint}


class StatelessActor(Model):
    def __init__(self, nb_actions, name='dec_collective_actor'):
        super(StatelessActor, self).__init__(name=name)
        self.nb_actions = nb_actions


    def __call__(self, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = tf.get_variable('x', [self.nb_actions], tf.float64,
                                tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                trainable=True)
            x = tf.nn.softmax(x)
            x = tf.reshape(x, shape=[1, self.nb_actions])
        return x




class CollectiveDecCriticGrid(Model):
    def __init__(self, nb_actions, name='dec_collective_critic', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[] , stateNum = 0):
        super(CollectiveDecCriticGrid, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        self.stateNum = stateNum
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            # local_obs_joint = []
            for i in range(len(self.nb_actions)):
                local_obs = [obs[:,j] for j in self.adjacent_list[i]]
                local_obs.extend([obs[:, self.stateNum + j] for j in self.adjacent_list[i]])
                local_obs = tf.stack(local_obs, axis=1)
                # local_obs_joint.append(local_obs)
                x = local_obs
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                x = tf.layers.dense(x, self.nb_actions[i])
                output.append(x)
            x = tf.stack(output, axis=1)
        return x



class CollectiveDecSharedCriticGrid(Model):
    def __init__(self, nb_actions, name='dec_collective_critic', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[] , stateNum = 0):
        super(CollectiveDecSharedCriticGrid, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        self.stateNum = stateNum
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # construct shared dense layers
            # indices = np.arange(len(self.nb_actions))
            # loc_indices = tf.one_hot(indices, len(self.nb_actions))
            # batch_size = [0]
            # loc_indices = tf.tile(loc_indices, tf.stack([batch_size, 1, 1]))

            i = 0
            Mask = np.zeros(len(self.nb_actions)*2)
            for j in self.adjacent_list[i]:
                Mask[j] = 1
                Mask[j + len(self.nb_actions)]  = 1
            Mask = tf.Variable(Mask, dtype=tf.float32)
            local_obs = tf.multiply(obs,Mask)
            # local_obs_joint.append(local_obs)
            x = local_obs
            hidden = 0
            loc_weights = []
            # output_biases = []
            for hidden_size in self.hidden_sizes:
                loc_weight = tf.get_variable('loc_weight_' + str(hidden), [hidden_size], tf.float32, trainable=True)
                loc_weights.append(loc_weight)
                x = tf.layers.dense(x,
                                    hidden_size, name = 'hidden_layer_' + str(hidden))  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                x = tf.add(x, loc_weight)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                if self.batch_norm:
                    x = tc.layers.batch_norm(x)
                x = self.hidden_nonlinearity(x)
                hidden += 1
            output_bias = tf.get_variable('output_bias_' + str(i), [self.nb_actions[i]], tf.float32, trainable=True)
            # output_biases.append(output_bias)
            x = tf.layers.dense(x, self.nb_actions[i])
            x = tf.add(x, output_bias)
            output.append(x)

            for i in range(1,len(self.nb_actions)):
                Mask = np.zeros(len(self.nb_actions) * 2)
                for j in self.adjacent_list[i]:
                    Mask[j] = 1
                    Mask[j + len(self.nb_actions)] = 1
                Mask = tf.Variable(Mask, dtype=tf.float32)
                local_obs = tf.multiply(obs , Mask)
                # local_obs_joint.append(local_obs)
                x = local_obs
                hidden = 0
                for hidden_size in self.hidden_sizes:

                    x = tf.layers.dense(x,
                                        hidden_size, name='hidden_layer_' + str(
                            hidden), reuse=True)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    x = tf.add(x, loc_weights[hidden])
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                    hidden += 1
                output_bias = tf.get_variable('output_bias_' + str(i), [self.nb_actions[i]], tf.float32, trainable=True)
                x = tf.layers.dense(x, self.nb_actions[i])
                x = tf.add(x, output_bias)
                output.append(x)
            x = tf.stack(output, axis=1)
        return x


class CollectiveDecCriticGridNObs(Model):
    def __init__(self, nb_actions, name='dec_collective_critic', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[] , stateNum = 0):
        super(CollectiveDecCriticGridNObs, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        self.stateNum = stateNum
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            # local_obs_joint = []
            for i in range(len(self.nb_actions)):
                local_obs = [obs[:,j] for j in self.adjacent_list[i]]
                local_obs = tf.stack(local_obs, axis=1)
                # local_obs_joint.append(local_obs)
                x = local_obs
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                x = tf.layers.dense(x, self.nb_actions[i])
                output.append(x)
            x = tf.stack(output, axis=1)
        return x

class CollectiveDecActorTaxi(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[]):
        super(CollectiveDecActorTaxi, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            # local_obs_joint = []
            for i in range(len(self.nb_actions)):
                local_obs = [obs[:,48 + j] for j in self.adjacent_list[i]]
                local_obs.extend([obs[:, 48 + 81 + j] for j in self.adjacent_list[i]])
                local_obs = tf.stack(local_obs, axis=1)
                local_obs = tf.concat([local_obs, obs[:,:48]], axis=1)
                # local_obs_joint.append(local_obs)
                x = local_obs
                adjacent_num = len(self.adjacent_list[i])
                # local_output = tf.zeros([None, adjacent_num], tf.float64)
                local_output = [obs[:,0]*0.0]*81
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



class CollectiveDecActorTaxi0Obs(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[]):
        super(CollectiveDecActorTaxi0Obs, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            # local_obs_joint = []
            for i in range(len(self.nb_actions)):
                # local_obs_joint.append(local_obs)
                x = obs[:,:48]
                adjacent_num = len(self.adjacent_list[i])
                # local_output = tf.zeros([None, adjacent_num], tf.float64)
                local_output = [obs[:,0]*0.0]*81
                x = tf.layers.dense(x, adjacent_num)
                x = tf.nn.softmax(x)
                for j in range(len(self.adjacent_list[i])):
                    local_output[self.adjacent_list[i][j]] = x[:,j]
                local_output = tf.stack(local_output, axis=1)
                output.append(local_output)
            x = tf.stack(output, axis=1)
            # x = tf.reshape(x, [-1])
        return x



class CollectiveDecCriticTaxi0Obs(Model):
    def __init__(self, nb_actions, name='dec_collective_critic', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[]):
        super(CollectiveDecCriticTaxi0Obs, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            # local_obs_joint = []
            for i in range(len(self.nb_actions)):
                x = obs[:, :48]
                adjacent_num = len(self.adjacent_list[i])
                # local_output = tf.zeros([None, adjacent_num], tf.float64)
                local_output = [obs[:,0]*0.0]*81
                x = tf.layers.dense(x, adjacent_num)
                for j in range(len(self.adjacent_list[i])):
                    local_output[self.adjacent_list[i][j]] = x[:,j]
                local_output = tf.stack(local_output, axis=1)
                output.append(local_output)
            x = tf.stack(output, axis=1)
            # x = tf.reshape(x, [-1])
        return x



class CollectiveDecCriticTaxi(Model):
    def __init__(self, nb_actions, name='dec_collective_critic', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[]):
        super(CollectiveDecCriticTaxi, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            # local_obs_joint = []
            for i in range(len(self.nb_actions)):
                local_obs = [obs[:,48 + j] for j in self.adjacent_list[i]]
                local_obs.extend([obs[:, 48 + 81 + j] for j in self.adjacent_list[i]])
                local_obs = tf.stack(local_obs, axis=1)
                local_obs = tf.concat([local_obs, obs[:,:48]], axis=1)
                # local_obs_joint.append(local_obs)
                x = local_obs
                adjacent_num = len(self.adjacent_list[i])
                # local_output = tf.zeros([None, adjacent_num], tf.float64)
                local_output = [obs[:,0]*0.0]*81
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


class CollectiveDecInitializationTaxi(Model):
    def __init__(self, nb_actions, name='dec_collective_initialization_actor', zone_number = 81):
        super(CollectiveDecInitializationTaxi, self).__init__(name=name)
        self.zone_number = zone_number

    def __call__(self, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = tf.get_variable('x', [self.zone_number], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)
            x = tf.nn.softmax(x)
        return x

class CollectiveDecActorTaxiDecObs(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[]):
        super(CollectiveDecActorTaxiDecObs, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            # y = []
            # local_obs_joint = []
            for i in range(len(self.nb_actions)):
                x = obs[:, i]
                adjacent_num = len(self.adjacent_list[i])
                # local_output = tf.zeros([None, adjacent_num], tf.float64)
                local_output = [obs[:,0,0]*0.0]*81
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




class CollectiveDecActorTaxiN(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[]):
        super(CollectiveDecActorTaxiN, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = adjacent_list
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            obs = obs*8000.0
            # y = []
            # local_obs_joint = []
            for i in range(len(self.nb_actions)):
                local_obs = [obs[:,48 + j] for j in self.adjacent_list[i]]
                local_obs.extend([obs[:, 48 + 81 + j] for j in self.adjacent_list[i]])
                local_obs = tf.stack(local_obs, axis=1)
                local_obs = tf.concat([local_obs, obs[:,:48]], axis=1)
                # local_obs_joint.append(local_obs)
                x = local_obs
                adjacent_num = len(self.adjacent_list[i])
                # local_output = tf.zeros([None, adjacent_num], tf.float64)
                local_output = [obs[:,0]*0.0]*81
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

class CollectiveDecActor(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=True, batch_norm = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(CollectiveDecActor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            # obs = obs*8000.0
            for i in range(len(self.nb_actions)):
                x = obs[:,i]
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x, hidden_size)#, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                x = tf.layers.dense(x, self.nb_actions[i])#,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
            #x = tf.reshape(x, [-1])
        return x

class CollectiveDecCritic(Model):
    def __init__(self, nb_actions, name='dec_collective_critic', layer_norm=True, batch_norm = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(CollectiveDecCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            output = []
            for i in range(len(self.nb_actions)):
                x = obs[:, i]
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x, hidden_size)#,
                                        # kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                x = tf.layers.dense(x, self.nb_actions[i],
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                output.append(x)
            x = tf.stack(output, axis=1)
        return x


class CollectiveDecActorMinOut(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=True, batch_norm = True, num_channels = 10, num_maxout_units = 2, hidden_nonlinearity = tf.nn.relu):
        super(CollectiveDecActor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.num_channels = num_channels
        self.num_maxout_units = num_maxout_units
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            for i in range(len(self.nb_actions)):
                x = obs[:,i]
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x, hidden_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    x = self.hidden_nonlinearity(x)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                x = tf.layers.dense(x, self.nb_actions[i],
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
            #x = tf.reshape(x, [-1])
        return x

class CollectiveDecCriticMinOut(Model):
    def __init__(self, nb_actions, name='dec_collective_critic', layer_norm=True, batch_norm = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(CollectiveDecCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            output = []
            for i in range(len(self.nb_actions)):
                x = obs[:, i]
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x, hidden_size,
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    x = self.hidden_nonlinearity(x)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                x = tf.layers.dense(x, self.nb_actions[i],
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                output.append(x)
            x = tf.stack(output, axis=1)
        return x






class GridCollectiveCritic(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=True,  relu_output = True, num_channels = 20, num_maxout_units = 5):
        super(GridCollectiveCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.num_channels = num_channels
        self.num_maxout_units = num_maxout_units

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # if self.relu_output:
            output = []
            for i in range(len(self.nb_actions)):
                x = obs[:,i]
                x = tf.layers.dense(x, self.num_channels)
                x = minout(x, self.num_maxout_units)
                x = tf.layers.dense(x, self.nb_actions[i])
                output.append(x)
            x = tf.stack(output, axis=1)
        return x

class GridCollectiveActor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True, num_channels = 20, num_maxout_units = 5):
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
                x = obs[:,i]
                x = tf.layers.dense(x, self.num_channels,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = minout(x, self.num_maxout_units)
                x = tf.layers.dense(x, self.nb_actions[i],
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
        return x


class TaxiBasicCollectiveCritic(Model):
    def __init__(self, nb_actions, name='collective_critic', layer_norm=True,  relu_output = True, hidden_nonlinearity = tf.nn.relu):
        super(TaxiBasicCollectiveCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            output = []
            #time
            output.append(obs[:,0,:self.H])
            #payment
            immediateRewards = []
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                immediateRewards.append(tf.reduce_min(tf.stack([x, obs[:, i, self.H + 1]], axis=1), axis=1))
            immediateRewards = tf.stack(immediateRewards, axis=1)
            output.append(immediateRewards)
            #cost
            flatten_action_count = tf.reshape(action_count, shape=[-1, (action_count.shape[1]*action_count.shape[2]).value])
            output.append(flatten_action_count)
            output = tf.concat(output, axis=1)
            x = tf.layers.dense(output, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
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
            # time
            time = obs[:, 0, :self.H]

            normalizedDenseCounter = 0
            normalizedDenseCounter += 1

            state_count = obs[:, :, self.H ]#obs[:,self.H:self.H + self.zoneNum]
            demand_count = obs[:, :, self.H + 1]#obs[:,self.H + self.zoneNum:]
            initializer = tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5)#tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float64)
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
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
                x = minout(x, self.num_maxout_units)
                if self.layer_norm != None:
                    x = self.layer_norm(x)
                x = tf.concat([x, time], axis=1)
                x = tf.layers.dense(x, self.nb_actions[i],
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
            print('initialized actor network')
        return x

class TaxiCollectiveCriticWithCost(Model):
    def __init__(self, nb_actions, costMatrix,  name='collective_critic', layer_norm=True,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(TaxiCollectiveCriticWithCost, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.costMatrix = costMatrix

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputLayer = obs
            output = []


            next_state_count = []
            normalizedDenseCounter = 0
            customer_flow = normalizedDense(obs[:, :, self.H + 1], self.zoneNum, counter = normalizedDenseCounter)
            state_count = obs[:, :, self.H]
            demand_count = obs[:, :, self.H + 1]
            normalizedDenseCounter +=1
            #payment
            immediateRewards = []
            cost = tf.multiply(action_count, -self.costMatrix)
            output1 = []
            output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                # next_state_count.append(x - demand_count[:,i] + customer_flow[:,i])
                stateReward = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                stateReward = tf.multiply(trip_weights[i],stateReward)
                output1.append(stateReward)
                output2.append(tf.reduce_sum(cost[:,:,i], axis=1))
                stateReward = tf.reduce_sum(tf.stack([tf.squeeze(stateReward), tf.reduce_sum(cost[:,:,i], axis=1)], axis=1), axis=1)
                immediateRewards.append(stateReward)
                output.append(stateReward)
            immediateRewards = tf.stack(immediateRewards, axis=1)


            # output.append(flatten_action_count)

            # #future reward
            # next_state_count = tf.stack(next_state_count, axis=1)
            # x = normalizedDense(next_state_count, self.zoneNum, counter = normalizedDenseCounter)
            # normalizedDenseCounter += 1
            # d = tf.get_variable("d", [self.zoneNum], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=0, maxval=1.0))
            # x = tf.minimum(x, d)
            # output.append(x)
            output = tf.stack(output, axis=1)

            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'state_reward':immediateRewards, 'output1':output1, 'output2':output2, 'cost':cost}


class TaxiCollectiveCriticWithCostAndBiasOld(Model):
    def __init__(self, nb_actions, costMatrix,  name='collective_critic', layer_norm=True,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(TaxiCollectiveCriticWithCostAndBiasOld, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.costMatrix = costMatrix

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputLayer = obs
            output = []


            next_state_count = []
            normalizedDenseCounter = 0
            customer_flow = normalizedDense(obs[:, :, self.H + 1], self.zoneNum, counter = normalizedDenseCounter)
            state_count = obs[:, :, self.H]
            demand_count = obs[:, :, self.H + 1]
            normalizedDenseCounter +=1
            #payment
            immediateRewards = []
            futureRewards = []
            cost = tf.multiply(action_count, -self.costMatrix)
            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)
            # future reward
            d = tf.get_variable("d", [self.zoneNum], trainable=True,
                                initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
            future_trip_weights = tf.get_variable('rd', [len(self.nb_actions)], tf.float32,
                                           tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                           trainable=True)
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                # next_state_count.append(x - demand_count[:,i] + customer_flow[:,i])
                stateReward = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                stateReward = tf.multiply(trip_weights[i],stateReward)
                biasReward = tf.minimum(x, d[i])
                future_reward = tf.multiply(biasReward, future_trip_weights[i])
                immediateRewards.append(stateReward)
                stateReward = tf.add(stateReward, future_reward)
                futureRewards.append(stateReward)
                # output1.append(stateReward)
                # output2.append(tf.reduce_sum(cost[:,:,i], axis=1))
                stateReward = tf.add(stateReward, tf.reduce_sum(cost[:,:,i], axis=1))
                output.append(stateReward)
            immediateRewards = tf.stack(immediateRewards, axis=1)


            # output.append(flatten_action_count)

            # #future reward
            # next_state_count = tf.stack(next_state_count, axis=1)
            # x = normalizedDense(next_state_count, self.zoneNum, counter = normalizedDenseCounter)
            # normalizedDenseCounter += 1
            # d = tf.get_variable("d", [self.zoneNum], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=0, maxval=1.0))
            # x = tf.minimum(x, d)
            # output.append(x)
            output = tf.stack(output, axis=1)

            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'immediateRewards':immediateRewards, 'cost':cost, 'd':d, 'future_trip_weights':future_trip_weights, 'trip_weights':trip_weights}

class TaxiCollectiveCriticWithCostAndBias(Model):
    def __init__(self, nb_actions, costMatrix,  name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(TaxiCollectiveCriticWithCostAndBias, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.N = 8000.0
        self.costMatrix = costMatrix
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputLayer = obs
            output = []
            time_period = obs[:, 0, :self.H]*self.N


            next_state_count = []
            normalizedDenseCounter = 0


            demand_count = obs[:, :, self.H + 1]
            normalizedDenseCounter +=1
            #payment
            state_returns = []
            cost = tf.multiply(action_count, -self.costMatrix)
            # output1 = []
            # output2 = []
            # trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float64, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
            #                     trainable=True)
            trip_weights = []
            # served_demands = []
            features = []

            state_count = []
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                state_count.append(x)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                features.append(x)
                features.append(served_demand)
            #     served_demands.append(served_demand)
            #
            # served_demands = tf.stack(served_demands, axis=1)
            features = tf.stack(features, axis=1)
            features = tf.concat([features, time_period], axis=1)
            d = []
            for i in range(len(self.nb_actions)):
                x = features
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x)
                    x = self.hidden_nonlinearity(x)
                trip_weight = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))[:,0]
                trip_weights.append(trip_weight)
                val_UB = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))[:,0]
                d.append(val_UB)
                stateReward = tf.multiply(trip_weight,tf.reduce_min(tf.stack([state_count[i],val_UB], axis=1), axis=1))
                stateReward = tf.add(stateReward, tf.reduce_sum(cost[:, :, i], axis=1))
                state_returns.append(stateReward)
                output.append(stateReward)

            d = tf.stack(d, axis=1)
            trip_weights = tf.stack(trip_weights, axis=1)
            output = tf.stack(output, axis=1)
            state_returns = tf.stack(state_returns, axis=1)

            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'state_returns':state_returns, 'cost':cost, 'd':d, 'trip_weights':trip_weights}




class TaxiCollectiveCriticWithCostAndBiasVPN(Model):
    def __init__(self, nb_actions, costMatrix,  name='collective_critic', layer_norm=True,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(TaxiCollectiveCriticWithCostAndBiasVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.N = 8000.0
        self.costMatrix = costMatrix

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
            cost = tf.multiply(action_count, -self.costMatrix)
            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)

            served_demands = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                stateReward = tf.multiply(trip_weights[i],served_demand)
                stateReward = tf.add(stateReward, tf.reduce_sum(cost[:, :, i], axis=1))
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
            #                     tf.float64, initializer,
            #                     trainable=True)
            V_norm = tf.nn.softmax(V, dim=2)
            customer_flow = tf.squeeze(tf.matmul(tf.reshape(served_demands, [-1, 1, self.zoneNum]), V_norm))#tf.matmul(served_demands, V_norm)

            # predict the next upper bound
            d = tf.layers.dense(time_period, self.zoneNum, activation=tf.nn.relu, kernel_initializer=tf.random_uniform_initializer(minval=100, maxval=self.N), name= "d", use_bias=False)

            # d = tf.get_variable("d", [self.zoneNum], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
            # predict the next payment
            future_trip_weights = tf.layers.dense(time_period, len(self.nb_actions), activation=tf.nn.relu, kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3), use_bias=False)
            # future_trip_weights = tf.get_variable('rd', [len(self.nb_actions)], tf.float64,
            #                                       tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
            #                                       trainable=True)

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                #future reward
                next_x = x - served_demands[:,i] + customer_flow[:,i]
                next_state_count.append(next_x)
                future_val = tf.minimum(tf.multiply(next_x, future_trip_weights[:,i]), d[:,i])
                # future_val = tf.multiply(future_served_demand, future_trip_weights[:,i])
                future_vals.append(future_val)
                output.append(future_val)

            future_vals = tf.stack(future_vals, axis=1)

            output = tf.stack(output, axis=1)

            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'V_norm':V_norm, 'customer_flow':customer_flow, 'immediateRewards':immediateRewards, 'cost':cost, 'd':d, 'future_trip_weights':future_trip_weights, 'trip_weights':trip_weights, 'next_vals':future_vals, 'next_state_count':next_state_count}

class TaxiCollectiveCriticWithCostAndBiasDenseVPN(Model):
    def __init__(self, nb_actions, costMatrix,  name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu, adjacent_list = []):
        super(TaxiCollectiveCriticWithCostAndBiasDenseVPN, self).__init__(name=name)
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
        self.costMatrix = costMatrix
        self.adjacent_list = adjacent_list

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
            cost = tf.multiply(action_count, -self.costMatrix)
            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)

            served_demands = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                stateReward = tf.multiply(trip_weights[i],served_demand)
                stateReward = tf.add(stateReward, tf.reduce_sum(cost[:, :, i], axis=1))
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
            #                     tf.float64, initializer,
            #                     trainable=True)
            V_norm = tf.nn.softmax(V, dim=2)
            customer_flow = tf.matmul(tf.reshape(served_demands, [-1, 1, self.zoneNum]), V_norm)[:,0,:]#tf.matmul(served_demands, V_norm)

            # predict the next upper bound
            d = tf.layers.dense(time_period, self.zoneNum, activation=tf.nn.relu, kernel_initializer=tf.random_uniform_initializer(minval=100, maxval=self.N*100), name= "d", use_bias=False)

            # d = tf.get_variable("d", [self.zoneNum], trainable=True,
            #                     initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
            # predict the next payment
            future_trip_weights = []

            # next_local_obs = []
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                # future reward
                next_x = x - served_demands[:, i] + customer_flow[:, i]
                next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)

            for i in range(len(self.nb_actions)):
                local_state_count = tf.stack([next_state_count[:, k] for k in self.adjacent_list[i]], axis=1)
                feature = tf.concat([local_state_count, time_period], axis=1)
                for h in self.hidden_sizes:
                    feature = tf.layers.dense(feature,h)
                    if self.layer_norm:
                        feature = tc.layers.layer_norm(feature, center=True, scale=True)
                    if self.batch_norm:
                        feature = tc.layers.batch_norm(feature)
                    feature = self.hidden_nonlinearity(feature)
                future_trip_weight = tf.layers.dense(feature, 1,
                                                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                                      maxval=3e-3))[:,0]
                future_val = tf.minimum(tf.multiply(next_x, future_trip_weight), d[:,i])
                future_trip_weights.append(future_trip_weight)
                # future_val = tf.multiply(future_served_demand, future_trip_weights[:,i])
                future_vals.append(future_val)
                output.append(future_val)

            future_trip_weights = tf.stack(future_trip_weights, axis=1)
            future_vals = tf.stack(future_vals, axis=1)

            output = tf.stack(output, axis=1)

            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'V_norm':V_norm, 'customer_flow':customer_flow, 'immediateRewards':immediateRewards, 'cost':cost, 'd':d, 'future_trip_weights':future_trip_weights, 'trip_weights':trip_weights, 'next_vals':future_vals, 'next_state_count':next_state_count}


class TaxiCollectiveCriticWithCostAndBiasAndFutureRelationVPN(Model):
    def __init__(self, nb_actions, costMatrix, adjacent_list,  name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(TaxiCollectiveCriticWithCostAndBiasAndFutureRelationVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.N = 8000.0
        self.costMatrix = costMatrix
        self.adjacent_list = adjacent_list
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputLayer = obs
            output = []
            time_period = obs[:, 0, :self.H]*self.N


            next_state_count = []
            normalizedDenseCounter = 0

            state_count = obs[:, :, self.H]
            demand_count = obs[:, :, self.H + 1]
            normalizedDenseCounter +=1
            #payment
            immediateRewards = []
            future_vals = []
            cost = tf.multiply(action_count, -self.costMatrix)
            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)

            served_demands = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                stateReward = tf.multiply(trip_weights[i],served_demand)
                stateReward = tf.add(stateReward, tf.reduce_sum(cost[:, :, i], axis=1))
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
            #                     tf.float64, initializer,
            #                     trainable=True)
            V_norm = tf.nn.softmax(V, dim=2)
            customer_flow = tf.matmul(tf.reshape(served_demands, [-1, 1, self.zoneNum]), V_norm)[:,0,:]#tf.matmul(served_demands, V_norm)

            # predict the threshold values
            d = tf.layers.dense(time_period, self.zoneNum, kernel_initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5), name= "d")

            # predict the next payment
            future_trip_weights = []#tf.layers.dense(time_period, len(self.nb_actions), kernel_initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
            # future_trip_weights = tf.get_variable('rd', [len(self.nb_actions)], tf.float64,
            #                                       tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
            #                                       trainable=True)

            # next_local_obs = []
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                #future reward
                next_x = x - served_demands[:,i] + customer_flow[:,i]
                next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)

            for i in range(len(self.nb_actions)):
                local_state_count = tf.stack([next_state_count[:, k] for k in self.adjacent_list[i]], axis=1)
                local_demand_count = tf.stack([d[:, k] for k in self.adjacent_list[i]], axis=1)
                next_local_obs = tf.concat([local_state_count, local_demand_count, time_period], axis=1)
                feature = next_local_obs
                for h in self.hidden_sizes:
                    feature = tf.layers.dense(feature,h)
                    if self.layer_norm:
                        feature = tc.layers.layer_norm(feature, center=True, scale=True)
                    if self.batch_norm:
                        feature = tc.layers.batch_norm(feature)
                    feature = self.hidden_nonlinearity(feature)
                future_trip_weight = tf.layers.dense(feature, 1,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))[:,0]
                future_trip_weights.append(future_trip_weight)
                future_served_demand = tf.minimum(next_state_count[:,i], d[:,i])
                future_val = tf.multiply(future_served_demand, future_trip_weight)
                future_vals.append(future_val)
                output.append(future_val)

            future_vals = tf.stack(future_vals, axis=1)
            future_trip_weights = tf.stack(future_trip_weights, axis=1)

            output = tf.stack(output, axis=1)

            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'V_norm':V_norm, 'customer_flow':customer_flow, 'immediateRewards':immediateRewards, 'cost':cost, 'd':d, 'future_trip_weights':future_trip_weights, 'trip_weights':trip_weights, 'next_vals':future_vals}

class TaxiCollectiveCriticDenseVPN(Model):
    def __init__(self, nb_actions, costMatrix, adjacent_list,  name='collective_critic', layer_norm=False, batch_norm = False,  relu_output = True, hidden_sizes = (), hidden_nonlinearity = tf.nn.relu):
        super(TaxiCollectiveCriticDenseVPN, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        # self.actionNum = np.sum(nb_actions)
        self.relu_output = relu_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.zoneNum = 81
        self.H = 48
        self.N = 8000.0
        self.costMatrix = costMatrix
        self.adjacent_list = adjacent_list
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

    def __call__(self, obs, action_count, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputLayer = obs
            output = []
            time_period = obs[:, 0, :self.H]*self.N


            next_state_count = []
            normalizedDenseCounter = 0

            state_count = obs[:, :, self.H]
            demand_count = obs[:, :, self.H + 1]
            normalizedDenseCounter +=1
            #payment
            immediateRewards = []
            future_vals = []
            cost = tf.multiply(action_count, -self.costMatrix)
            # output1 = []
            # output2 = []
            trip_weights = tf.get_variable('r', [len(self.nb_actions)], tf.float32, tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
                                trainable=True)

            served_demands = []

            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                served_demand = tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)#tf.stack([tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1)], axis=1)
                stateReward = tf.multiply(trip_weights[i],served_demand)
                stateReward = tf.add(stateReward, tf.reduce_sum(cost[:, :, i], axis=1))
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
            #                     tf.float64, initializer,
            #                     trainable=True)
            V_norm = tf.nn.softmax(V, dim=2)
            customer_flow = tf.matmul(tf.reshape(served_demands, [-1, 1, self.zoneNum]), V_norm)[:,0,:]#tf.matmul(served_demands, V_norm)

            # predict the threshold values
            d = tf.layers.dense(time_period, self.zoneNum, kernel_initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5), name= "d")

            # predict the next payment
            future_trip_weights = []#tf.layers.dense(time_period, len(self.nb_actions), kernel_initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
            # future_trip_weights = tf.get_variable('rd', [len(self.nb_actions)], tf.float64,
            #                                       tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5),
            #                                       trainable=True)

            # next_local_obs = []
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:, :, i], axis=1)
                #future reward
                next_x = x - served_demands[:,i] + customer_flow[:,i]
                next_state_count.append(next_x)

            next_state_count = tf.stack(next_state_count, axis=1)

            for i in range(len(self.nb_actions)):
                local_state_count = tf.stack([next_state_count[:, k] for k in self.adjacent_list[i]], axis=1)
                local_demand_count = tf.stack([d[:, k] for k in self.adjacent_list[i]], axis=1)
                next_local_obs = tf.concat([local_state_count, local_demand_count, time_period], axis=1)
                feature = next_local_obs
                for h in self.hidden_sizes:
                    feature = tf.layers.dense(feature,h)
                    if self.layer_norm:
                        feature = tc.layers.layer_norm(feature, center=True, scale=True)
                    if self.batch_norm:
                        feature = tc.layers.batch_norm(feature)
                    feature = self.hidden_nonlinearity(feature)
                future_trip_weight = tf.layers.dense(feature, 1,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))[:,0]
                future_trip_weights.append(future_trip_weight)
                future_served_demand = tf.minimum(next_state_count[:,i], d[:,i])
                future_val = tf.multiply(future_served_demand, future_trip_weight)
                future_vals.append(future_val)
                output.append(future_val)

            future_vals = tf.stack(future_vals, axis=1)
            future_trip_weights = tf.stack(future_trip_weights, axis=1)

            output = tf.stack(output, axis=1)

            # time

            # output = append(obs[:, 0, :self.H])

            x = tf.reduce_sum(output, axis=1)
        return {'symbolic_val':x, 'V_norm':V_norm, 'customer_flow':customer_flow, 'immediateRewards':immediateRewards, 'cost':cost, 'd':d, 'future_trip_weights':future_trip_weights, 'trip_weights':trip_weights, 'next_vals':future_vals}


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
            # time
            output.append(obs[:, 0, :self.H])

            next_state_count = []
            normalizedDenseCounter = 0
            customer_flow = normalizedDense(obs[:, :, self.H + 1], self.zoneNum, counter = normalizedDenseCounter)
            state_count = obs[:, :, self.H]
            demand_count = obs[:, :, self.H + 1]
            normalizedDenseCounter +=1
            #payment
            immediateRewards = []
            for i in range(len(self.nb_actions)):
                x = tf.reduce_sum(action_count[:,:,i], axis=1)
                next_state_count.append(x - demand_count[:,i] + customer_flow[:,i])
                immediateRewards.append(tf.reduce_min(tf.stack([x, demand_count[:,i]], axis=1), axis=1))
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
                                initializer=tf.random_uniform_initializer(minval=0, maxval=1.0))
            x = tf.minimum(x, d)
            output.append(x)
            output = tf.concat(output, axis=1)
            x = tf.layers.dense(output, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5))
        return x





class DecActorGrid(Model):
    def __init__(self, nb_actions, name='dec_collective_actor', layer_norm=True, batch_norm=True, hidden_sizes=(),
                 hidden_nonlinearity=tf.nn.relu, adjacent_list=[] , stateNum = 0, N = 1):
        super(DecActorGrid, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.adjacent_list = np.zeros((len(adjacent_list),len(adjacent_list)), dtype=np.float32)
        for i in range(len(adjacent_list)):
            for j in adjacent_list[i]:
                self.adjacent_list[i,j] = 1
        self.stateNum = stateNum
        self.N = float(N)
        # self.obs_mask = np.zeros((81, 48 + 81*2))
        # for i in range(81):
        #     self.obs_mask[i,:48] = 1
        #     self.obs_mask[i,48:48+81] = self.adjacent_array[i]
        #     self.obs_mask[i,48 + 81:] = self.adjacent_array[i]

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = []
            i = 0
            local_state = tf.one_hot(tf.cast(obs[:, i + self.stateNum * 2],tf.int32), self.stateNum)
            obs_masks = tf.matmul(local_state, self.adjacent_list)
            local_obs = tf.concat([obs[:, :self.stateNum]*obs_masks, obs[:, self.stateNum:self.stateNum*2]*obs_masks, local_state], axis=1)
            x = local_obs
            hidden = 0
            for hidden_size in self.hidden_sizes:
                x = tf.layers.dense(x,
                                    hidden_size, name = 'hidden_layer_' + str(hidden))  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True, scope = 'layer_norm_' + str(hidden))
                if self.batch_norm:
                    x = tc.layers.batch_norm(x, name = 'batch_norm_' + str(hidden))
                x = self.hidden_nonlinearity(x)
                hidden += 1
            x = tf.layers.dense(x, int(self.nb_actions[i]), name = 'hidden_layer_' + str(hidden))
            x = tf.nn.softmax(x)
            output.append(x)
            for i in range(1, int(self.N)):
                local_state = tf.one_hot(tf.cast(obs[:, i + self.stateNum * 2],tf.int32), self.stateNum)
                local_obs = tf.concat([obs[:, :self.stateNum * 2], local_state], axis=1)
                x = local_obs
                hidden = 0
                for hidden_size in self.hidden_sizes:
                    x = tf.layers.dense(x,
                                        hidden_size, name='hidden_layer_' + str(
                            hidden), reuse=True)  # , kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True, scope='layer_norm_' + str(hidden), reuse=True)
                    if self.batch_norm:
                        x = tc.layers.batch_norm(x, name='batch_norm_' + str(hidden), reuse=True)
                    x = self.hidden_nonlinearity(x)
                    hidden += 1
                x = tf.layers.dense(x, int(self.nb_actions[i]), name='hidden_layer_' + str(hidden), reuse=True)
                x = tf.nn.softmax(x)
                output.append(x)
            x = tf.stack(output, axis=1)
            # local_obs_joint = tf.stack(local_obs_joint, axis=1)
        return x

def normalizedDense(x, num_units, nonlinearity=None, initializer = tf.random_normal_initializer(0, 0.05), counter = 0):
    ''' fully connected layer '''
    initializer = tf.truncated_normal_initializer(mean=1.0 / 9.0, stddev=1.0 / 90.0, dtype=tf.float32)
    V = tf.get_variable('V_' +str(counter), [int(x.get_shape()[1]),num_units], tf.float32, initializer, trainable=True)
    # with ops.name_scope(None, "softmax_normalize", [V]) as name:
    #     V = ops.convert_to_tensor(V, name="x")
    V_norm = tf.nn.softmax(V, dim=1)
    return tf.matmul(x, V_norm)