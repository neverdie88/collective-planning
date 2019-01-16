import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.FakeVecEnv import FakeVecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.misc_util import (
    parse_args,
    SimpleMonitor
)
import itertools

from baselines.utils.utils import discount_with_dones, td_fictitious_returns, state_returns
from baselines.utils.utils import Scheduler, make_path, find_trainable_variables
from baselines.models.decmodels import CollectiveDecActorTaxi, CollectiveDecCriticTaxi
from baselines.utils.utils import cat_entropy, mse
from baselines.environments.cgmRealTaxi_envNObsCentralizedPenaltyandReward import CGMRealTaxi as CGMRealTaxi


TINY = 1e-8


class Model(object):
    def __init__(self, actor, critic, observation_space, action_space, num_procs,
                 ent_coef=0.01, critic_l2_reg=0.01, actor_lr=1e-4, critic_lr=1e-3, population = 0, rewardScaler = 1.0):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # nact = ac_space.shape
        # nbatch = nenvs*nsteps

        obs0 = tf.placeholder(tf.float32, shape=(None, observation_space.shape[0]), name='obs')
        state_count = []  # tf.placeholder(tf.float32, shape=(None,) + action_space.shape, name='state_count')
        action_count = []
        ADV = []
        R = []
        for component in range(len(action_space.shapes)):
            action_count.append(tf.placeholder(tf.float32, shape=(None, action_space.shapes[component]),
                                               name=('action' + str(component))))
            state_count.append(tf.placeholder(tf.float32, shape=(None, action_space.shapes[component]),
                                              name=('state_count' + str(component))))
            ADV.append(tf.placeholder(tf.float32, shape=(None, action_space.shapes[component]),
                                      name=('adv' + str(component))))
            R.append(tf.placeholder(tf.float32, shape=(None, action_space.shapes[component]),
                                    name=('R' + str(component))))

        action_count = tf.stack(action_count, axis=1, name='joint_actions')
        state_count = tf.stack(state_count, axis=1, name='joint_state_counts')
        ADV = tf.stack(ADV, axis=1, name='joint_ADV')
        R = tf.stack(R, axis=1, name='joint_R')

        actor_tf = actor(obs0)
        critic_tf = critic(obs0)
        # critic_tf_with_actor_tf = critic(obs0, actor_tf*state_count, reuse=True)

        # neglogpac = -tf.reduce_sum(tf.multiply(A, tf.log(self.actor_tf + 1e-8)), axis=1) #tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        entropy = -tf.reduce_mean(tf.reduce_sum(state_count * actor_tf * tf.log(actor_tf + TINY), axis=(1, 2)))

        pg_loss = -tf.reduce_mean(
            tf.reduce_sum(ADV * action_count * tf.log(actor_tf + TINY), axis=(1, 2)))# - 0.01 * entropy

        real_vf_loss = tf.reduce_mean(tf.square(tf.reduce_sum(critic_tf* action_count - R* action_count , axis=(1, 2))))
        vf_loss = tf.reduce_mean(tf.reduce_sum(tf.square(critic_tf - R) * action_count, axis=(1, 2)))

        if critic_l2_reg > 0:
            critic_reg_vars = [var for var in critic_tf.trainable_vars if
                               'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(critic_l2_reg),
                weights_list=critic_reg_vars
            )
            vf_loss += critic_reg

        # loss = pg_loss  + vf_loss * vf_coef #pg_loss + entropy*ent_coef + vf_loss * vf_coef

        actor_params = actor.trainable_vars
        actor_grads = list(zip(tf.gradients(pg_loss, actor_params), actor_params))
        actor_optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr)
        actor_update = actor_optimizer.apply_gradients(actor_grads)
        critic_params = critic.trainable_vars
        critic_grads = list(zip(tf.gradients(vf_loss, critic_params), critic_params))
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr)
        critic_update = critic_optimizer.apply_gradients(critic_grads)

        # pre_critic_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr * 10)
        # pre_critic_update = pre_critic_optimizer.apply_gradients(critic_grads)
        # lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, state_count_samples, action_count_samples, returns, advantages):
            # advs = rewards - values
            # for step in range(len(obs)):
            #     cur_lr = lr.value()
            td_map = {obs0: obs, state_count: state_count_samples, action_count: action_count_samples, R: returns,
                      ADV: advantages}
            # if states != []:
            #     td_map[train_model.S] = states
            #     td_map[train_model.M] = masks
            policy_loss, real_value_loss, value_loss, _, _ = sess.run(
                [pg_loss, real_vf_loss, vf_loss, actor_update, critic_update],
                td_map
            )
            # policy_loss, real_value_loss, value_loss, policy_entropy = sess.run(
            #     [pg_loss, real_vf_loss, vf_loss, entropy],
            #     td_map
            # )
            return policy_loss, real_value_loss, value_loss

        def trainCritic(obs, state_count_samples, action_count_samples, returns):
            # advs = rewards - values
            # for step in range(len(obs)):
            #     cur_lr = lr.value()
            td_map = {obs0: obs, state_count: state_count_samples, action_count: action_count_samples, R: returns}
            # if states != []:
            #     td_map[train_model.S] = states
            #     td_map[train_model.M] = masks
            real_value_loss, value_loss,  _ = sess.run(
                [real_vf_loss, vf_loss, critic_update],
                td_map
            )
            return real_value_loss, value_loss

        self.trainCritic  = trainCritic

        def step(obs):
            return sess.run(actor_tf, {obs0: obs})

        def value(obs):
            return sess.run(critic_tf, {obs0: obs})

        saver = tf.train.Saver()

        def save(save_path, fileName, checkPoint):
            # actor_param_values = sess.run(actor_params)
            # critic_param_values = sess.run(critic_params)
            make_path(save_path)
            save_file = saver.save(sess, save_path + fileName, global_step=checkPoint)
            print("Model saved in file: %s" % save_file)

        def saveActor(save_path, fileName):
            actor_param_values = sess.run(actor_params)
            make_path(save_path)
            joblib.dump(actor_param_values, save_path + fileName)

        self.saveActor = saveActor

        def load(load_path,):
            saver.restore(sess, load_path)

        self.load = load

        def get_actor_params():
            return sess.run(actor_params)

        def get_critic_params():
            return sess.run(critic_params)

        #
        # def load(load_path):
        #     loaded_params = joblib.load(load_path)
        #     restores = []
        #     for p, loaded_p in zip(params, loaded_params):
        #         restores.append(p.assign(loaded_p))
        #     ps = sess.run(restores)
        self.get_critic_params = get_critic_params
        self.get_actor_params = get_actor_params
        self.train = train
        self.actor = actor_tf
        self.critic = critic_tf
        self.step = step
        self.value = value
        self.save = save
        self.population = float(population)
        self.rewardScaler = rewardScaler
        # self.load = load
        tf.global_variables_initializer().run(session=sess)


class Runner(object):
    def __init__(self, env, model, nsteps=5, nstack=4, gamma=0.99, lam=0.8, vf_normalization=False):
        self.env = env
        self.model = model
        # nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.nenv = nenv
        self.batch_ob_shape = (nenv * nsteps, env.observation_space.shape[0])
        self.batch_ac_shape = (nenv * nsteps, len(env.action_space.shapes), env.action_space.shapes[0])
        self.batch_inflow_shape = (nenv * nsteps, env.action_space.shapes[0])
        self.batch_ac_state_shape = (nenv * nsteps, len(env.action_space.shapes))
        self.batch_ac_unflatten_shape = (nenv, nsteps, len(env.action_space.shapes), env.action_space.shapes[0])
        self.obs = np.zeros((nenv, env.observation_space.shape[0]), dtype=np.float32)
        self.obs, self.stateCount = env.reset()
        # self.update_obs(obs)
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]
        self.actor_params = []
        self.critic_params = []
        self.vf_normalization = vf_normalization
        self.epoch = 0

    # def update_obs(self, obs):
    #     self.obs = np.roll(self.obs, shift=-1, axis=3)
    #     self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def run(self):
        mb_obs, mb_actions, mb_rewards, mb_action_probs, mb_dones = [], [], [], [], []
        mb_counts = []
        mb_atomic_rewards = []
        total_shortage = 0
        total_ratio = 0
        total_missing = 0
        mb_trip_pays = []
        mb_penalties = []
        total_local_trip_pay = 0
        for n in range(self.nsteps):
            action_probs = self.model.step(self.obs)
            mb_obs.append(np.copy(self.obs))
            mb_action_probs.append(action_probs)
            mb_dones.append(self.dones)
            obs, action, rewards, dones, info = self.env.step(action_probs)
            total_shortage += np.sum([info[i]['shortage'] for i in range(len(info))])
            total_ratio += np.sum([info[i]['ratio'] for i in range(len(info))])
            total_missing += np.sum([info[i]['missing'] for i in range(len(info))])
            total_local_trip_pay += np.sum([info[i]['local_trip_pay'] for i in range(len(info))])
            count = [info[i]['count'] for i in range(len(info))]
            atomic_rewards = [info[i]['action_rewards'] for i in range(len(info))]
            trip_pay = [info[i]['trip_pay'] for i in range(len(info))]
            mb_trip_pays.append(trip_pay)
            penalty = [info[i]['penalty'] for i in range(len(info))]
            mb_penalties.append(penalty)
            mb_counts.append(count)
            mb_atomic_rewards.append(atomic_rewards)
            # vf = self.model.value(self.obs)
            mb_actions.append(action)
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            self.obs = obs
            mb_rewards.append(rewards)


        mb_dones.append(self.dones)

        print('total_shortage:' + str(total_shortage))
        print('total_ratio:' + str(total_ratio))
        print('missing:' + str(total_missing))
        print('total_revenue:' + str(np.sum(mb_trip_pays)))
        print('total_local_trip_pay:' + str(total_local_trip_pay))

        mb_counts = np.asarray(mb_counts, dtype=np.float32).swapaxes(1, 0)
        mb_atomic_rewards = np.asarray(mb_atomic_rewards, dtype=np.float32).swapaxes(1, 0)/self.model.rewardScaler
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_values = self.model.value(mb_obs).reshape(self.batch_ac_unflatten_shape)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_action_probs = np.asarray(mb_action_probs, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.float32).swapaxes(1, 0)

        mb_penalties = np.asarray(mb_penalties, dtype=np.float32).swapaxes(1, 0) / self.model.rewardScaler

        mb_penalties_returns = []
        for n, atomic_rewards in enumerate(mb_penalties):
            rewards = state_returns(atomic_rewards, self.gamma)
            # mb_returns.append(np.sum(rewards, axis=1))
            mb_penalties_returns.append(rewards)

        mb_penalties_returns = np.array(mb_penalties_returns).reshape(
            self.batch_ac_state_shape) / self.model.population

        # mb_values += np.sum(mb_penalties_returns, axis=1)[np.newaxis, :, np.newaxis, np.newaxis]

        # action_grads = mb_values.reshape(self.batch_ac_shape)
        #
        # if self.epoch == 0:
        #     # Thien check dimension here
        #     self.meanAds = np.sum(action_grads * mb_actions) / (len(action_grads) * self.model.population)
        #     self.stdAds = np.sqrt(np.sum(np.square(action_grads - self.meanAds) * mb_actions) / (
        #     len(action_grads) * self.model.population))
        #     self.epoch = 1
        # else:
        #     self.meanAds1 = np.sum(action_grads * mb_actions) / (len(action_grads) * self.model.population)
        #     self.stdAds1 = np.sqrt(np.sum(np.square(action_grads - self.meanAds) * mb_actions) / (
        #     len(action_grads) * self.model.population))
        #     self.meanAds = 0.9 * self.meanAds1 + 0.1 * self.meanAds
        #     self.stdAds = 0.9 * self.stdAds1 + 0.1 * self.stdAds


        # mb_baselines = (mb_values * mb_action_probs).reshape(self.batch_ac_shape)
        # mb_baselines[:, :, :] = np.sum(mb_baselines, axis=2)[:, :, np.newaxis]  # HACK

        mb_state_values = mb_values * mb_actions
        mb_state_values = mb_state_values.reshape(self.batch_ac_unflatten_shape)
        mb_state_values = (
            mb_state_values / np.maximum(0.1, mb_actions.reshape(self.batch_ac_unflatten_shape).sum(axis=3))[:, :, :,
                              np.newaxis])
        mb_state_values = mb_state_values.sum(axis=3)


        average_reward = (np.sum(mb_rewards) / self.env.num_envs)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)#/self.model.rewardScaler

        mb_advantages = mb_values.reshape(self.batch_ac_shape)  + np.sum(mb_penalties_returns, axis=1)[:, np.newaxis, np.newaxis]#- mb_baselines

        action_count = np.array(mb_actions).reshape(self.batch_ac_shape)
        action_count[:, :, :] = np.sum(action_count, axis=2)[:, :, np.newaxis]  # HACK
        mb_states = action_count.reshape(self.batch_ac_shape)
        mb_action_probs = mb_action_probs.reshape(self.batch_ac_shape)

        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        last_values = [0.0] * len(mb_rewards)
        self.obs, self.stateCount = self.env.reset()

        for n, (state_values, action_counts, counts, atomic_rewards) in enumerate(
                zip(mb_state_values, mb_actions.reshape(self.batch_ac_unflatten_shape), mb_counts, mb_atomic_rewards)):
            next_values = np.array(state_values)
            next_values[:-1] = next_values[1:]
            next_values[-1] = last_values[n]
            # td_fictitious_returns(rewards, state_values, counts, action_counts, gamma, lam):
            rewards = td_fictitious_returns(atomic_rewards, next_values, counts, action_counts, self.gamma, self.lam)
            mb_atomic_rewards[n] = rewards
        # mb_atomic_returns = np.asarray(mb_atomic_returns).reshape(self.batch_ac_shape)
        mb_atomic_rewards = mb_atomic_rewards.reshape(self.batch_ac_shape)
        mb_masks = mb_masks.flatten()
        mb_states = mb_states/self.model.population
        mb_actions = mb_actions/self.model.population
        return mb_obs, mb_actions.reshape(
            self.batch_ac_shape), mb_states, mb_atomic_rewards, mb_masks, mb_action_probs, mb_advantages, mb_values, average_reward




def learn(actor, critic, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), ent_coef=0.01, gamma=0.99,
          lam=0.8, log_interval=1, actor_lr=1e-4, critic_lr=1e-3, critic_l2_reg=0.01, population=0, rewardScaler=0.0,
          layer_norm=False, batch_norm=False, satisfied_percentage=0.5, critic_training_type=1, vf_normalization=False,
          max_var=10.0, penalty_weight=1.0):
    tf.reset_default_graph()
    set_global_seeds(seed)
    norm_type = ''
    if layer_norm:
        norm_type = 'LayerNorm'
    if batch_norm:
        norm_type = 'BatchNorm'
    if vf_normalization:
        norm_type += '.criticNormalization'

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK
    model = Model(actor=actor, critic=critic, observation_space=ob_space, action_space=ac_space,
                  num_procs=num_procs, ent_coef=ent_coef, actor_lr=actor_lr, critic_lr=critic_lr,
                  critic_l2_reg=critic_l2_reg, population = population, rewardScaler = rewardScaler)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma, lam=lam)

    nbatch = nenvs * nsteps
    tstart = time.time()
    average_running_reward = 0

    print('Intializing with training critic function')
    for k in range(100):
        obs, actions, states, returns, masks, action_probs, advantages, _, average_reward = runner.run()
        real_value_loss,  value_loss = model.trainCritic(obs, states, actions, returns)
        print(str(k) + ',' + str(value_loss) + ',' + str(real_value_loss))
    # model.load('a2cFictitiousSamplingTaxiDecNormalizedObsRewardScaler/a2cFictitiousSamplingTaxiDecNormalizedObsRewardScaler.param-9000')
    # model.saveActor('./a2cFictitiousSamplingTaxiDecNormalizedObsRewardScaler/', 'a2cFictitiousSamplingTaxiDecNormalizedObsRewardScaler.actor')
    for update in range(1, 21001):
        obs, actions, states, returns, masks, action_probs, advantages, _, average_reward = runner.run()
        average_running_reward += average_reward
        # (obs, state_count_samples, action_count_samples, returns)
        policy_loss, real_value_loss,  value_loss = model.train(obs, states, actions, returns, advantages)
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0:
            # ev = explained_variance(values, returns)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("real_value_loss", float(real_value_loss))
            logger.record_tabular("policy_loss", policy_loss)
            # logger.record_tabular("max_predict", np.max(abs(values)))
            logger.record_tabular("max_return", np.max(abs(returns)))
            # logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("return", (float(average_running_reward) / log_interval))
            logger.dump_tabular()
            # if update % 1000 == 0:
                # model.save(
                #     './FACTaxi' + norm_type + '.criticTraining' + str(
                #         critic_training_type) + '.lr' + str(actor_lr) + '-' + str(
                #         critic_lr) + '.satisfied_percentage' + str(satisfied_percentage) + '.max_var' + str(
                #         max_var) + '.penalty_weight' + str(penalty_weight) + '/',
                #     'FACTaxi' + norm_type + '.param',
                #     checkPoint=update)
            average_running_reward = 0
    env.close()


def computing_adjacent_array(adjacentMatrix):
    adjacent_array = np.zeros((81, 81))
    for i in range(len(adjacentMatrix)):
        # limited view into current zone
        for j in range(9):
            if (adjacentMatrix[i][j] >= 0):
                adjacent_array[i, adjacentMatrix[i][j]] = 1
                # print(self.h + j * 2 +1)
    return adjacent_array


def computing_adjacent_list(adjacentMatrix):
    adjacent_list = []
    for i in range(len(adjacentMatrix)):
        # limited view into current zone
        temp = []
        for j in range(9):
            if (adjacentMatrix[i][j] >= 0):
                temp.append(adjacentMatrix[i][j])
                # print(self.h + j * 2 +1)
        # temp[self.t] = 1
        adjacent_list.append(np.array(temp))
    return adjacent_list


def main():
    seed = 42
    args = parse_args()
    nenvs = args['num_cpu']
    max_var = args['max_var']
    penalty_weight = args['penalty_weight']
    env = CGMRealTaxi(8000, 10, args['satisfied_percentage'], penalty_weight)
    penalty_zones = env.high_demand_zones
    populationSize = env.N
    # rewardScaler = env.rewardMagnitude
    rewardScaler = args['reward_scaler']
    costMatrix = env.costMatrix / rewardScaler
    adjacentMatrix = env.adjacentMatrix
    adjacent_list = computing_adjacent_list(adjacentMatrix)
    print('population size, reward magnitude:' + str(populationSize) + "," + str(rewardScaler))
    print('satisfied_percentage, max_var, penalty_weight:' + str(args['satisfied_percentage']) + ',' + str(
        max_var) + ',' + str(penalty_weight))

    env.seed(seed)
    env = FakeVecEnvWrapper(env)
    vf_normalization = args['vf_normalization']
    set_global_seeds(seed)
    hidden_layers = ()
    hidden_unit_nums = args['hidden_unit_num']
    if hidden_unit_nums > 0:
        hidden_layers = (hidden_unit_nums, hidden_unit_nums)
    critic = CollectiveDecCriticTaxi(env.action_space.shapes, layer_norm=args['layer_norm'],
                                   batch_norm=args['batch_norm'],
                                   hidden_sizes=hidden_layers, adjacent_list=adjacent_list)
    actor = CollectiveDecActorTaxi(env.action_space.shapes, layer_norm=args['layer_norm'],
                                   batch_norm=args['batch_norm'],
                                   hidden_sizes=hidden_layers, adjacent_list=adjacent_list)
    # critic = GridCollectiveCritic(env.action_space.shapes)
    # actor = GridCollectiveActor(env.action_space.shapes)
    learn(actor, critic, env, seed, nsteps=48, ent_coef=args['ent_coef'], actor_lr=args['actor_lr'],
          critic_lr=args['critic_lr'], critic_l2_reg=args['critic_l2_reg'], lam=args['lambda'],
          population=populationSize, rewardScaler=rewardScaler, layer_norm=args['layer_norm'],
          batch_norm=args['batch_norm'], satisfied_percentage=args['satisfied_percentage'],
          critic_training_type=args['critic_training_type'], vf_normalization=vf_normalization, max_var=max_var,
          penalty_weight=penalty_weight)

if __name__ == '__main__':
    main()
