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

from baselines.utils.utils import discount_with_dones, td_fictitious_returns, td_returns, fictitious_returns, \
    state_returns
from baselines.utils.utils import Scheduler, make_path, find_trainable_variables
from baselines.models.decmodels import CollectiveDecActorTaxi, CollectiveDecCritic, GridCollectiveCritic, \
    GridCollectiveActor, TaxiBasicCollectiveCritic, TaxiCollectiveActorPseudoFlowMaxout, TaxiCollectiveCritic, \
    TaxiCollectiveCriticWithCost, TaxiCollectiveCriticWithCostAndBiasVPN
from baselines.models.centralized_critic_models import TaxiCentralizedPenaltyandRewardDeepVPN
from baselines.environments.cgmRealTaxi_envNObsCentralizedPenaltyandReward import CGMRealTaxi as CGMRealTaxi
from baselines.utils.utils import flatten
import time

TINY = 1e-8


class Model(object):
    def __init__(self, actor, critic, observation_space, action_space, num_procs,
                 ent_coef=0.01, critic_l2_reg=0.01, population=0, rewardScaler=0.0,
                 critic_training_type=1):
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

        # immediate revenue
        r = tf.placeholder(tf.float32, shape=(None, action_space.shapes[0]))
        # next state accumulated penalty
        next_P = tf.placeholder(tf.float32, shape=(None, action_space.shapes[0]))
        # next state accumulated revenue
        next_R = tf.placeholder(tf.float32, shape=(None, action_space.shapes[0]))
        #
        for component in range(len(action_space.shapes)):
            action_count.append(tf.placeholder(tf.float32, shape=(None, action_space.shapes[component]),
                                               name=('action' + str(component))))
            state_count.append(tf.placeholder(tf.float32, shape=(None, action_space.shapes[component]),
                                              name=('state_count' + str(component))))

        action_count = tf.stack(action_count, axis=1, name='joint_actions')
        state_count = tf.stack(state_count, axis=1, name='joint_state_counts')

        actor_tf = actor(obs0)
        critic_outputs = critic(obs0, action_count)
        critic_tf = critic_outputs['symbolic_val']
        immediateRevenues_tf = critic_outputs['immediateRevenues']
        future_revenues_tf = critic_outputs['future_revenues']
        future_penalty_tf = critic_outputs['future_penalty']
        next_state_count = critic_outputs['next_state_count']

        # critic_tf_with_actor_tf = critic(obs0, actor_tf*state_count, reuse=True)


        ADV = tf.placeholder(tf.float32, shape=(None, len(action_space.shapes), action_space.shapes[0]))
        action_grads = tf.gradients(critic_tf, action_count)
        entropy = -tf.reduce_mean(tf.reduce_sum(state_count * actor_tf * tf.log(actor_tf + TINY), axis=(1, 2)))
        pg_loss = -tf.reduce_mean(
            tf.reduce_sum(ADV * state_count * actor_tf, axis=(1, 2)))/population-ent_coef*entropy/population

        # if critic_training_type == 1:
        #     vf_loss = tf.reduce_mean(tf.reduce_sum(tf.square(immediateRevenues_tf - r), axis=1)) \
        #               + tf.reduce_mean(
        #         tf.reduce_sum(tf.square(future_revenues_tf - next_R + future_penalty_tf - next_P), axis=1))
        # elif critic_training_type == 2:
        vf_loss = tf.reduce_mean(tf.reduce_sum(tf.square(immediateRevenues_tf - r), axis=1))/(population) \
                  + tf.reduce_mean(tf.reduce_sum(tf.square(future_revenues_tf - next_R), axis=1))/(population) \
                  + tf.reduce_mean(tf.reduce_sum(tf.square(future_penalty_tf - next_P), axis=1))/(population)

        # tf.reduce_mean(tf.reduce_sum(tf.square(r_tf - r), axis=1)) +
        vf_immediate_loss = tf.reduce_mean(tf.reduce_sum(tf.square(immediateRevenues_tf - r), axis=1))
        vf_revenue_VF_loss = tf.reduce_mean(tf.reduce_sum(tf.square(future_revenues_tf - next_R), axis=1))
        vf_penalty_VF_loss = tf.reduce_mean(tf.reduce_sum(tf.square(future_penalty_tf - next_P), axis=1))

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
        actor_lr = tf.placeholder(dtype=tf.float32, name='actor_lr')
        critic_lr = tf.placeholder(dtype=tf.float32, name='critic_lr')
        actor_params = actor.trainable_vars
        actor_grads = list(zip(tf.gradients(pg_loss, actor_params), actor_params))
        actor_optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr)
        actor_update = actor_optimizer.apply_gradients(actor_grads)
        critic_params = critic.trainable_vars
        critic_grads = list(zip(tf.gradients(vf_loss, critic_params), critic_params))
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr)
        critic_update = critic_optimizer.apply_gradients(critic_grads)


        def get_action_grads(obs, action_count_samples):
            td_map = {obs0: obs, action_count: action_count_samples}
            return sess.run(action_grads, td_map)

        self.get_action_grads = get_action_grads

        # lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, COMA_ADV, state_count_samples, action_count_samples, sample_r, sample_next_R, sample_next_P, actor_lr_val, critic_lr_val):
            td_map = {obs0: obs, ADV: COMA_ADV, state_count: state_count_samples, action_count: action_count_samples,
                      r: sample_r, next_R: sample_next_R, next_P: sample_next_P, actor_lr:actor_lr_val, critic_lr:critic_lr_val}
            policy_loss, value_loss, _, _, immediate_val_loss, revenue_VF_loss, penalty_VF_loss = sess.run(
                [pg_loss, vf_loss, actor_update, critic_update, vf_immediate_loss, vf_revenue_VF_loss,
                 vf_penalty_VF_loss],#, actor_grads
                td_map
            )#, actor_grad_vals
            return policy_loss, value_loss, immediate_val_loss, revenue_VF_loss, penalty_VF_loss#, actor_grad_vals

        def trainCritic(obs, state_count_samples, action_count_samples, sample_r, sample_next_R, sample_next_P, critic_lr_val):
            td_map = {obs0: obs, state_count: state_count_samples, action_count: action_count_samples, r: sample_r,
                      next_R: sample_next_R, next_P: sample_next_P, critic_lr:critic_lr_val}
            value_loss, _, immediate_val_loss, revenue_VF_loss, penalty_VF_loss = sess.run(
                [vf_loss, critic_update, vf_immediate_loss, vf_revenue_VF_loss, vf_penalty_VF_loss],
                td_map
            )

            return value_loss, immediate_val_loss, revenue_VF_loss, penalty_VF_loss

        self.trainCritic = trainCritic

        def step(obs):
            return sess.run(actor_tf, {obs0: obs})

        def value(obs, action_count_input):
            return sess.run(critic_tf, {obs0: obs, action_count: action_count_input})

        def get_next_state_count(obs, action_count_input):
            return sess.run(next_state_count, {obs0: obs, action_count: action_count_input})

        def cost(obs, action_count_input):
            return sess.run(critic_outputs['cost'], {obs0: obs, action_count: action_count_input})

        def costRaw(obs, action_count_input):
            return sess.run(critic_outputs['cost'], {obs0: obs, action_count: action_count_input})

        def reward(obs, action_count_input):
            return sess.run(critic_outputs['immediateRewards'], {obs0: obs, action_count: action_count_input})

        def next_vals(obs, action_count_input):
            return sess.run(critic_outputs['next_vals'], {obs0: obs, action_count: action_count_input})

        self.cost = cost
        self.costRaw = costRaw
        self.reward = reward
        self.next_vals = next_vals
        saver = tf.train.Saver()

        def save(save_path, fileName, checkPoint):
            # actor_param_values = sess.run(actor_params)
            # critic_param_values = sess.run(critic_params)
            make_path(save_path)
            save_file = saver.save(sess, save_path + fileName, global_step=checkPoint)
            print("Model saved in file: %s" % save_file)

        def load(load_path, ):
            saver.restore(sess, load_path)

        self.load = load

        def loadActor(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(actor_params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.loadActor = loadActor

        def get_actor_params():
            return sess.run(actor_params)

        def get_critic_params():
            return sess.run(critic_params)

        def get_d(obs):
            return sess.run(critic_outputs['d'], {obs0: obs})

        def get_future_trip_weights(obs):
            return sess.run(critic_outputs['future_trip_weights'], {obs0: obs})

        def get_V_norm(obs):
            return sess.run(critic_outputs['V_norm'], {obs0: obs})

        self.get_V_norm = get_V_norm

        def get_customer_flow(obs, action_count_input):
            return sess.run(critic_outputs['customer_flow'], {obs0: obs, action_count: action_count_input})

        self.get_customer_flow = get_customer_flow

        def get_trip_weights():
            return sess.run(critic_outputs['trip_weights'])

        self.get_d = get_d
        self.get_future_trip_weights = get_future_trip_weights
        self.get_trip_weights = get_trip_weights
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
        mb_obs, mb_actions, mb_rewards, mb_action_probs, mb_values = [], [], [], [], []
        mb_trip_pays = []
        mb_penalties = []
        # time1 = time.perf_counter()
        total_shortage = 0
        total_ratio = 0
        total_missing = 0
        total_local_trip_pay = 0

        for n in range(self.nsteps):
            action_probs = self.model.step(self.obs)
            mb_obs.append(np.copy(self.obs))
            mb_action_probs.append(action_probs)
            obs, action, rewards, dones, info = self.env.step(action_probs)
            total_shortage += np.sum([info[i]['shortage'] for i in range(len(info))])
            total_ratio += np.sum([info[i]['ratio'] for i in range(len(info))])
            total_missing += np.sum([info[i]['missing'] for i in range(len(info))])
            total_local_trip_pay += np.sum([info[i]['local_trip_pay'] for i in range(len(info))])
            trip_pay = [info[i]['trip_pay'] for i in range(len(info))]
            penalty = [info[i]['penalty'] for i in range(len(info))]
            mb_trip_pays.append(trip_pay)
            mb_penalties.append(penalty)
            # vf = self.model.value(self.obs)
            # values = self.model.value(self.obs, action / self.model.population)
            # mb_values.append(values)
            mb_actions.append(action)
            self.obs = obs
            mb_rewards.append(rewards)

        # print('total_shortage:' + str(total_shortage))
        # print('total_ratio:' + str(total_ratio))
        # print('missing:' +str(total_missing))
        # print('total_revenue:' + str(np.sum(mb_trip_pays)))
        # print('total_local_trip_pay:' + str(total_local_trip_pay))

        mb_trip_pays = np.asarray(mb_trip_pays, dtype=np.float32).swapaxes(1, 0) / self.model.rewardScaler
        mb_penalties = np.asarray(mb_penalties, dtype=np.float32).swapaxes(1, 0) / self.model.rewardScaler

        # self.model.value(mb_obs, mb_actions[0] / self.model.population)
        mb_action_probs = np.asarray(mb_action_probs, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.float32).swapaxes(1, 0)

        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape)
        average_reward = (np.sum(mb_rewards) / self.env.num_envs)
        average_penalty = np.sum(mb_penalties) / self.env.num_envs
        average_revenue = np.sum(mb_trip_pays) / self.env.num_envs
        # mb_rewards = (np.asarray(mb_rewards, dtype=np.float32).swapaxes(1,
        #                                                                 0) / self.model.rewardScaler) / self.model.population

        action_count = np.array(mb_actions).reshape(self.batch_ac_shape)
        # mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        action_count[:, :, :] = np.sum(action_count, axis=2)[:, :, np.newaxis]  # HACK
        mb_states = action_count.reshape(self.batch_ac_shape)

        mb_action_probs = mb_action_probs.reshape(self.batch_ac_shape)

        self.obs, self.stateCount = self.env.reset()

        mb_next_penalty_returns = []
        mb_next_trip_pay_returns = []
        mb_returns = []
        for n, atomic_rewards in enumerate(mb_penalties):
            rewards = state_returns(atomic_rewards, self.gamma)
            mb_returns.append(np.sum(rewards, axis=1))
            rewards = np.roll(rewards, -1, axis=0)
            rewards[-1, :] = 0
            mb_next_penalty_returns.append(rewards)

        for n, atomic_rewards in enumerate(mb_trip_pays):
            rewards = state_returns(atomic_rewards, self.gamma)
            mb_returns.append(np.sum(rewards, axis=1))
            rewards = np.roll(rewards, -1, axis=0)
            rewards[-1, :] = 0
            mb_next_trip_pay_returns.append(rewards)

        mb_next_penalty_returns = np.array(mb_next_penalty_returns).reshape(
            self.batch_ac_state_shape)  # /self.model.population
        mb_next_trip_pay_returns = np.array(mb_next_trip_pay_returns).reshape(self.batch_ac_state_shape)

        # mb_atomic_returns = np.asarray(mb_atomic_returns).reshape(self.batch_ac_shape)
        # mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions# / self.model.population
        mb_actions = mb_actions.reshape(
            self.batch_ac_shape)
        mb_trip_pays = mb_trip_pays.reshape(self.batch_inflow_shape)
        action_grads = self.model.get_action_grads(mb_obs, mb_states*mb_action_probs)[0]

        mb_states = mb_states# / self.model.population


        if self.vf_normalization:
            if self.epoch == 0:
                # Thien check dimension here
                self.meanAds = np.sum(action_grads * mb_actions) / (len(action_grads)*self.model.population)
                self.stdAds = np.sqrt(np.sum(np.square(action_grads - self.meanAds) * mb_actions) / (len(action_grads)*self.model.population))
                self.epoch = 1
            else:
                self.meanAds1 = np.sum(action_grads * mb_actions) / (len(action_grads)*self.model.population)
                self.stdAds1 = np.sqrt(np.sum(np.square(action_grads - self.meanAds) * mb_actions) / (len(action_grads)*self.model.population))
                self.meanAds = 0.9 * self.meanAds1 + 0.1 * self.meanAds
                self.stdAds = 0.9 * self.stdAds1 + 0.1 * self.stdAds
            action_grads = (action_grads - self.meanAds) / (self.stdAds + 1e-8)
       # mb_baselines = np.sum(action_grads * mb_action_probs, axis=2)
        mb_advantages = action_grads# - mb_baselines[:, :, np.newaxis]
        # print('max advantage, min advantage:' + str(np.max(mb_advantages)) + ',' + str(np.min(mb_advantages)))
        # print(time.perf_counter() - time1)
        return mb_obs, mb_actions, mb_states, mb_action_probs, mb_trip_pays, mb_next_penalty_returns, mb_next_trip_pay_returns, mb_advantages, average_reward, average_penalty, average_revenue


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
    if ent_coef >0.0:
        norm_type += '.ent_coef' + str(ent_coef)
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK
    model = Model(actor=actor, critic=critic, observation_space=ob_space, action_space=ac_space,
                  num_procs=num_procs, ent_coef=ent_coef,
                  critic_l2_reg=critic_l2_reg, population=population, rewardScaler=rewardScaler,
                  critic_training_type=critic_training_type)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma, lam=lam, vf_normalization=vf_normalization)

    nbatch = nenvs * nsteps
    tstart = time.time()
    average_running_reward = 0
    best_value = -10e10

    print('Intializing with training critic function')
    for k in range(100):
        obs, actions, states, action_probs, mb_trip_pays, mb_next_penalty_returns, mb_next_trip_pay_returns, mb_advantages, average_reward, average_penalty, average_revenue = runner.run()
        value_loss, immediate_val_loss, revenue_VF_loss, penalty_VF_loss = model.trainCritic(obs, states, actions,
                                                                                             mb_trip_pays,
                                                                                             mb_next_trip_pay_returns,
                                                                                             mb_next_penalty_returns, critic_lr*10
                                                                                             )
        print(str(k) + ',' + str(value_loss) + ',' + str(immediate_val_loss) + ',' + str(revenue_VF_loss) + ',' + str(
            penalty_VF_loss))

    for update in range(1, 20105):
        obs, actions, states, action_probs, mb_trip_pays, mb_next_penalty_returns, mb_next_trip_pay_returns, mb_advantages, average_reward, average_penalty, average_revenue = runner.run()
        average_running_reward += average_reward
        # (obs, state_count_samples, action_count_samples, returns)
        # time1 = time.perf_counter()
        policy_loss, value_loss, immediate_val_loss, revenue_VF_loss, penalty_VF_loss = model.train(obs,
                                                                                                                 mb_advantages,
                                                                                                                 states,
                                                                                                                 actions,
                                                                                                                 mb_trip_pays,
                                                                                                                 mb_next_trip_pay_returns,
                                                                                                                 mb_next_penalty_returns, actor_lr, critic_lr)
       # min_grad = np.min([np.min(grads) for grads in actor_grad_vals])
       # max_grad = np.min([np.min(grads) for grads in actor_grad_vals])
       # print('max grad, min grad:'+ str(max_grad) +',' + str(min_grad))
        logger.record_tabular("nupdates", update)
        logger.record_tabular("total_timesteps", update * nbatch)
        # logger.record_tabular("fps", fps)
        logger.record_tabular("value_loss", float(value_loss))
        logger.record_tabular("immediate_val_loss", float(immediate_val_loss))
        logger.record_tabular("revenue_VF_loss", float(revenue_VF_loss))
        logger.record_tabular("penalty_VF_loss", float(penalty_VF_loss))
        logger.record_tabular("policy_loss", float(policy_loss))
        logger.record_tabular("return", (float(average_reward)))
        logger.record_tabular("penalty", (float(average_penalty)))
        logger.record_tabular("revenue", (float(average_revenue)))
        logger.dump_tabular()
        if update % 1000 == 0:
            if average_running_reward > best_value:
                best_value = average_running_reward
                model.save('./TaxiFCCACEMean'+norm_type+'.criticTraining'+str(critic_training_type) +'.lr'+str(actor_lr)+'-'+str(critic_lr)+'.satisfied_percentage'+str(satisfied_percentage)+'.max_var'+str(max_var)+'.penalty_weight'+str(penalty_weight)+'/','TaxiFCCAC'+norm_type+'.param', checkPoint=update)
        if update % 100 == 0:
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
    critic = TaxiCentralizedPenaltyandRewardDeepVPN(env.action_space.shapes,
                                                    satisfied_percentage=args['satisfied_percentage'],
                                                    hidden_sizes=hidden_layers, layer_norm=args['layer_norm'],
                                                    batch_norm=args['batch_norm'], penalty_weight = penalty_weight, penalty_zones = penalty_zones, adjacent_list=adjacent_list)
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
