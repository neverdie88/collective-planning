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

from baselines.utils.utils import discounted_returns
from baselines.utils.utils import Scheduler, make_path, find_trainable_variables
from baselines.models.decmodels import CollectiveDecActorGrid
from baselines.models.centralized_critic_models import GridDeepVPN
from baselines.utils.utils import cat_entropy, mse
from baselines.environments.patrolling import PatrollingGame
from baselines.utils.utils import flatten

TINY = 1e-8


class Model(object):
    def __init__(self, actor, critic, observation_space, action_space, num_procs,
                 ent_coef=0.01, critic_l2_reg=0.01, actor_lr=1e-4, critic_lr=1e-3, population=0, rewardScaler=0.0,
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
        r = tf.placeholder(tf.float32, shape=(None,))

        # next state accumulated return
        next_R = tf.placeholder(tf.float32, shape=(None,))
        vf_mean_tf = tf.placeholder(tf.float32, shape=())
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
        immediate_reward_tf = critic_outputs['immediate_reward']
        next_return_tf = critic_outputs['next_return']
        next_state_count = critic_outputs['next_state_count']

        # critic_tf_with_actor_tf = critic(obs0, actor_tf*state_count, reuse=True)


        # critic_tf_with_actor_tf = critic(obs0, actor_tf * state_count, reuse=True)['symbolic_val']
        entropy = -tf.reduce_mean(tf.reduce_sum(state_count * actor_tf * tf.log(actor_tf + TINY), axis=(1, 2)))
        pg_loss = -tf.reduce_mean(
            critic_tf * tf.reduce_sum(action_count * tf.log(actor_tf + TINY), axis=(1, 2)))/population# - 0.01*entropy/population

        # vf_loss = tf.reduce_mean(mse(tf.reduce_sum(tf.squeeze(critic_tf)* action_count - R* action_count , axis=(1, 2))))
        # vf_loss = tf.reduce_mean(tf.reduce_sum(tf.square(critic_tf - R), axis = 1))

        vf_loss = tf.reduce_mean(tf.square(immediate_reward_tf - r)) \
                  + tf.reduce_mean(tf.square(next_return_tf - next_R))

        # tf.reduce_mean(tf.reduce_sum(tf.square(r_tf - r), axis=1)) +
        vf_immediate_loss = tf.reduce_mean(tf.square(immediate_reward_tf - r))
        vf_next_return_loss = tf.reduce_mean(tf.square(next_return_tf - next_R))

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

        pre_critic_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr * 10)
        pre_critic_update = pre_critic_optimizer.apply_gradients(critic_grads)

        def get_next_return(obs, action_count_samples):
            td_map = {obs0: obs, action_count: action_count_samples}
            next_state_revenue = sess.run(
                next_return_tf,
                td_map
            )
            return next_state_revenue

        def get_immediate_reward(obs, action_count_samples):
            td_map = {obs0: obs, action_count: action_count_samples}
            next_state_revenue = sess.run(
                immediate_reward_tf,
                td_map
            )
            return next_state_revenue

        def get_next_state_count(obs, action_count_samples):
            td_map = {obs0: obs, action_count: action_count_samples}
            next_state_revenue = sess.run(
                next_state_count,
                td_map
            )
            return next_state_revenue

        self.get_next_return = get_next_return
        self.get_immediate_reward = get_immediate_reward
        self.get_next_state_count = get_next_state_count

        def train(obs, state_count_samples, action_count_samples, sample_r, sample_next_R, vf_mean):
            td_map = {obs0: obs, state_count: state_count_samples, action_count: action_count_samples, r: sample_r,
                      next_R: sample_next_R, vf_mean_tf: vf_mean}
            policy_loss, value_loss, _, _, immediate_val_loss, revenue_VF_loss = sess.run(
                [pg_loss, vf_loss, actor_update, critic_update, vf_immediate_loss,
                 vf_next_return_loss],
                td_map
            )
            return policy_loss, value_loss, immediate_val_loss, revenue_VF_loss

        def trainCritic(obs, state_count_samples, action_count_samples, sample_r, sample_next_R):
            td_map = {obs0: obs, state_count: state_count_samples, action_count: action_count_samples, r: sample_r,
                      next_R: sample_next_R}
            value_loss, _, immediate_val_loss, revenue_VF_loss = sess.run(
                [vf_loss, pre_critic_update, vf_immediate_loss, vf_next_return_loss],
                td_map
            )

            return value_loss, immediate_val_loss, revenue_VF_loss

        self.trainCritic = trainCritic

        def step(obs):
            return sess.run(actor_tf, {obs0: obs})

        def value(obs, action_count_input):
            return sess.run(critic_tf, {obs0: obs, action_count: action_count_input})

        def get_next_state_count(obs, action_count_input):
            return sess.run(next_state_count, {obs0: obs, action_count: action_count_input})

        self.get_next_state_count = get_next_state_count

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
        self.batch_inflow_shape = (nenv * nsteps, len(env.action_space.shapes))
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
        self.meanAds = 1

    # def update_obs(self, obs):
    #     self.obs = np.roll(self.obs, shift=-1, axis=3)
    #     self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def run(self):
        mb_obs, mb_actions, mb_rewards, mb_action_probs, mb_dones, mb_values = [], [], [], [], [], []
        mb_counts = []
        for n in range(self.nsteps):
            action_probs = self.model.step(self.obs)
            mb_obs.append(np.copy(self.obs))
            mb_action_probs.append(action_probs)
            mb_dones.append(self.dones)
            obs, action, rewards, dones, info = self.env.step(action_probs)
            # action /=20.0
            count = [info[i]['count'] for i in range(len(info))]
            mb_counts.append(count)
            mb_actions.append(action)
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        last_values = self.model.get_next_return(mb_obs[-1], np.array(mb_actions[-1]))# / self.model.population)

        # batch of steps to batch of rollouts

        # self.model.value(mb_obs, mb_actions[0] / self.model.population)
        mb_action_probs = np.asarray(mb_action_probs, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.float32).swapaxes(1, 0)

        average_reward = (np.sum(mb_rewards) / self.env.num_envs)

        mb_rewards = (np.asarray(mb_rewards, dtype=np.float32).swapaxes(1,
                                                                        0) / self.model.rewardScaler)# / self.model.population
        maxReward = np.max(mb_rewards)
        minReward = np.min(mb_rewards)
        action_count = np.array(mb_actions).reshape(self.batch_ac_shape)
        # mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        action_count[:, :, :] = np.sum(action_count, axis=2)[:, :, np.newaxis]  # HACK
        mb_states = action_count.reshape(self.batch_ac_shape)

        mb_action_probs = mb_action_probs.reshape(self.batch_ac_shape)

        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        self.obs, self.stateCount = self.env.reset()

        mb_next_penalty_returns = []
        mb_next_trip_pay_returns = []
        mb_returns = []
        mb_actions = mb_actions# / self.model.population

        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0)

        mb_returns = []
        mb_next_vals = []

        for n, atomic_rewards in enumerate(mb_rewards):
            rewards = discounted_returns(atomic_rewards, last_values[n], self.gamma)
            mb_returns.append(rewards)
            rewards = np.roll(rewards, -1, axis=0)
            rewards[-1] = last_values[n]
            mb_next_vals.append(rewards)


        mb_obs = mb_obs.reshape(self.batch_ob_shape)
        mb_masks = mb_masks.flatten()

        mb_states = mb_states# / self.model.population
        mb_actions = mb_actions.reshape(
            self.batch_ac_shape)
        mb_rewards = mb_rewards.flatten()
        mb_next_vals = np.array(mb_next_vals).flatten()
        return mb_obs, mb_actions, mb_states, mb_rewards, mb_masks, mb_action_probs, mb_next_vals, average_reward, self.meanAds

from os.path import dirname, abspath
def learn(actor, critic, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), ent_coef=0.01, gamma=0.9,
          lam=0.7, log_interval=1, actor_lr=1e-4, critic_lr=1e-3, critic_l2_reg=0.01, population=0, rewardScaler=0.0,
          layer_norm=False, batch_norm=False, critic_training_type=1, vf_normalization=False, victimNum=1, N=1
          ):
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
                  critic_l2_reg=critic_l2_reg, population=population, rewardScaler=rewardScaler,
                  critic_training_type=critic_training_type)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma, lam=lam, vf_normalization=vf_normalization)

    nbatch = nenvs * nsteps
    tstart = time.time()
    average_running_reward = 0
    average_running_penalty = 0
    average_running_revenue = 0
    # model.load(
    #   './a2cGETaxiStatePenaltyandRewardTaxiCentralizedPenaltyandRewardDeepVPNLayerNorm.satisfied_percentage0.9/a2cGETaxiStatePenaltyandRewardPseudoFlowMinoutACWithCostVPNSequelLayerNorm.param-6000')
    print('Intializing with training critic function')
    for k in range(500):
        print('Pre-train critic Iteration ' + str(k))
        obs, action_count_samples, state_count_samples, sample_r, mb_masks, mb_action_probs, sample_next_R, average_reward, meanAds = runner.run()
        value_loss, immediate_val_loss, revenue_VF_loss = model.trainCritic(obs, state_count_samples,
                                                                                             action_count_samples,
                                                                                             sample_r, sample_next_R)

        print(str(k) + ',' + str(value_loss) + ',' + str(immediate_val_loss) + ',' + str(revenue_VF_loss))
        # model.load('./a2cGETaxiPseudoFlowMinoutACWithCostVPN/a2cGETaxiPseudoFlowMinoutACWithCostVPN.param-51700')
        # 'a2cGETaxiCenNewCriticDecNormalizedObsRewardScaler/a2cGETaxiCenNewCriticDecNormalizedObsRewardScaler.param-9000')
    # model.loadActor('./a2cFictitiousSamplingTaxiDecNormalizedObsRewardScaler/fileName')

    for update in range(1, total_timesteps // nbatch + 1):
        obs, action_count_samples, state_count_samples, sample_r, mb_masks, mb_action_probs, sample_next_R, average_reward, meanAds = runner.run()
        average_running_reward += average_reward
        # (obs, state_count_samples, action_count_samples, returns)
        policy_loss, value_loss, immediate_val_loss, revenue_VF_loss = model.train(obs, state_count_samples, action_count_samples, sample_r, sample_next_R, meanAds)
        # actor_grads = np.array(list(flatten(actor_grads)))
        # max_actor_grads = np.max([np.max(i) for i in actor_grads])
        # min_actor_grads = np.min([np.min(i) for i in actor_grads])
        # print('max grad, min grad:' + str(max_actor_grads) + ',' + str(min_actor_grads))
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0:
            # ev = explained_variance(values, returns)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("immediate_val_loss", float(immediate_val_loss))
            logger.record_tabular("revenue_VF_loss", float(revenue_VF_loss))
            logger.record_tabular("policy_loss", float(policy_loss))
            # logger.record_tabular("max_predict", np.max(abs(values)))            # logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("return", (float(average_running_reward) / log_interval))
            logger.record_tabular("penalty", (float(average_running_penalty) / log_interval))
            logger.record_tabular("revenue", (float(average_running_revenue) / log_interval))
            logger.dump_tabular()
            if update % 10000 == 0:
                # if (average_running_reward >1e6)|(update% 20 == 0):
                best_value = average_running_reward
                model.save('./GridPatrollingVanillaAC' + norm_type + '.criticTraining' + str(
                    critic_training_type) + '.lr' + str(actor_lr) + '-' + str(critic_lr) + '.victimNum' + str(
                    victimNum) + '.N' + str(N) + '/',
                           'VanillaAC' + norm_type + '.param',
                           checkPoint=update)
            average_running_reward = 0
            average_running_penalty = 0
            average_running_revenue = 0
    env.close()


def main():
    seed = 42
    args = parse_args()
    nenvs = args['num_cpu']
    edge1 = args['grid_size']
    edge2 = args['grid_size']
    stateNum = edge1 * edge2
    initialDistribution = np.zeros(stateNum)
    initialDistribution[int(edge1 / 2) * edge2 + int(edge2 / 2)] = 1.0
    victimNum = args['victim_num']
    N = args['population_size']
    env = PatrollingGame(N, initialDistribution, victimNum, edge1=edge1, edge2=edge2)
    populationSize = env.N
    rewardScaler = args['reward_scaler']
    adjacentMatrix = env.neighbors
    adjacent_list = []
    for i in range(stateNum):
        adjacent_list.append(adjacentMatrix[i])
    print('population size' + str(populationSize))

    env.seed(seed)
    incomeFlows = env.incomeFlows
    env = FakeVecEnvWrapper(env)
    vf_normalization = args['vf_normalization']
    set_global_seeds(seed)
    hidden_layers = ()
    hidden_unit_nums = args['hidden_unit_num']
    if hidden_unit_nums > 0:
        hidden_layers = (hidden_unit_nums, hidden_unit_nums)
    critic = GridDeepVPN(env.action_space.shapes, incomeFlows, adjacent_list, layer_norm=args['layer_norm'], batch_norm=args['batch_norm'], stateNum=stateNum, N=N)
    actor = CollectiveDecActorGrid(env.action_space.shapes, layer_norm=args['layer_norm'],
                                   batch_norm=args['batch_norm'],
                                   hidden_sizes=hidden_layers, adjacent_list=adjacent_list, stateNum=stateNum)
    # critic = GridCollectiveCritic(env.action_space.shapes)
    # actor = GridCollectiveActor(env.action_space.shapes)
    learn(actor, critic, env, seed, nsteps=100, ent_coef=args['ent_coef'], actor_lr=args['actor_lr'],
          critic_lr=args['critic_lr'], critic_l2_reg=args['critic_l2_reg'], lam=args['lambda'],
          population=populationSize, rewardScaler=rewardScaler, layer_norm=args['layer_norm'],
          batch_norm=args['batch_norm'],
          critic_training_type=args['critic_training_type'], vf_normalization=vf_normalization, victimNum=victimNum,
          N=N)


if __name__ == '__main__':
    main()
