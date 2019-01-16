import gym


#environment wrapper to fit into the open AI baseline
#use baselines.common.vec_env.subproc_vec_env.SubprocVecEn for parallel sampling
class FakeVecEnvWrapper(gym.Env):
    def __init__(self, env):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.env = env
        self.remotes = [0]

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


    def step(self, actions):
        obs, actions, rews, dones, infos = self.env.step(actions[0])
        return [obs], [actions], [rews], [dones], [infos]

    def reset(self):
        obs, stateCount = self.env.reset()
        return [obs], [stateCount]#,np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        self.env.close()

    @property
    def num_envs(self):
        return 1