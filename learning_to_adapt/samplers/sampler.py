from learning_to_adapt.samplers.base import BaseSampler
from learning_to_adapt.samplers.vectorized_env_executor import ParallelEnvExecutor, IterativeEnvExecutor
from learning_to_adapt.logger import logger
from learning_to_adapt.utils import utils
from pyprind import ProgBar
import numpy as np
import time
import itertools


class Sampler(BaseSampler):
    """
    Sampler for Meta-RL

    """

    def __init__(
            self,
            env,
            policy,
            num_rollouts,
            max_path_length,
            n_parallel=1,
            adapt_batch_size=None,

    ):
        super(Sampler, self).__init__(env, policy, n_parallel, max_path_length)

        self.total_samples = num_rollouts * max_path_length
        self.n_parallel = n_parallel
        self.total_timesteps_sampled = 0
        self.adapt_batch_size = adapt_batch_size

        # setup vectorized environment

        if self.n_parallel > 1:
            self.vec_env = ParallelEnvExecutor(env, n_parallel, num_rollouts, self.max_path_length)
        else:
            self.vec_env = IterativeEnvExecutor(env, num_rollouts, self.max_path_length)

    def update_tasks(self):
        pass

    def obtain_samples(self, log=False, log_prefix='', random=False):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger
            random (boolean): whether the actions are random

        Returns:
            (list): A list of dicts with the samples
        """

        # initial setup / preparation
        paths = []

        n_samples = 0
        num_envs = self.vec_env.num_envs
        running_paths = [_get_empty_running_paths_dict() for _ in range(num_envs)]

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy
        policy.reset(dones=[True] * self.vec_env.num_envs)

        # initial reset of meta_envs
        obses = np.asarray(self.vec_env.reset())

        while n_samples < self.total_samples:

            # execute policy
            t = time.time()
            if random:
                actions = np.stack([self.env.action_space.sample() for _ in range(num_envs)], axis=0)
                agent_infos = {}
            else:
                a_bs = self.adapt_batch_size
                if a_bs is not None and len(running_paths[0]['observations']) > a_bs + 1:
                    adapt_obs = [np.stack(running_paths[idx]['observations'][-a_bs - 1:-1])
                                 for idx in range(num_envs)]
                    adapt_act = [np.stack(running_paths[idx]['actions'][-a_bs-1:-1])
                                 for idx in range(num_envs)]
                    adapt_next_obs = [np.stack(running_paths[idx]['observations'][-a_bs:])
                                      for idx in range(num_envs)]
                    policy.dynamics_model.switch_to_pre_adapt()
                    policy.dynamics_model.adapt(adapt_obs, adapt_act, adapt_next_obs)
                actions, agent_infos = policy.get_actions(obses)
            policy_time += time.time() - t

            # step environments
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            agent_infos, env_infos = self._handle_info_dicts(agent_infos, env_infos)

            new_samples = 0
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                # append new samples to running paths
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["dones"].append(done)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)

                # if running path is done, add it to paths and empty the running path
                if done:
                    paths.append(dict(
                        observations=np.asarray(running_paths[idx]["observations"]),
                        actions=np.asarray(running_paths[idx]["actions"]),
                        rewards=np.asarray(running_paths[idx]["rewards"]),
                        dones=np.asarray(running_paths[idx]["dones"]),
                        env_infos=utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()

            pbar.update(self.vec_env.num_envs)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        return paths

    def _handle_info_dicts(self, agent_infos, env_infos):
        if not env_infos:
            env_infos = [dict() for _ in range(self.vec_env.num_envs)]
        if not agent_infos:
            agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
        return agent_infos, env_infos


def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], dones=[], env_infos=[], agent_infos=[])
