from learning_to_adapt.utils import utils
from learning_to_adapt.logger import logger
import time
import numpy as np
import copy
from pyprind import ProgBar


class BaseSampler(object):
    """
    Sampler interface
    """

    def __init__(self, env, policy, num_rollouts, max_path_length):
        assert hasattr(env, 'reset') and hasattr(env, 'step')

        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length

        self.total_samples = num_rollouts * max_path_length
        self.total_timesteps_sampled = 0

    def obtain_samples(self, log=False, log_prefix='', random=False):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger
            random (boolean): whether the actions are random

        Returns:
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        paths = []

        n_samples = 0
        running_paths = _get_empty_running_paths_dict()

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy
        policy.reset(dones=[True])

        # initial reset of meta_envs
        obs = np.asarray(self.env.reset())

        ts = 0

        while n_samples < self.total_samples:

            # execute policy
            t = time.time()
            if random:
                action = self.env.action_space.sample()
                agent_info = {}
            else:
                action, agent_info = policy.get_action(obs)
                if action.ndim == 2:
                    action = action[0]
            policy_time += time.time() - t

            # step environments
            t = time.time()
            next_obs, reward, done, env_info = self.env.step(action)

            ts += 1
            done = done or ts >= self.max_path_length
            if done:
                next_obs = self.env.reset()
                ts = 0
                
            env_time += time.time() - t

            new_samples = 0

            # append new samples to running paths
            if isinstance(reward, np.ndarray):
                reward = reward[0]
            running_paths["observations"].append(obs)
            running_paths["actions"].append(action)
            running_paths["rewards"].append(reward)
            running_paths["dones"].append(done)
            running_paths["env_infos"].append(env_info)
            running_paths["agent_infos"].append(agent_info)

            # if running path is done, add it to paths and empty the running path
            if done:
                paths.append(dict(
                    observations=np.asarray(running_paths["observations"]),
                    actions=np.asarray(running_paths["actions"]),
                    rewards=np.asarray(running_paths["rewards"]),
                    dones=np.asarray(running_paths["dones"]),
                    env_infos=utils.stack_tensor_dict_list(running_paths["env_infos"]),
                    agent_infos=utils.stack_tensor_dict_list(running_paths["agent_infos"]),
                ))
                new_samples += len(running_paths["rewards"])
                running_paths = _get_empty_running_paths_dict()

            pbar.update(new_samples)
            n_samples += new_samples
            obs = next_obs
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        return paths


def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], dones=[], env_infos=[], agent_infos=[])


class SampleProcessor(object):
    """
    Sample processor interface
        - fits a reward baseline (use zero baseline to skip this step)
        - performs Generalized Advantage Estimation to provide advantages (see Schulman et al. 2015 - https://arxiv.org/abs/1506.02438)

    Args:
        baseline (Baseline) : a reward baseline object
        discount (float) : reward discount factor
        gae_lambda (float) : Generalized Advantage Estimation lambda
        normalize_adv (bool) : indicates whether to normalize the estimated advantages (zero mean and unit std)
        positive_adv (bool) : indicates whether to shift the (normalized) advantages so that they are all positive
    """

    def __init__(
            self,
            baseline,
            discount=0.99,
            gae_lambda=1,
            normalize_adv=False,
            positive_adv=False,
            ):

        assert 0 <= discount <= 1.0, 'discount factor must be in [0,1]'
        assert 0 <= gae_lambda <= 1.0, 'gae_lambda must be in [0,1]'
        assert hasattr(baseline, 'fit') and hasattr(baseline, 'predict')
        
        self.baseline = baseline
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.positive_adv = positive_adv

    def process_samples(self, paths, log=False, log_prefix=''):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths (list): A list of paths of size (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (dict) : Processed sample data of size [7] x (batch_size x max_path_length)
        """
        assert type(paths) == list, 'paths must be a list'
        assert paths[0].keys() >= {'observations', 'actions', 'rewards'}
        assert self.baseline, 'baseline must be specified - use self.build_sample_processor(baseline_obj)'

        # fits baseline, compute advantages and stack path data
        samples_data, paths = self._compute_samples_data(paths)

        # 7) log statistics if desired
        self._log_path_stats(paths, log=log, log_prefix=log_prefix)

        assert samples_data.keys() >= {'observations', 'actions', 'rewards', 'advantages', 'returns'}
        return samples_data

    """ helper functions """

    def _compute_samples_data(self, paths):
        assert type(paths) == list

        # 1) compute discounted rewards (returns)
        for idx, path in enumerate(paths):
            path["returns"] = utils.discount_cumsum(path["rewards"], self.discount)

        # 2) fit baseline estimator using the path returns and predict the return baselines
        self.baseline.fit(paths, target_key="returns")
        all_path_baselines = [self.baseline.predict(path) for path in paths]

        # 3) compute advantages and adjusted rewards
        paths = self._compute_advantages(paths, all_path_baselines)

        # 4) stack path data
        observations, actions, rewards, dones, returns, advantages, env_infos, agent_infos = self._concatenate_path_data(copy.deepcopy(paths))

        # 5) if desired normalize / shift advantages
        if self.normalize_adv:
            advantages = utils.normalize_advantages(advantages)
        if self.positive_adv:
            advantages = utils.shift_advantages_to_positive(advantages)

        # 6) create samples_data object
        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
        )

        return samples_data, paths

    def _log_path_stats(self, paths, log=False, log_prefix=''):
        # compute log stats
        average_discounted_return = np.mean([path["returns"][0] for path in paths])
        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        if log == 'reward':
            logger.logkv(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))

        elif log == 'all' or log is True:
            logger.logkv(log_prefix + 'AverageDiscountedReturn', average_discounted_return)
            logger.logkv(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))
            logger.logkv(log_prefix + 'NumTrajs', len(paths))
            logger.logkv(log_prefix + 'StdReturn', np.std(undiscounted_returns))
            logger.logkv(log_prefix + 'MaxReturn', np.max(undiscounted_returns))
            logger.logkv(log_prefix + 'MinReturn', np.min(undiscounted_returns))

    def _compute_advantages(self, paths, all_path_baselines):
        assert len(paths) == len(all_path_baselines)

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = utils.discount_cumsum(
                deltas, self.discount * self.gae_lambda)

        return paths

    def _concatenate_path_data(self, paths):
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        dones = np.concatenate([path["dones"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        env_infos = utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos = utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])
        return observations, actions, rewards, dones, returns, advantages, env_infos, agent_infos

    def _stack_path_data(self, paths):
        max_path = max([len(path['observations']) for path in paths])

        observations = self._stack_padding(paths, 'observations', max_path)
        actions = self._stack_padding(paths, 'actions', max_path)
        rewards = self._stack_padding(paths, 'rewards', max_path)
        dones = self._stack_padding(paths, 'dones', max_path)
        returns = self._stack_padding(paths, 'returns', max_path)
        advantages = self._stack_padding(paths, 'advantages', max_path)
        env_infos = utils.stack_tensor_dict_list([path["env_infos"] for path in paths], max_path)
        agent_infos = utils.stack_tensor_dict_list([path["agent_infos"] for path in paths], max_path)

        return observations, actions, rewards, dones, returns, advantages, env_infos, agent_infos


    def _stack_padding(self, paths, key, max_path):
        padded_array = np.stack([
            np.concatenate([path[key], np.zeros((max_path - path[key].shape[0],) + path[key].shape[1:])])
            for path in paths
        ])
        return padded_array

