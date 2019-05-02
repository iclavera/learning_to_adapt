import numpy as np
import pickle as pickle
from multiprocessing import Process, Pipe
import copy


class IterativeEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Internally, the environments are executed iteratively.

    """

    def __init__(self, env, num_rollouts, max_path_length):
        self._num_envs = num_rollouts
        self.envs = np.asarray([copy.deepcopy(env) for _ in range(self._num_envs)])
        self.ts = np.zeros(len(self.envs), dtype='int')  # time steps
        self.max_path_length = max_path_length

    def step(self, actions):
        """
        Steps the wrapped environments with the provided actions

        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool),
             env_infos (dict). Each list is of length meta_batch_size x envs_per_task
             (assumes that every task has same number of meta_envs)
        """
        assert len(actions) == self.num_envs

        all_results = [env.step(a) for (a, env) in zip(actions, self.envs)]

        # stack results split to obs, rewards, ...
        obs, rewards, dones, env_infos = list(map(list, zip(*all_results)))

        # reset env when done or max_path_length reached
        dones = np.asarray(dones)
        self.ts += 1
        dones = np.logical_or(self.ts >= self.max_path_length, dones)

        for i in np.argwhere(dones).flatten():
            obs[i] = self.envs[i].reset()
            self.ts[i] = 0

        return obs, rewards, dones, env_infos

    def reset(self):
        """
        Resets the environments

        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        obses = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return obses

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return self._num_envs


class ParallelEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Thereby the environments are distributed among meta_batch_size processes and
    executed in parallel.

    """

    def __init__(self, env, n_parallel, num_rollouts, max_path_length):
        assert num_rollouts % n_parallel == 0
        self.envs_per_proc = int(num_rollouts/n_parallel)
        self._num_envs = n_parallel * self.envs_per_proc
        self.n_parallel = n_parallel
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_parallel)])
        seeds = np.random.choice(range(10**6), size=n_parallel, replace=False)

        self.ps = [
            Process(target=worker, args=(work_remote, remote, pickle.dumps(env), self.envs_per_proc, max_path_length, seed))
            for (work_remote, remote, seed) in zip(self.work_remotes, self.remotes, seeds)]  # Why pass work remotes?

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        """
        Executes actions on each env

        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool), env_infos (dict)
                      each list is of length meta_batch_size x envs_per_task (assumes that every task has same number of meta_envs)
        """
        assert len(actions) == self.num_envs

        # split list of actions in list of list of actions per meta tasks
        chunks = lambda l, n: [l[x: x + n] for x in range(0, len(l), n)]
        actions_per_meta_task = chunks(actions, self.envs_per_proc)

        # step remote environments
        for remote, action_list in zip(self.remotes, actions_per_meta_task):
            remote.send(('step', action_list))

        results = [remote.recv() for remote in self.remotes]

        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*results))

        return obs, rewards, dones, env_infos

    def reset(self):
        """
        Resets the environments of each worker

        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        return sum([remote.recv() for remote in self.remotes], [])

    def set_tasks(self, tasks=None):
        """
        Sets a list of tasks to each worker

        Args:
            tasks (list): list of the tasks for each worker
        """
        for remote, task in zip(self.remotes, tasks):
            remote.send(('set_task', task))
        for remote in self.remotes:
            remote.recv()

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return self._num_envs


def worker(remote, parent_remote, env_pickle, n_envs, max_path_length, seed):
    """
    Instantiation of a parallel worker for collecting samples. It loops continually checking the task that the remote
    sends to it.

    Args:
        remote (multiprocessing.Connection):
        parent_remote (multiprocessing.Connection):
        env_pickle (pkl): pickled environment
        n_envs (int): number of environments per worker
        max_path_length (int): maximum path length of the task
        seed (int): random seed for the worker
    """
    parent_remote.close()

    envs = [pickle.loads(env_pickle) for _ in range(n_envs)]
    np.random.seed(seed)

    ts = np.zeros(n_envs, dtype='int')

    while True:
        # receive command and data from the remote
        cmd, data = remote.recv()

        # do a step in each of the environment of the worker
        if cmd == 'step':
            all_results = [env.step(a) for (a, env) in zip(data, envs)]
            obs, rewards, dones, infos = map(list, zip(*all_results))
            ts += 1
            for i in range(n_envs):
                if dones[i] or (ts[i] >= max_path_length):
                    dones[i] = True
                    obs[i] = envs[i].reset()
                    ts[i] = 0
            remote.send((obs, rewards, dones, infos))

        # reset all the environments of the worker
        elif cmd == 'reset':
            obs = [env.reset() for env in envs]
            ts[:] = 0
            remote.send(obs)

        # set the specified task for each of the environments of the worker
        elif cmd == 'set_task':
            for env in envs:
                env.set_task(data)
            remote.send(None)

        # close the remote and stop the worker
        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError
