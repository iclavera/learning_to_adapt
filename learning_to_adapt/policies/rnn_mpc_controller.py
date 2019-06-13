from learning_to_adapt.policies.base import Policy
from learning_to_adapt.utils.serializable import Serializable
import numpy as np
import tensorflow as tf


class RNNMPCController(Policy, Serializable):
    def __init__(
            self,
            name,
            env,
            dynamics_model,
            reward_model=None,
            discount=1,
            use_cem=False,
            n_candidates=1024,
            horizon=10,
            num_cem_iters=8,
            percent_elites=0.05,
            use_reward_model=False
    ):
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.discount = discount
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.use_cem = use_cem
        self.num_cem_iters = num_cem_iters
        self.percent_elites = percent_elites
        self.env = env
        self.use_reward_model = use_reward_model
        self._hidden_state = None

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, 'wrapped_env'):
            self.unwrapped_env = self.unwrapped_env.wrapped_env

        # make sure that env has reward function
        if not self.use_reward_model:
            assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        Serializable.quick_init(self, locals())
        super(RNNMPCController, self).__init__(env=env)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        if observation.ndim == 1:
            observation = observation[None]

        action = self.get_actions(observation)[0]

        return action, dict()

    def get_actions(self, observations):
        if self.use_cem:
            actions = self.get_cem_action(observations)
        else:
            actions = self.get_rs_action(observations)

        _, self._hidden_state = self.dynamics_model.predict(np.array(observations), actions, self._hidden_state)

        return actions, dict()

    def get_random_action(self, n):
        return np.random.uniform(low=self.action_space.low,
                                 high=self.action_space.high, size=(n,) + self.action_space.low.shape)

    def get_cem_action(self, observations):

        n = self.n_candidates
        m = len(observations)
        h = self.horizon
        act_dim = self.action_space.shape[0]

        num_elites = max(int(self.n_candidates * self.percent_elites), 1)
        mean = np.zeros((m, h * act_dim))
        std = np.ones((m, h * act_dim))
        clip_low = np.concatenate([self.action_space.low] * h)
        clip_high = np.concatenate([self.action_space.high] * h)

        for i in range(self.num_cem_iters):
            z = np.random.normal(size=(n, m,  h * act_dim))
            a = mean + z * std
            a_stacked = np.clip(a, clip_low, clip_high)
            a = a.reshape((n * m, h, act_dim))
            a = np.transpose(a, (1, 0, 2))
            returns = np.zeros((n * m,))

            cand_a = a[0].reshape((m, n, -1))
            observation = np.repeat(observations, n, axis=0)
            hidden_state = self.repeat_hidden(self._hidden_state, n)
            for t in range(h):
                next_observation, hidden_state = self.dynamics_model.predict(observation, a[t], hidden_state)
                if self.use_reward_model:
                    assert self.reward_model is not None
                    rewards = self.reward_model.predict(observation, a[t], next_observation)
                else:
                    rewards = self.unwrapped_env.reward(observation, a[t], next_observation)
                returns += self.discount ** t * rewards
                observation = next_observation
            returns = returns.reshape(m, n)
            elites_idx = ((-returns).argsort(axis=-1) < num_elites).T
            elites = a_stacked[elites_idx]
            mean = np.mean(elites, axis=0)
            std = np.std(elites, axis=0)

        return cand_a[range(m), np.argmax(returns, axis=1)]

    def get_rs_action(self, observations):
        n = self.n_candidates
        m = len(observations)
        h = self.horizon
        returns = np.zeros((n * m,))

        a = self.get_random_action(h * n * m).reshape((h, n * m, -1))

        cand_a = a[0].reshape((m, n, -1))
        observation = np.repeat(observations, n, axis=0)
        hidden_state = self.repeat_hidden(self._hidden_state, n)

        for t in range(h):
            next_observation, hidden_state = self.dynamics_model.predict(observation, a[t], hidden_state)
            if self.use_reward_model:
                assert self.reward_model is not None
                rewards = self.reward_model.predict(observation, a[t], next_observation)
            else:
                rewards = self.unwrapped_env.reward(observation, a[t], next_observation)
            returns += self.discount ** t * rewards
            observation = next_observation
        returns = returns.reshape(m, n)
        return cand_a[range(m), np.argmax(returns, axis=1)]

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple

        if dones is None:
            dones = [True]
        if self._hidden_state is None:
            self._hidden_state = self.dynamics_model.get_initial_hidden(batch_size=len(dones))

        zero_hidden_state = self.dynamics_model.get_initial_hidden(batch_size=1)

        if type(zero_hidden_state[0]) is list:
            for i, z_h in enumerate(zero_hidden_state):
                for idx in range(len(zero_hidden_state[0])):
                    if isinstance(z_h[idx], LSTMStateTuple):
                        self._hidden_state[i][idx].c[dones] = z_h[idx].c
                        self._hidden_state[i][idx].h[dones] = z_h[idx].h
                    else:
                        self._hidden_state[i][idx][dones] = z_h[idx]
        else:
            for i, z_h in enumerate(zero_hidden_state):
                if isinstance(z_h, LSTMStateTuple):
                    self._hidden_state[i].c[dones] = z_h.c
                    self._hidden_state[i].h[dones] = z_h.h
                else:
                    self._hidden_state[i][dones] = z_h

    def repeat_hidden(self, hidden, n):
        LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
        if not isinstance(hidden, list) and not isinstance(hidden, tuple):
            if isinstance(hidden, LSTMStateTuple):
                hidden_c = hidden.c
                hidden_h = hidden.h
                return LSTMStateTuple(np.repeat(hidden_c, n, axis=0), np.repeat(hidden_h, n, axis=0))

            else:
                return np.repeat(hidden, n, axis=0)
        else:
            _hidden = []
            for h in hidden:
                _h = self.repeat_hidden(h, n)
                _hidden.append(_h)
                # if isinstance(h, LSTMStateTuple):
                #     hidden_c = h.c
                #     hidden_h = h.h
                #     _hidden.append(LSTMStateTuple(np.repeat(hidden_c, n, axis=0), np.repeat(hidden_h, n, axis=0)))
                #
                # else:
                #     _hidden.append(np.repeat(h, n, axis=0))
            return _hidden

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
