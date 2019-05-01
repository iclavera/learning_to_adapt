from learning_to_adapt.utils.serializable import Serializable


class Policy(Serializable):
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        self.env = env
        while hasattr(self.env, 'wrapped_env'):
            self.env = self.env.wrapped_env

    def get_action(self, observation):
        raise NotImplementedError

    def get_actions(self, observations):
        raise NotImplementedError

    def reset(self, dones=None):
        pass

    @property
    def vectorized(self):
        """
        Indicates whether the policy is vectorized. If True, it should implement get_actions(), and support resetting
        with multiple simultaneous states.
        """
        return False

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def recurrent(self):
        """
        Indicates whether the policy is recurrent.
        :return:
        """
        return False

    def log_diagnostics(self, paths, prefix=''):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def state_info_keys(self):
        """
        Return keys for the information related to the policy's state when taking an action.
        :return:
        """
        return [k for k, _ in self.state_info_specs]

    @property
    def state_info_specs(self):
        """
        Return keys and shapes for the information related to the policy's state when taking an action.
        :return:
        """
        return list()

    def terminate(self):
        """
        Clean up operation
        """
        pass
