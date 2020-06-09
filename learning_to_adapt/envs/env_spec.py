from learning_to_adapt.utils.serializable import Serializable
from learning_to_adapt.spaces.base import Space


class EnvSpec(Serializable):

    def __init__(
            self,
            observation_space,
            action_space):
        """
        :type observation_space: Space
        :type action_space: Space
        """
        Serializable.quick_init(self, locals())
        self._observation_space = observation_space
        self._action_space = action_space
        self.max_episode_steps = None

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
