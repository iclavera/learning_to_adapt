import numpy as np
from learning_to_adapt.utils.serializable import Serializable
from learning_to_adapt.envs.mujoco_env import MujocoEnv
from learning_to_adapt.logger import logger
import os


class Arm7DofEnv(MujocoEnv, Serializable):

    def __init__(self, task='force', reset_every_episode=False, fixed_goal=False):
        Serializable.quick_init(self, locals())

        self.reset_every_episode = reset_every_episode
        self.first = True
        self.fixed_goal = fixed_goal
        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                              "assets", "arm_7dof.xml"))
        task = None if task == 'None' else task

        self.cripple_mask = np.ones(self.action_space.shape)

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_body_pos = self.model.body_pos.copy()
        self._init_body_masses = self.model.body_mass.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        self.dt = self.model.opt.timestep

        assert task in [None, 'cripple', 'damping', 'mass', 'force']

        self.task = task

    def step(self, action):
        action = self.cripple_mask * action
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        vec = self.get_body_com("object")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + 0.01 * 0.5 * reward_ctrl
        done = False
        info = {}
        return next_obs, reward, done, info

    def reset_mujoco(self, init_state=None):
        if init_state is None:
            low = np.array([-0.1, -.2, .5])
            high = np.array([0.4, .2, -.5])
            qpos = np.ones(self.init_qpos.shape) * 0.5

            # set random goal position
            self.fixed_goal = True
            if  self.fixed_goal:
                self.goal = np.array([0.3, 0.15, 0])
            else:
                self.goal = np.random.uniform(size=3) * (high - low) + low

            qpos[-3:, 0] = self.goal
            qvel = self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.init_qvel.shape)
            qvel[-3:, 0] = 0

            # reset task, if supposed to
            self.model.data.qpos = qpos
            self.model.data.qvel = qvel
            self.model.data.qacc = self.init_qacc
            self.model.data.ctrl = self.init_ctrl

        else:
            start = 0
            for datum_name in ["qpos", "qvel", "qacc", "ctrl"]:
                datum = getattr(self.model.data, datum_name)
                datum_dim = datum.shape[0]
                datum = init_state[start: start + datum_dim]
                setattr(self.model.data, datum_name, datum)
                start += datum_dim

        if self.reset_every_episode and not self.first:
            self.reset_task()
        else:
            self.first = False

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            (self.get_body_com('object') - self.get_body_com("target"))
        ])

    def reward(self, obs, action, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == action.shape[0]
        vec = next_obs[:, -3:]
        reward_dist = -np.linalg.norm(vec, axis=1)
        reward_ctrl = -np.sum(np.square(action), axis=1)
        reward = reward_dist + 0.01 * 0.5 * reward_ctrl
        return reward

    def reset_task(self, value=None):

        if self.task == 'cripple':
            crippled_joint = value if value is not None else np.random.randint(0, 7)
            self.cripple_mask = np.ones(self.action_space.shape)
            self.cripple_mask[crippled_joint] = 0
            geom_rgba = self._init_geom_rgba.copy()
            geom_idx = crippled_joint+5
            geom_rgba[geom_idx, :3] = np.array([1, 0, 0])
            self.model.geom_rgba = geom_rgba

        elif self.task == 'damping':
            damping = np.random.uniform(0, 2, self.model.dof_damping.shape)
            damping[-2:, :] = 0
            self.model.dof_damping = damping

        elif self.task == 'mass':
            mass_multiplier = value if value is not None else np.random.randint(1, 4)
            masses = self._init_body_masses
            object_mass = masses[-2]
            masses[-2] = object_mass*mass_multiplier
            self.model.body_mass = masses

        elif self.task == 'force':
            g = value if value is not None else np.random.uniform(.1, 2)
            masses = self._init_body_masses
            xfrc = np.zeros_like(self.model.data.xfrc_applied)
            object_mass = masses[-2]
            xfrc[-2, 2] -= object_mass * g
            self.model.data.xfrc_applied = xfrc

        elif self.task is None:
            pass

        else:
            raise NotImplementedError

        self.model.forward()


if __name__ == '__main__':
    env = Arm7DofEnv(task='force')
    while True:
        env.reset()
        env.reset_task()
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()


