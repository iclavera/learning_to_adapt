import numpy as np
from learning_to_adapt.utils.serializable import Serializable
from learning_to_adapt.envs.mujoco_env import MujocoEnv
from learning_to_adapt.logger import logger
import os


class AntEnv(MujocoEnv,  Serializable):

    def __init__(self, task='cripple', reset_every_episode=False):
        Serializable.quick_init(self, locals())
        self.cripple_mask = None
        self.reset_every_episode = reset_every_episode
        self.first = True
        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets", "ant.xml"))
        task = None if task == 'None' else task

        self.cripple_mask = np.ones(self.action_space.shape)

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        self.dt = self.model.opt.timestep

        assert task in [None, 'cripple']

        self.task = task
        self.crippled_leg = 0

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def step(self, action):
        if self.cripple_mask is not None:
            action = self.cripple_mask * action
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        # lb, ub = self.action_space.low, self.action_space.high
        # scaling = (ub - lb) * 0.5
        ctrl_cost = 0 #0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        done = False
        ob = self.get_current_obs()
        info = {}
        return ob, reward, done, info

    def reward(self, obs, action, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == action.shape[0]
        # lb, ub = self.action_bounds
        # scaling = (ub - lb) * 0.5
        ctrl_cost = 0 # 5e-3 * np.sum(np.square(action / scaling), axis=1)
        vel = (next_obs[:, -3] - obs[:, -3]) / self.dt
        survive_reward = 0.05
        reward = vel - ctrl_cost + survive_reward
        return reward

    def reset_mujoco(self, init_state=None):
        super(AntEnv, self).reset_mujoco(init_state=init_state)
        if self.reset_every_episode and not self.first:
            self.reset_task()
        if self.first:
            self.first = False

    '''
    our "front" is in +x direction, to the right side of screen

    LEG 4 (they call this back R)
    action0: front-right leg, top joint 
    action1: front-right leg, bottom joint
    
    LEG 1 (they call this front L)
    action2: front-left leg, top joint
    action3: front-left leg, bottom joint 
    
    LEG 2 (they call this front R)
    action4: back-left leg, top joint
    action5: back-left leg, bottom joint 
    
    LEG 3 (they call this back L)
    action6: back-right leg, top joint
    action7: back-right leg, bottom joint 

    geom_names has 
            ['floor','torso_geom',
            'aux_1_geom','left_leg_geom','left_ankle_geom', --1
            'aux_2_geom','right_leg_geom','right_ankle_geom', --2
            'aux_3_geom','back_leg_geom','third_ankle_geom', --3
            'aux_4_geom','rightback_leg_geom','fourth_ankle_geom'] --4
    '''

    def reset_task(self, value=None):

        if self.task == 'cripple':
            # Pick which leg to remove (0 1 2 are train... 3 is test)
            self.crippled_leg = value if value is not None else np.random.randint(0, 3)

            # Pick which actuators to disable
            self.cripple_mask = np.ones(self.action_space.shape)
            if self.crippled_leg == 0:
                self.cripple_mask[2] = 0
                self.cripple_mask[3] = 0
            elif self.crippled_leg == 1:
                self.cripple_mask[4] = 0
                self.cripple_mask[5] = 0
            elif self.crippled_leg == 2:
                self.cripple_mask[6] = 0
                self.cripple_mask[7] = 0
            elif self.crippled_leg == 3:
                self.cripple_mask[0] = 0
                self.cripple_mask[1] = 0

            # Make the removed leg look red
            geom_rgba = self._init_geom_rgba.copy()
            if self.crippled_leg == 0:
                geom_rgba[3, :3] = np.array([1, 0, 0])
                geom_rgba[4, :3] = np.array([1, 0, 0])
            elif self.crippled_leg == 1:
                geom_rgba[6, :3] = np.array([1, 0, 0])
                geom_rgba[7, :3] = np.array([1, 0, 0])
            elif self.crippled_leg == 2:
                geom_rgba[9, :3] = np.array([1, 0, 0])
                geom_rgba[10, :3] = np.array([1, 0, 0])
            elif self.crippled_leg == 3:
                geom_rgba[12, :3] = np.array([1, 0, 0])
                geom_rgba[13, :3] = np.array([1, 0, 0])
            self.model.geom_rgba = geom_rgba

            # Make the removed leg not affect anything
            temp_size = self._init_geom_size.copy()
            temp_pos = self._init_geom_pos.copy()

            if self.crippled_leg == 0:
                # Top half
                temp_size[3, 0] = temp_size[3, 0]/2
                temp_size[3, 1] = temp_size[3, 1]/2
                # Bottom half
                temp_size[4, 0] = temp_size[4, 0]/2
                temp_size[4, 1] = temp_size[4, 1]/2
                temp_pos[4, :] = temp_pos[3, :]

            elif self.crippled_leg == 1:
                # Top half
                temp_size[6, 0] = temp_size[6, 0]/2
                temp_size[6, 1] = temp_size[6, 1]/2
                # Bottom half
                temp_size[7, 0] = temp_size[7, 0]/2
                temp_size[7, 1] = temp_size[7, 1]/2
                temp_pos[7, :] = temp_pos[6, :]

            elif self.crippled_leg == 2:
                # Top half
                temp_size[9, 0] = temp_size[9, 0]/2
                temp_size[9, 1] = temp_size[9, 1]/2
                # Bottom half
                temp_size[10, 0] = temp_size[10, 0]/2
                temp_size[10, 1] = temp_size[10, 1]/2
                temp_pos[10, :] = temp_pos[9, :]

            elif self.crippled_leg == 3:
                # Top half
                temp_size[12, 0] = temp_size[12, 0]/2
                temp_size[12, 1] = temp_size[12, 1]/2
                # Bottom half
                temp_size[13, 0] = temp_size[13, 0]/2
                temp_size[13, 1] = temp_size[13, 1]/2
                temp_pos[13, :] = temp_pos[12, :]

            self.model.geom_size = temp_size
            self.model.geom_pos = temp_pos

        elif self.task is None:
            pass

        else:
            raise NotImplementedError

        self.model.forward()

    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.logkv(prefix + 'AverageForwardProgress', np.mean(progs))
        logger.logkv(prefix + 'MaxForwardProgress', np.max(progs))
        logger.logkv(prefix + 'MinForwardProgress', np.min(progs))
        logger.logkv(prefix + 'StdForwardProgress', np.std(progs))


if __name__ == '__main__':
    env = AntEnv(task='cripple')
    while True:
        env.reset()
        env.reset_task()
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()


