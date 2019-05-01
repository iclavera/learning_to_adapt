import numpy as np
import os.path as osp
from learning_to_adapt import spaces
from learning_to_adapt.envs.base import Env
from learning_to_adapt.mujoco_py import MjModel, MjViewer

MODEL_DIR = osp.abspath(
    osp.join(
        osp.dirname(__file__),
        '../../../vendor/mujoco_models'
    )
)

BIG = 1e6


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b): # multiply two quaternion
    w = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
    i = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
    j = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
    k = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    return [w, i, j, k]


class MujocoEnv(Env):
    FILE = None

    def __init__(self, file_path=None, action_noise=0.0, random_init_state=True):
        # compile template
        assert file_path is not None
        self.model = MjModel(file_path)
        self.data = self.model.data
        self.viewer = None
        self.init_qpos = self.model.data.qpos
        self.init_qvel = self.model.data.qvel
        self.init_qacc = self.model.data.qacc
        self.init_ctrl = self.model.data.ctrl
        self.qpos_dim = self.init_qpos.size
        self.qvel_dim = self.init_qvel.size
        self.ctrl_dim = self.init_ctrl.size
        self.action_noise = action_noise
        self.random_init_state = random_init_state
        if "frame_skip" in self.model.numeric_names:
            frame_skip_id = self.model.numeric_names.index("frame_skip")
            addr = self.model.numeric_adr.flat[frame_skip_id]
            self.frame_skip = int(self.model.numeric_data.flat[addr])
        else:
            self.frame_skip = 1
        if "init_qpos" in self.model.numeric_names:
            init_qpos_id = self.model.numeric_names.index("init_qpos")
            addr = self.model.numeric_adr.flat[init_qpos_id]
            size = self.model.numeric_size.flat[init_qpos_id]
            init_qpos = self.model.numeric_data.flat[addr:addr + size]
            self.init_qpos = init_qpos
        self.dcom = None
        self.current_com = None
        self.reset()
        super(MujocoEnv, self).__init__()

    @property
    def action_space(self):
        bounds = self.model.actuator_ctrlrange
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return spaces.Box(lb, ub)

    @property
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @property
    def action_bounds(self):
        return self.action_space.bounds

    def reset_mujoco(self, init_state=None):
        if init_state is None:
            if self.random_init_state:
                self.model.data.qpos = self.init_qpos + \
                    np.random.normal(size=self.init_qpos.shape) * 0.01
                self.model.data.qvel = self.init_qvel + \
                    np.random.normal(size=self.init_qvel.shape) * 0.1
            else:
                self.model.data.qpos = self.init_qpos
                self.model.data.qvel = self.init_qvel

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

    def reset(self, init_state=None):
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    def get_current_obs(self):
        return self._get_full_obs()

    def _get_full_obs(self):
        data = self.model.data
        cdists = np.copy(self.model.geom_margin).flat
        for c in self.model.data.contact:
            cdists[c.geom2] = min(cdists[c.geom2], c.dist)
        obs = np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            # data.cdof.flat,
            data.cinert.flat,
            data.cvel.flat,
            # data.cacc.flat,
            data.qfrc_actuator.flat,
            data.cfrc_ext.flat,
            data.qfrc_constraint.flat,
            cdists,
            # data.qfrc_bias.flat,
            # data.qfrc_passive.flat,
            self.dcom.flat,
        ])
        return obs

    @property
    def _state(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])

    @property
    def _full_state(self):
        return np.concatenate([
            self.model.data.qpos,
            self.model.data.qvel,
            self.model.data.qacc,
            self.model.data.ctrl,
        ]).ravel()

    def inject_action_noise(self, action):
        # generate action noise
        noise = self.action_noise * \
                np.random.normal(size=action.shape)
        # rescale the noise to make it proportional to the action bounds
        lb, ub = self.action_bounds
        noise = 0.5 * (ub - lb) * noise
        return action + noise

    def forward_dynamics(self, action):
        self.model.data.ctrl = self.inject_action_noise(action)
        for _ in range(self.frame_skip):
            self.model.step()
        self.model.forward()
        new_com = self.model.data.com_subtree[0]
        self.dcom = new_com - self.current_com
        self.current_com = new_com

    def get_viewer(self, config=None):
        if self.viewer is None:
            self.viewer = MjViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
        if config is not None:
            self.viewer.set_window_pose(config["xpos"], config["ypos"])
            self.viewer.set_window_size(config["width"], config["height"])
            self.viewer.set_window_title(config["title"])
        return self.viewer

    def render(self, close=False, mode='human', config=None):
        if mode == 'human':
            viewer = self.get_viewer(config=config)
            viewer.loop_once()
        elif mode == 'rgb_array':
            viewer = self.get_viewer(config=config)
            viewer.loop_once()
            # self.get_viewer(config=config).render()
            data, width, height = self.get_viewer(config=config).get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
        if close:
            self.stop_viewer()

    def start_viewer(self):
        viewer = self.get_viewer()
        if not viewer.running:
            viewer.start()

    def stop_viewer(self):
        if self.viewer:
            self.viewer.finish()
            self.viewer = None

    def release(self):
        # temporarily alleviate the issue (but still some leak)
        from learning_to_adapt.mujoco_py.mjlib import mjlib
        mjlib.mj_deleteModel(self.model._wrapped)
        mjlib.mj_deleteData(self.data._wrapped)

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.body_comvels[idx]

    def print_stats(self):
        super(MujocoEnv, self).print_stats()
        print("qpos dim:\t%d" % len(self.model.data.qpos))

    def action_from_key(self, key):
        raise NotImplementedError

    def set_state_tmp(self, state, restore=True):
        if restore:
            prev_pos = self.model.data.qpos
            prev_qvel = self.model.data.qvel
            prev_ctrl = self.model.data.ctrl
            prev_act = self.model.data.act
        qpos, qvel = self.decode_state(state)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model.forward()
        yield
        if restore:
            self.model.data.qpos = prev_pos
            self.model.data.qvel = prev_qvel
            self.model.data.ctrl = prev_ctrl
            self.model.data.act = prev_act
            self.model.forward()

    def get_param_values(self):
        return {}

    def set_param_values(self, values):
        pass
