import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box


class HalfCheetahEnv_lya(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(self, '/home/sarvan/Classes/Lyapunov-Actor-Critic/env/mujoco_half_cheetah/half_cheetah.xml', 5, observation_space=observation_space, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        self.step_count = 0
        self.max_step_per_episode = 200
        self.truncated = False


    def step(self, action):
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        # reward_run = xposafter
        reward = reward_ctrl + reward_run
        done = False

        if self.render_mode == "human":
            self.render()

        l_rewards = max(abs(reward_run) - 0.9*3., 0.) ** 2

        if abs(reward_run)>3:
            violation_of_constraint = 1
        else:
            violation_of_constraint = 0

        self.step_count += 1
        if self.step_count >= self.max_step_per_episode:
            self.truncated = True

        # print(xposafter)
        return ob, l_rewards, self.truncated, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,l_rewards=l_rewards,violation_of_constraint=violation_of_constraint)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.random(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.step_count = 0
        self.truncated = False
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5