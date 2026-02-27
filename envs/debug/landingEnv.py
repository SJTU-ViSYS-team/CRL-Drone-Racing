from collections import deque
import numpy as np
from envs.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from collections import deque
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt

from utils.type import TensorDict


class LandingEnv(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = None,
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
    ):
        random_kwargs = {
            "state_generator":
                {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [2., 0., 3.0], "half": [0.5, 0.5, 0.5]}},
                        # {"position": {"mean": [0., 0., 1.5], "half": [0.5, 0.5, 0.5]}},
                    ]
                },
        }
        dynamics_kwargs = {
            "dt": 0.01,
            "ctrl_dt": 0.03,
            "action_type": "bodyrate",
            "ctrl_delay": True,
        }
        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            sensor_kwargs=sensor_kwargs,
            scene_kwargs=scene_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
        )

        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([2., 0., 0.1] if target is None else target).reshape(1, -1)
        # if is_eval:
        #     self.target = th.as_tensor([[0., 1., 0],[0., 0., 0],[0., -1., 0]])
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
        })
        self.initial_height = 1.5
        self.previous_position = deque(maxlen=2)  # 初始化上一步位32
        self.pastAction = th.zeros((self.num_envs, 12))  # 初始化过去动作
        self.previous_actions = deque(maxlen=4)  # 初始化动作队列
        self.last_action = th.zeros((self.num_envs, 4)) 
        self.last_position = th.zeros((self.num_envs, 3))

    def get_success(self) -> th.Tensor:
        landing_half = 0.3
        return (self.position-self.target).norm(dim=1) < landing_half 
    
    def get_failure(self) -> th.Tensor:
        return self.is_collision
    #
    # def get_reward(self) -> th.Tensor:
    #     eta = th.as_tensor(1.2)
    #     v_l = -(0.1 * (eta.pow(self.position[:, 2])-1)).clip(min=0.05, max=1).clone().detach()
    #     r_p = -0.1
    #     descent_v = self.velocity[:, 2] - 0
    #     # r_z_punish = ((descent_v > v_l) | (descent_v < 0))
    #     # r_z = r_z_punish * r_p + \
    #     #       ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1
    #
    #     r_z_first = descent_v <= v_l
    #     r_z = ~r_z_first * ((eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1)+ \
    #           r_z_first * ((eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1)
    #
    #     # remove nan value
    #     r_z = (descent_v - v_l).abs() * -0.2
    #     d_v = (descent_v - v_l).abs()
    #     # r_z = ((eta.pow(1-d_v / 1) - 1) / (eta - 1)) * 0.1 # + (descent_v <= 0) * -0.1
    #     # r_z = th.where(th.isnan(r_z), th.zeros_like(r_z), r_z)
    #     rho = th.as_tensor(1.2)
    #     d_s = 2. * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
    #     d_x = (self.target - self.position)[:, 0].abs()
    #     d_y = (self.target - self.position)[:, 1].abs()
    #     r_xy_punish = d_xy > d_s
    #     r_xy = (rho.pow(1 - d_xy / d_s) - 1) / (rho - 1) * 0.1
    #     r_x = (rho.pow(1 - d_x / d_s) - 1) / (rho - 1) * 0.1
    #     r_y = (rho.pow(1 - d_y / d_s) - 1) / (rho - 1) * 0.1
    #     r_x = -d_x * 0.1
    #     r_y = -d_y * 0.1
    #     # toward_v = ((self.velocity[:, :2] -0)* ((self.target - self.position)[:, :2])).sum(dim=1) / d_xy
    #     # r_xy_is_first_sec = toward_v <= v_l
    #     # r_xy = r_xy_is_first_sec * 0.1 * (rho.pow(toward_v/v_l)-1)/(rho-1)+ \
    #     #     ~r_xy_is_first_sec * 0.1*(rho.pow(-4 * descent_v / v_l + 5) - 1) / (rho-1) * 0.1
    #     r_ori = (self.orientation - th.tensor([1,0,0,0])).norm(dim=1)* (2-self.position[:, 2]).clamp_min(0.)
    #     r_omega = (self.angular_velocity - 0).norm(dim=1) * (2-self.position[:, 2]).clamp_min(0.2)
    #     r_s = 20.
    #     r_l = self.success * r_s + self.failure * -0.1
    #     # reward = 1. * r_l + 1. * r_x + r_y + 1. * r_z + 0.1 * r_ori + 0.0001 * r_omega * 0.0001
    #     disc_r = r_l
    #     diff_r = 1. * r_x + r_y + 1. * r_z + 0.1 * r_ori + 0.0001 * r_omega * 0.0001
    #     r = {
    #         "reward": disc_r + diff_r,
    #         "disc_r": disc_r,
    #         "diff_r": diff_r
    #     }
    #     return r

    # def get_reward(self) -> th.Tensor:
    #     eta = th.as_tensor(1.2)
    #     v_l = 0.5 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     r_p = -0.1
    #     descent_v = -self.velocity[:, 2] - 0
    #     # r_z_punish = ((descent_v > v_l) | (descent_v < 0))
    #     # r_z = r_z_punish * r_p + \
    #     #       ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1
    #
    #     r_z_first = descent_v <= v_l
    #     r_z = ~r_z_first * ((eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1)+ \
    #           r_z_first * ((eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1)
    #     # remove nan value
    #     # r_z = (descent_v - v_l).abs() * -0.1
    #     r_z = th.where(th.isnan(r_z), th.zeros_like(r_z), r_z)
    #     rho = th.as_tensor(1.2)
    #     d_s = 2. * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
    #     r_xy_punish = d_xy > d_s
    #     r_xy = (rho.pow(1 - d_xy / d_s) - 1) / (rho - 1) * 0.1
    #
    #     # toward_v = ((self.velocity[:, :2] -0)* ((self.target - self.position)[:, :2])).sum(dim=1) / d_xy
    #     # r_xy_is_first_sec = toward_v <= v_l
    #     # r_xy = r_xy_is_first_sec * 0.1 * (rho.pow(toward_v/v_l)-1)/(rho-1)+ \
    #     #     ~r_xy_is_first_sec * 0.1*(rho.pow(-4 * descent_v / v_l + 5) - 1) / (rho-1) * 0.1
    #     r_ori = (self.orientation - th.tensor([1,0,0,0])).norm(dim=1)* (2-self.position[:, 2]).clamp_min(0.)
    #     r_omega = (self.angular_velocity - 0).norm(dim=1) * (2-self.position[:, 2]).clamp_min(0.2)
    #     r_s = 20.
    #     r_l = self.success * r_s + self.failure * -0.1
    #     reward = 1. * r_l + 1. * r_xy + 1. * r_z + 0.1 * r_ori + 0.0001 * r_omega * 0.1
    #
    #     return reward
    
    # # reward 1
    # def get_reward(self) -> th.Tensor:
    #     eta = th.as_tensor(1.2)
    #     v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.05, max=0.5).clone().detach()
    #     # v_l = 1 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     r_p = -0.1
    #     descent_v = -self.velocity[:, 2] - 0
    #     r_z_punish = ((descent_v > v_l) | (descent_v < 0))
    #     r_z = r_z_punish * r_p + \
    #           ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1

    #     r_z_first = descent_v <= v_l
    #     r_z = ~r_z_first * (eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1 + \
    #           r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1

    #     r_z = 1/(1+(-v_l-self.velocity[:,-1]).abs()) * 0.1
    #     d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
    #     r_xy = -d_xy * 0.02 * 2
    #     r_cmd = -0.005 * (self._action - 0).norm(dim=1) - 0.0025 * (self._action - self.last_action).norm(dim=1)
    #     r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
    #     r_s = 5.
        
    #     # reward = r_l + r_xy + r_z + r_omega #+ base_r
    #     r_l = self.success * r_s + self.failure * -1.
    #     diff_r = r_xy + r_z
    #     disc_r = r_l
    #     reward = diff_r + disc_r
    #     return reward
    
    #     # reward 2
    # def get_reward(self) -> th.Tensor:
    #     eta = th.as_tensor(1.2)
    #     v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.05, max=0.5).clone().detach()
    #     # v_l = 1 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     r_p = -0.1
    #     base_r = th.ones((self.num_envs,)) * 0.1
    #     descent_v = -self.velocity[:, 2] - 0
    #     r_z_punish = ((descent_v > v_l) | (descent_v < 0))
    #     r_z = r_z_punish * r_p + \
    #           ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1
    #     r_z_first = descent_v <= v_l
    #     r_z = ~r_z_first * (eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1 + \
    #           r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1
    #     # r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1
    #     d_z = (self.target - self.position)[:, 2].abs()
    #     # r_z = -0.02 * d_z
    #     # r_z = -(-v_l-self.velocity[:,-1]).abs() * 0.02
    #     r_z = 1/(1+(-v_l-self.velocity[:,-1]).abs()) * 0.1
    #     rho = th.as_tensor(1.2)
    #     d_s = 2. * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
    #     r_xy_punish = d_xy > d_s
    #     r_xy = (rho.pow(1 - d_xy / d_s) - 1) / (rho - 1) * 0.1
    #     r_xy = -d_xy * 0.02 * 2
    #     d_x = (self.target - self.position)[:,0].abs()
    #     d_y = (self.target - self.position)[:,1].abs()
    #     r_x = -d_x * 0.02
    #     r_y = -d_y * 0.02
    #     r_cmd = -0.005 * (self._action - 0).norm(dim=1) - 0.0025 * (self._action - self.last_action).norm(dim=1)
    #     r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
    #     r_s = 5.
    #     r_l = self.success * r_s + self.failure * -1.
    #     diff_r = r_xy + r_z + r_omega + r_cmd
    #     disc_r = r_l
    #     reward = diff_r + disc_r
    #     return reward
    
    # # reward 3
    # def get_reward(self) -> th.Tensor:
    #     eta = th.as_tensor(1.2)
    #     v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.05, max=0.5).clone().detach()
    #     # v_l = 1 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     r_p = -0.1
    #     base_r = th.ones((self.num_envs,)) * 0.0
    #     descent_v = -self.velocity[:, 2] - 0
    #     r_z_punish = ((descent_v > v_l) | (descent_v < 0))
    #     r_z = r_z_punish * r_p + \
    #           ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1

    #     r_z_first = descent_v <= v_l
    #     r_z = ~r_z_first * (eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1 + \
    #           r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1
    #     # r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1
    #     d_z = (self.target - self.position)[:, 2].abs()
    #     # r_z = -0.02 * d_z
    #     # r_z = -(-v_l-self.velocity[:,-1]).abs() * 0.02

    #     r_z = 1/(1+(v_l+self.velocity[:,-1]).abs()) * 0.1

    #     rho = th.as_tensor(1.2)
    #     d_s = 2. * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
    #     r_xy_punish = d_xy > d_s
    #     r_xy = (rho.pow(1 - d_xy / d_s) - 1) / (rho - 1) * 0.1
    #     r_xy = -d_xy * 0.02 * 2
    #     d_x = (self.target - self.position)[:,0].abs()
    #     d_y = (self.target - self.position)[:,1].abs()
    #     r_x = -d_x * 0.02
    #     r_y = -d_y * 0.02
    #     r_cmd = -0.005 * (self._action - 0).norm(dim=1) - 0.0025 * (self._action - self.last_action).norm(dim=1)

    #     r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
    #     r_s = 5.
    #     # reward = r_l + r_xy + r_z + r_omega #+ base_r
    #     r_l = self.success * r_s + self.failure * -4.
    #     diff_r = r_xy + r_z + r_cmd
    #     disc_r = r_l
    #     reward = diff_r + disc_r
    #     return reward

    # # reward 4
    # def get_reward(self) -> th.Tensor:
    #     eta = th.as_tensor(1.2)
    #     v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.05, max=0.5).clone().detach()
    #     # v_l = 1 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     r_p = -0.1
    #     base_r = th.ones((self.num_envs,)) * 0.0
    #     descent_v = -self.velocity[:, 2] - 0
    #     r_z_punish = ((descent_v > v_l) | (descent_v < 0))
    #     r_z = r_z_punish * r_p + \
    #           ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1

    #     r_z_first = descent_v <= v_l
    #     r_z = ~r_z_first * (eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1 + \
    #           r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1
    #     # r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1
    #     d_z = (self.target - self.position)[:, 2].abs()
    #     # r_z = -0.02 * d_z
    #     # r_z = -(-v_l-self.velocity[:,-1]).abs() * 0.02

    #     r_z = 1/(1+(-v_l-self.velocity[:,-1]).abs()) * 0.1

    #     rho = th.as_tensor(1.2)
    #     d_s = 2. * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
    #     r_xy_punish = d_xy > d_s
    #     r_xy = (rho.pow(1 - d_xy / d_s) - 1) / (rho - 1) * 0.1
    #     r_xy = -d_xy * 0.02 * 2
    #     d_x = (self.target - self.position)[:,0].abs()
    #     d_y = (self.target - self.position)[:,1].abs()
    #     r_x = -d_x * 0.02
    #     r_y = -d_y * 0.02
    #     r_cmd = -0.005 * (self._action - 0).norm(dim=1) - 0.0025 * (self._action - self.last_action).norm(dim=1)

    #     r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
    #     r_s = 5.
    #     # reward = r_l + r_xy + r_z + r_omega #+ base_r
    #     r_l = self.success * r_s + self.failure * -4.
    #     diff_r = r_xy + r_z + r_omega + r_x + r_y + r_cmd
    #     disc_r = r_l
    #     reward = diff_r + disc_r
    #     return reward
    
    #     # reward full
    # def get_reward(self) -> th.Tensor:
    #     eta = th.as_tensor(1.2)
    #     v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.05, max=0.5).clone().detach()
    #     # v_l = 1 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     r_p = -0.1
    #     descent_v = -self.velocity[:, 2] - 0
    #     r_z_punish = ((descent_v > v_l) | (descent_v < 0))
    #     r_z = r_z_punish * r_p + \
    #           ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1

    #     r_z_first = descent_v <= v_l
    #     r_z = ~r_z_first * (eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1 + \
    #           r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1

    #     r_z = 1/(1+(-v_l-self.velocity[:,-1]).abs()) * 0.1
    #     d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
    #     r_xy = -d_xy * 0.02 * 2
    #     r_cmd = -0.005 * (self._action - 0).norm(dim=1) - 0.0025 * (self._action - self.last_action).norm(dim=1)
    #     r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
    #     r_s = 5.
        
    #     # reward = r_l + r_xy + r_z + r_omega #+ base_r
    #     r_l = self.success * r_s + self.failure * -1.
    #     diff_r = r_xy + r_z + r_cmd + r_omega
    #     disc_r = r_l
    #     reward = diff_r + disc_r
    #     return reward
    
    #         # reward w/o r_z
    # def get_reward(self) -> th.Tensor:
    #     eta = th.as_tensor(1.2)
    #     v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.05, max=0.5).clone().detach()
    #     # v_l = 1 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     r_p = -0.1
    #     descent_v = -self.velocity[:, 2] - 0
    #     r_z_punish = ((descent_v > v_l) | (descent_v < 0))
    #     r_z = r_z_punish * r_p + \
    #           ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1

    #     r_z_first = descent_v <= v_l
    #     r_z = ~r_z_first * (eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1 + \
    #           r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1

    #     r_z = 1/(1+(-v_l-self.velocity[:,-1]).abs()) * 0.1
    #     d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
    #     r_xy = -d_xy * 0.02 * 2
    #     r_cmd = -0.005 * (self._action - 0).norm(dim=1) - 0.0025 * (self._action - self.last_action).norm(dim=1)
    #     r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
    #     r_s = 5.
        
    #     # reward = r_l + r_xy + r_z + r_omega #+ base_r
    #     r_l = self.success * r_s + self.failure * -1.
    #     diff_r = r_xy + r_cmd + r_omega
    #     disc_r = r_l
    #     reward = diff_r + disc_r
    #     return reward
    
    #             # reward w/o sparse
    # def get_reward(self) -> th.Tensor:
    #     eta = th.as_tensor(1.2)
    #     v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.05, max=0.5).clone().detach()
    #     # v_l = 1 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     r_p = -0.1
    #     descent_v = -self.velocity[:, 2] - 0
    #     r_z_punish = ((descent_v > v_l) | (descent_v < 0))
    #     r_z = r_z_punish * r_p + \
    #           ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1

    #     r_z_first = descent_v <= v_l
    #     r_z = ~r_z_first * (eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1 + \
    #           r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1

    #     r_z = 1/(1+(-v_l-self.velocity[:,-1]).abs()) * 0.1
    #     d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
    #     r_xy = -d_xy * 0.02 * 2
    #     r_cmd = -0.005 * (self._action - 0).norm(dim=1) - 0.0025 * (self._action - self.last_action).norm(dim=1)
    #     r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
    #     r_s = 5.
        
    #     # reward = r_l + r_xy + r_z + r_omega #+ base_r
    #     r_l = self.success * r_s + self.failure * -1.
    #     diff_r = r_xy + r_cmd + r_omega + r_z
    #     disc_r = r_l
    #     reward = diff_r
    #     return reward
    
    # # reward w/o anglev
    # def get_reward(self) -> th.Tensor:
    #     eta = th.as_tensor(1.2)
    #     v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.05, max=0.5).clone().detach()
    #     # v_l = 1 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
    #     r_p = -0.1
    #     descent_v = -self.velocity[:, 2] - 0
    #     r_z_punish = ((descent_v > v_l) | (descent_v < 0))
    #     r_z = r_z_punish * r_p + \
    #           ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1

    #     r_z_first = descent_v <= v_l
    #     r_z = ~r_z_first * (eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1 + \
    #           r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1

    #     r_z = 1/(1+(-v_l-self.velocity[:,-1]).abs()) * 0.1
    #     d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
    #     r_xy = -d_xy * 0.02 * 2
    #     r_cmd = -0.005 * (self._action - 0).norm(dim=1) - 0.0025 * (self._action - self.last_action).norm(dim=1)
    #     r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
    #     r_s = 5.
        
    #     # reward = r_l + r_xy + r_z + r_omega #+ base_r
    #     r_l = self.success * r_s + self.failure * -1.
    #     diff_r = r_xy + r_cmd + r_z
    #     disc_r = r_l
    #     reward = diff_r + disc_r
    #     return reward
    
        # reward w/o prog
    def get_reward(self) -> th.Tensor:
        eta = th.as_tensor(1.2)
        v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.05, max=0.5).clone().detach()
        # v_l = 1 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
        r_p = -0.1
        descent_v = -self.velocity[:, 2] - 0
        r_z_punish = ((descent_v > v_l) | (descent_v < 0))
        r_z = r_z_punish * r_p + \
              ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1

        r_z_first = descent_v <= v_l
        r_z = ~r_z_first * (eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1 + \
              r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1

        r_z = 1/(1+(-v_l-self.velocity[:,-1]).abs()) * 0.1
        d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
        r_xy = -d_xy * 0.02 * 2
        r_cmd = -0.005 * (self._action - 0).norm(dim=1) - 0.0025 * (self._action - self.last_action).norm(dim=1)
        r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
        r_s = 5.
        
        # reward = r_l + r_xy + r_z + r_omega #+ base_r
        r_l = self.success * r_s + self.failure * -1.
        diff_r = r_cmd + r_z + r_omega
        disc_r = r_l
        reward = diff_r + disc_r
        return reward
    
    def get_observation(
            self,
            indices=None
    ) -> Dict:
        
        self.previous_position.append(self.position.clone())
        self.previous_actions.append(self._action.clone())
        
        if len(self.previous_position) > 1:
            self.last_position= self.previous_position[-2]
        if len(self.previous_actions) > 2:
            self.pastAction = th.cat(list(self.previous_actions)[:3], dim=-1)
            self.last_action = self.previous_actions[-2] #倒数第二个应该才是上一步的动作
            
        state = th.hstack([
            (self.target - self.position) / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        return TensorDict({
            "state": state,
        })