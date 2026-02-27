import os
import sys

import numpy as np
from envs.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from collections import deque
from habitat_sim import SensorType
from gymnasium import spaces
# from utils.tools.train_encoder import model as encoder
from utils.type import TensorDict


class HoverEnv(DroneGymEnvsBase):
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
            # tensor_output: bool = False,
    ):

        random_kwargs = {
            "state_generator":
                {
                    "class": "Uniform",
                    "kwargs": [
                        # {"position": {"mean": [1., 0., 1.5], "half": [0.0, 0.0, 0.0]}},
                        {"position": {"mean": [4., 2., 1.5], "half": [0.5, 0.5, 0.5]}},
                    ]
                }
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
            # tensor_output=tensor_output,

        )

        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([1, 0., 1.] if target is None else target).reshape(1,-1)
        self.success_radius = 0.3
        self.previous_position = deque(maxlen=2)  # 初始化上一步位32
        # self.pastAction = th.zeros((self.num_envs, 12))  # 初始化过去动作
        self.previous_actions = deque(maxlen=4)  # 初始化动作队列
        self.last_action = th.zeros((self.num_envs, 4)) 
        self.last_position = th.zeros((self.num_envs, 3))

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        self.previous_position.append(self.position.clone())
        self.previous_actions.append(self._action.clone())
        
        if len(self.previous_position) > 1:
            self.last_position= self.previous_position[-2]
        if len(self.previous_actions) > 2:
            # self.pastAction = th.cat(list(self.previous_actions)[:3], dim=-1)
            self.last_action = self.previous_actions[-2] 
            
        obs = TensorDict({
            "state": self.state,
        })

        # if self.latent is not None:
        #     if not self.requires_grad:
        #         obs["latent"] = self.latent.cpu().numpy()
        #     else:
        #         obs["latent"] = self.latent

        return obs
    
    def get_position_error(self):
        return (self.position - self.target).norm(dim=1)

    def get_success(self) -> th.Tensor:
        # return th.full((self.num_agent,), False)
        return (self.position - self.target).norm(dim=1) < self.success_radius

    # # reward1
    # def get_reward(self) -> th.Tensor:
    #     base_r = 0.1
    #     pos_factor = -0.1
    #     pos_error = (self.position - self.target).norm(dim=1)
    #     reward = (
    #             base_r +
    #              pos_error * pos_factor +
    #              (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
    #              (self.velocity - 0).norm(dim=1) * -0.002 +
    #              (self.angular_velocity - 0).norm(dim=1) * -0.002
    #     )
    #     return {"reward":reward, "position_error": pos_error/100}

    # reward2
    def get_reward(self) -> th.Tensor:
        base_r = 0.1
        pos_factor = -0.1
        pos_error = (self.position - self.target).norm(dim=1)
        reward = (
                base_r +
                 pos_error * pos_factor +
                 (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
                 (self.velocity - 0).norm(dim=1) * -0.002 +
                 (self.angular_velocity - 0).norm(dim=1) * -0.002 +
                 5 * self.get_success().float() +
                 -4. * self.failure
        )
        # return {"reward":reward, "position_error": pos_error}
        return reward
    
    # reward3
    def get_reward(self) -> th.Tensor:
        base_r = 0.1
        lamada1 = 0.1
        pos_error = (self.position - self.target).norm(dim=1)
        pos_factor = -0.1 * 1/9
        r_prog1 = lamada1 * ((self.last_position - self.target).norm(dim=1)-(self.position - self.target).norm(dim=1))
        reward = (
                base_r +
                 r_prog1 +
                 (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
                 (self.velocity - 0).norm(dim=1) * -0.002 +
                 (self.angular_velocity - 0).norm(dim=1) * -0.002
        )
        return {"reward":reward, "position_error": pos_error/100}

