import os
import sys

import numpy as np
from envs.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
# from ..utils.tools.train_encoder import model as encoder
from utils.type import TensorDict
from scipy.spatial.transform import Rotation as R
from collections import deque

class HoverEnv(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = None,
            dynamics_kwargs: dict = None,
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
            latent_dim=None,
    ):
        # sensor_kwargs = [{
        #     "sensor_type": SensorType.DEPTH,
        #     "uuid": "depth",
        #     "resolution": [64, 64],
        # }]
        sensor_kwargs = []
        random_kwargs = {
            "state_generator":
                {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [2., 0., 2.5], "half": [1., 1., 1.]},
                         "velocity": {"mean": [3., 3., 3.], "half": [0.5, 0.5, 0.5]}}
                    ]
                }
        } if random_kwargs is None else random_kwargs
        
        dynamics_kwargs = {
            "dt": 0.02,
            "ctrl_dt": 0.02,
            "action_type": "bodyrate",
            "ctrl_delay":True,
        } if dynamics_kwargs is None else dynamics_kwargs

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
            latent_dim=latent_dim,

        )

        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([1, 0., 1.5] if target is None else target).reshape(1,-1)
        self.success_radius = 0.1
        self.previous_position = deque(maxlen=2)  # 初始化上一步位32
        self.pastAction = th.zeros((self.num_envs, 12))  # 初始化过去动作
        self.previous_actions = deque(maxlen=4)  # 初始化动作队列
        self.last_action = th.zeros((self.num_envs, 4)) 
        self.last_position = th.zeros((self.num_envs, 3))

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        obs = TensorDict({
            "state": self.state,
        })

        if self.latent is not None:
            if not self.requires_grad:
                obs["latent"] = self.latent.cpu().numpy()
            else:
                obs["latent"] = self.latent

        return obs

    def world_to_body(self, relative_pos_world):
        rotation = R.from_quat(self.orientation.cpu().numpy())
        rotation_matrix = th.from_numpy(rotation.as_matrix()).to(self.device).float()  
        relative_pos_world = relative_pos_world.float()  
        relative_pos_body = th.einsum('bij,bjk->bik', rotation_matrix.transpose(1, 2), relative_pos_world.transpose(1, 2)).transpose(1, 2)
        return relative_pos_body
    
    def get_success(self) -> th.Tensor:
        # return th.full((self.num_agent,), False)
        return (self.position - self.target).norm(dim=1) < self.success_radius


    def get_reward(self) -> th.Tensor:
        lambda1 = 0.002
        lambda2 = 0.0002
        lambda3 = 0.00004
        lambda4 = 0.00001
        lambda5 = 0.0001
        
        r_height = -lambda1 * (self.position[:, 2] - self.target[:, 2]).abs()
        r_ori = -lambda2 * (self.orientation - 0).norm(dim=1)
        r_v = -lambda3 * (self.velocity - 0).norm(dim=1) 
        r_cmd = lambda5 * (self._action - self.last_action).norm(dim=1)
        r_crash = -4.0  * self.is_collision
        r_hover = 10.0 * self.get_success()
        reward = r_height + r_hover + r_cmd + r_ori + r_crash + r_v
        
        return reward  
    
class HoverEnv2(HoverEnv):

    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = None,
            dynamics_kwargs: dict = None,
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
            latent_dim=None,
    ):
        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            target=target,
            latent_dim=latent_dim
        )

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        
        relative_pos_world = (self.target - self.position)
        # relative_pos_body = self.world_to_body(relative_pos_world)
        # relative_pos = relative_pos_body.reshape(self.num_envs, -1)
        
        # 添加噪声
        # noise = th.randn_like(relative_pos) * 0.1  # 生成与 relative_pos 形状相同的噪声，标准差为 0.1
        # relative_pos_noisy = relative_pos + noise
        
        self.previous_position.append(self.position.clone())
        self.previous_actions.append(self._action.clone())
        
        if len(self.previous_position) > 1:
            self.last_position= self.previous_position[-2]
        if len(self.previous_actions) > 2:
            self.pastAction = th.cat(list(self.previous_actions)[:3], dim=-1)
            self.last_action = self.previous_actions[-2] #倒数第二个应该才是上一步的动作
        
        
        state = th.hstack([
            relative_pos_world / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        if not self.requires_grad:
            if self.visual:
                return TensorDict({
                    # "index": self._next_target_i.clone().detach().cpu().numpy().reshape(-1, 1),
                    "state": state,
                    # "pastAction": self.pastAction.cpu().numpy(),
                    # "vd": self.v_d.unsqueeze(1).cpu().numpy(),
                    # "depth": self.sensor_obs["depth"],
                    "latent": self.latent.cpu().numpy(),
                })
            else:
                return TensorDict({
                    # "index": self._next_target_i.clone().detach().cpu().numpy().reshape(-1, 1),
                    "state": state,
                    # "vd": self.v_d.unsqueeze(1).cpu().numpy(),
                    "latent": self.latent.cpu().numpy(),
                    # "pastAction": self.pastAction.cpu().numpy()
                })