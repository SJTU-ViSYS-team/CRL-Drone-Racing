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


class TrackEnv(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = {},
            dynamics_kwargs: dict = {},
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
            sensor_kwargs=sensor_kwargs,
            scene_kwargs=scene_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            latent_dim=latent_dim,
        )
        
        self.vd = th.tensor([2.0, 0.0, 0.0])
        self.v_d = 2.0*th.ones((self.num_envs,),dtype=th.float)
        # self.previous_position = deque(maxlen=2)  # 初始化上一步位32
        self.pastAction = th.zeros((self.num_envs, 12))  # 初始化过去动作
        self.previous_actions = deque(maxlen=4)  # 初始化动作队列
        self.last_action = th.zeros((self.num_envs, 4)) 
        # self.last_position = th.zeros((self.num_envs, 3))
        
        random_kwargs = {
            "state_generator":
                {
                    "class": "Union",
                    "kwargs": [
                        {"randomizers_kwargs":
                            [
                                {
                                    "class": "Uniform",
                                    "kwargs":
                                        {"position": {"mean": [1., 0., 1], "half": [.2, .2, 0.2]}
                                        #  "velocity": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.1]}
                                         },

                                },
                            ]
                        }
                    ]

                }
        }
        
        sensor_kwargs = [{
            "sensor_type": SensorType.DEPTH,
            "uuid": "depth",
            "position": [0.0, 0.0, -0.2],
            "resolution": [64, 64],
        }]
        
        # sensor_kwargs = []
        dynamics_kwargs = {
            "dt": 0.02,
            "ctrl_dt": 0.02,
            "action_type": "bodyrate",
            "ctrl_delay": True,
        }


        self.observation_space["state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_space["state"].shape[0],),
            dtype=np.float32
        )

    def get_observation(
            self,
            indices=None
    ) -> Dict:

        # self.previous_position.append(self.position.clone())
        self.previous_actions.append(self._action.clone())
        
        # if len(self.previous_position) > 1:
        #     self.last_position= self.previous_position[-2]
        if len(self.previous_actions) > 2:
            self.pastAction = th.cat(list(self.previous_actions)[:3], dim=-1)
            self.last_action = self.previous_actions[-2] #倒数第二个应该才是上一步的动作
        
        state = th.hstack([
            self.position / 10,
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
                    # "latent": self.latent.cpu().numpy(),
                })
            else:
                return TensorDict({
                    # "index": self._next_target_i.clone().detach().cpu().numpy().reshape(-1, 1),
                    "state": state,
                    # "vd": self.v_d.unsqueeze(1).cpu().numpy(),
                    # "latent": self.latent.cpu().numpy(),
                    # "pastAction": self.pastAction.cpu().numpy()
                })

    def world_to_body(self, relative_pos_world):
        rotation = R.from_quat(self.orientation.cpu().numpy())
        rotation_matrix = th.from_numpy(rotation.as_matrix()).to(self.device).float()  
        relative_pos_world = relative_pos_world.float()  
        relative_pos_body = th.einsum('bij,bjk->bik', rotation_matrix.transpose(1, 2), relative_pos_world.transpose(1, 2)).transpose(1, 2)
        return relative_pos_body
    
    def get_success(self) -> th.Tensor:
        return th.full((self.num_agent,), False)
        # return (self.position - self.target).norm(dim=1) < self.success_radius
    
    def get_reward(self) -> th.Tensor:
        lambda1 = -2e-4
        lambda2 = -1.2e-3
        lambda3 = -1e-4
        
        r_vel = lambda1 * (self.velocity-self.vd).norm(dim=1)
        # r_vel = lambda1 * ((self.velocity).norm(dim=1)-self.v_d).abs()
        r_anglevel = lambda2 *  (self.angular_velocity - th.tensor([0, 0, 0])).norm(dim=1)
        r_cmd = lambda3 * (self._action - self.last_action).norm(dim=1)
        r_crash = -4.0  * self.is_collision

        reward = r_vel + r_cmd + r_anglevel + r_crash
        return reward  

class TrackEnv2(TrackEnv):

    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = {},
            dynamics_kwargs: dict = {},
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
    



