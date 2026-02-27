import numpy as np
from envs.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from collections import deque
# from ..utils.tools.train_encoder import model as encoder
from utils.type import TensorDict
from scipy.spatial.transform import Rotation as R
# is_pos_reward = True


class RacingEnv(DroneGymEnvsBase):
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
                                        {"position": {"mean": [2., 2., 1], "half": [.2, .2, 0.2]}},

                                },
                                {
                                    "class": "Uniform",
                                    "kwargs":
                                        {"position": {"mean": [10., 2., 1.], "half": [.2, .2, 0.2]}},

                                },
                                {
                                    "class": "Uniform",
                                    "kwargs":
                                        {"position": {"mean": [2., -2., 1], "half": [.2, .2, 0.2]}},

                                },
                                {
                                    "class": "Uniform",
                                    "kwargs":
                                        {"position": {"mean": [10., -2., 1], "half": [.2, .2, 0.2]}},

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
        # random obstacle part
        # self.file_path = 'datasets/spy_datasets/configs/racing8/racing8.scene_instance.json'
        # self.num_obstacles = 6
        
        # pastactions and targets
        self.previous_position = deque(maxlen=2)  # 初始化上一步位32
        self.pastAction = th.zeros((self.num_envs, 12))  # 初始化过去动作
        self.previous_actions = deque(maxlen=4)  # 初始化动作队列
        self.last_action = th.zeros((self.num_envs, 4)) 
        self.last_position = th.zeros((self.num_envs, 3))
        self.targets = th.as_tensor([
            [4, 2, 1],    # 第一个门
            [2, 0, 1],    # 第二个门
            [4, -2, 1],   # 第三个门
            [8, -2, 1],   # 第四个门
            [10, 0, 1],   # 第五个门
            [8, 2, 1],  # 第六个门
        ])
        self.orientations = th.as_tensor([
            [-0.5,  0.5,  0.5, -0.5],
            [-0.70710678, 0, 0, -0.70710678],
            [-0.5,  0.5,  0.5, -0.5],
            [-0.5,  0.5,  0.5, -0.5],
            [-0.70710678, 0, 0, -0.70710678],
            [-0.5,  0.5,  0.5, -0.5],
        ])

        self.yaw_errors = th.zeros((self.num_envs, 1),dtype=float)
        
        self.length_target = len(self.targets)
        self._next_target_num = 2
        self._next_target_i = th.zeros((self.num_envs,), dtype=th.int)
        self._past_targets_num = th.zeros((self.num_envs,), dtype=th.int)
        self._is_pass_next = th.zeros((self.num_envs,), dtype=th.bool)
        self.success_radius = 0.3
        
        self.v_d = 2.0*th.ones((self.num_envs,),dtype=th.float)
        
        
        self.total_timesteps = 0
        self.target_update_interval = 10000
        
        self.observation_space["vd"] = spaces.Box(
            low=0.,
            high=30.,
            shape=(1,),
            dtype=np.float32
        )
        
        self.observation_space["index"] = spaces.Box(
            low=0,
            high=len(self.targets),
            shape=(1,),
            dtype=np.int32
        )
        # state observation includes gates
        self.observation_space["state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 * (self._next_target_num - 1) + self.observation_space["state"].shape[0],),
            dtype=np.float32
        )
        # self.observation_space["pastAction"] = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        # self.observation_space["noise_target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.success_r = 5

    def collect_info(self, indice, observations):
        _info = super().collect_info(indice, observations)
        _info['episode']["extra"]["past_gate"] = self._past_targets_num[indice].item()
        return _info

    @property
    def is_pass_next(self):
        return self._is_pass_next

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        
        if not self.requires_grad:
            if self.visual:
                obs = TensorDict({
                    "state": self.state.cpu().numpy(),
                    "vd": self.v_d.cpu().numpy(),
                    "index": self._next_target_i.cpu().numpy(),
                    # "pastAction": self.pastAction.cpu().numpy(),
                    # "noise_target" : self.noise_target.cpu().numpy(),
                    "latent": self.latent.cpu().numpy(),
                    "depth": self.sensor_obs["depth"],
                })
            else:
                obs = TensorDict({
                    "state": self.state.cpu().numpy(),
                    "vd": self.v_d.cpu().numpy(),
                    "index": self._next_target_i.cpu().numpy(),
                    # "noise_target" : self.noise_target.cpu().numpy(),
                    "latent": self.latent.cpu().numpy(),
                    # "pastAction": self.pastAction.cpu().numpy()
                })
        else:
            if self.visual:
                obs = TensorDict({
                    "state": self.state,
                    "vd": self.v_d,
                    "index": self._next_target_i,
                    # "pastAction": self.pastAction,
                    "latent": self.latent,
                    # "noise_target": self.noise_target,
                    "depth": th.from_numpy(self.sensor_obs["depth"]),
                })
            else:
                obs = TensorDict({
                    "state": self.state,
                    "vd": self.v_d,
                    "index": self._next_target_i,
                    "latent": self.latent,
                    # "noise_target": self.noise_target,
                    # "pastAction": self.pastAction
                })

        return obs

    def get_success(self) -> th.Tensor:
        _next_target_i_clamp = self._next_target_i
        self._is_pass_next = ((self.position - self.targets[_next_target_i_clamp]).norm(dim=1) <= self.success_radius)
        self._next_target_i = self._next_target_i + self._is_pass_next
        self._next_target_i = self._next_target_i % len(self.targets)
        self._past_targets_num = self._past_targets_num + self._is_pass_next
        return th.zeros((self.num_envs,), dtype=th.bool)

    def reset_by_id(self, indices=None, state=None, reset_obs=None):
        indices = th.arange(self.num_envs) if indices is None else indices
        if reset_obs is not None:
            self._next_target_i = reset_obs["index"].to(self.device).squeeze()
        else:
            self._choose_target(indices=indices)
            # self._random_obstacle(indices=indices)
            # self._next_target_i[indices] = th.zeros((len(indices),), dtype=th.int)

        self._past_targets_num[indices] = th.zeros((len(indices),), dtype=th.int)
        self._is_pass_next[indices] = th.zeros((len(indices),), dtype=th.bool)

        obs = super().reset_by_id(indices, state, reset_obs)

        return obs

    def reset(self, state=None, obs=None):
        obs = super().reset(state)
        self._next_target_i = th.zeros((self.num_envs,), dtype=th.int)
        # self._past_targets_num = th.zeros((self.num_envs,), dtype=th.int)
        self._choose_target()
        # self._random_obstacle()
        return obs
    
    def world_to_body(self, relative_pos_world):
        # 使用四元数将世界坐标系下的相对坐标转换到机体系下
        rotation = R.from_quat(self.orientation.cpu().numpy())
        rotation_matrix = th.from_numpy(rotation.as_matrix()).to(self.device).float()  # 确保 rotation_matrix 是 float 类型
        relative_pos_world = relative_pos_world.float()  # 确保 relative_pos_world 是 float 类型

        # 进行矩阵乘法，将世界坐标系下的相对坐标转换到机体系下
        relative_pos_body = th.einsum('bij,bjk->bik', rotation_matrix.transpose(1, 2), relative_pos_world.transpose(1, 2)).transpose(1, 2)
        return relative_pos_body
    
    def compute_yaw_error(self, _next_target_i_clamp)-> th.Tensor:
        # 使用方括号而不是圆括号来索引张量
        target_orientation = self.orientations[_next_target_i_clamp]
        rotation = R.from_quat((self.orientation - target_orientation).cpu().numpy())
        # 将旋转矩阵转换为欧拉角，顺序为 'xyz'，即 ['roll', 'pitch', 'yaw']
        euler_angles = rotation.as_euler('xyz', degrees=False)
        # 提取偏航角（yaw angle）
        self.yaw_errors = euler_angles[:, 2]  # 确保提取的是所有环境的偏航角
        # self.yaw_errors = th.tensor(yaw_error, dtype=th.float32).view(self.num_envs, 1)  # 确保维度为 (self.num_envs, 1)
        return self.yaw_errors
    
    def _choose_target(self, indices=None):
        indices = th.arange(self.num_envs) if indices is None else indices
        rela_poses = self.position - th.as_tensor([6,0,1])
        for index in indices:
            if rela_poses[index][0] < 0:
                if rela_poses[index][1] > 0:
                    self._next_target_i[index] = 1
                else:
                    self._next_target_i[index] = 2
            if rela_poses[index][0] > 0:
                if rela_poses[index][1] > 0:
                    self._next_target_i[index] = 5
                else:
                    self._next_target_i[index] = 4
                    
    def get_reward(self) -> th.Tensor:
        lambda1 = 0.6
        lambda2 = 0.0025
        lambda3 = 0.0005
        lambda4 = 0.0002
        lambda5 = 0.001
        lambda6 = 0.01
        
        _next_target_i_clamp = self._next_target_i.clamp_max(len(self.targets) - 1)
        r_prog1 = lambda1 * ((self.last_position - self.targets[_next_target_i_clamp]).norm(dim=1)-(self.position - self.targets[_next_target_i_clamp]).norm(dim=1))
        # r_prog2 = self._success * (self.max_episode_steps - self._step_count) * 1 / ((self.velocity-0).norm()+1)
        # r_perc = th.tensor(-lambda2 * np.exp(-np.power(self.compute_yaw_error(_next_target_i_clamp),4)))
        r_ori = -lambda2 * (self.orientation-self.orientations[_next_target_i_clamp]).norm(dim=1)
        # r_success = 10.0 * self.get_success() # no contribution to the reward
        r_cmd = -lambda3 * (self._action - 0).norm(dim=1) - lambda4 * (self._action - self.last_action).norm(dim=1)
        r_crash = -4.0  * self.is_collision
        r_v = -lambda6 * (self.velocity - 0).norm(dim=1) 
        r_col_avoid = -lambda5 * 1 / (self.collision_dis + 0.2) 
        # + (1-self.collision_dis ).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6+self.collision_dis)).relu() * -lambda6
        # r_pass = (1.0 -(self.position - self.targets[_next_target_i_clamp]).norm(dim=1))* self.is_pass_next
        r_pass = 5.0 * self.is_pass_next
        reward = r_prog1 + r_crash  + r_pass + r_cmd + r_v + r_col_avoid + r_ori
        
        return reward  


    # def get_reward(self) -> th.Tensor:
    #         base_r = 0.1
    #         pos_factor = -0.1 * 1 / 9
    #         self.success_r = 20
    #         _next_target_i_clamp = self._next_target_i
    #         reward = (
    #                 base_r +
    #                 (self.position - self.targets[_next_target_i_clamp]).norm(dim=1) * pos_factor +
    #                 # (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
    #                 (self.velocity - 0).norm(dim=1) * -0.002 +
    #                 (self.angular_velocity - 0).norm(dim=1) * -0.002 +
    #                 self.is_pass_next * self.success_r
    #                 # self.success * self.success_r 
    #                 # -0.001 * 1 / (self.collision_dis + 0.2) 
    #                 # + (1-self.collision_dis ).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6+self.collision_dis)).relu() * -0.0005
    #         )
    #         return reward

class RacingEnv2(RacingEnv):

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
            latent_dim=latent_dim
        )
    
    def get_observation(
            self,
            indices=None
    ) -> Dict:
        # self.get_depth_image()
        # self.total_timesteps += 1

        # 检查时间步长是否达到要求
        # if self.total_timesteps % self.target_update_interval == 0:
        #     # self.reset_by_id(indices=self.indice)
        #     self.reset()

        _next_targets_i_clamp = th.stack([self._next_target_i + i for i in range(self._next_target_num)]).T % len(self.targets)
        next_targets = self.targets[_next_targets_i_clamp]
        # relative_pos = (next_targets - self.position.unsqueeze(1)).reshape(self.num_envs, -1)
        
        relative_pos_world = (next_targets - self.position.unsqueeze(1))
        relative_pos_body = self.world_to_body(relative_pos_world)
        relative_pos = relative_pos_body.reshape(self.num_envs, -1)
        
        self.previous_position.append(self.position.clone())
        self.previous_actions.append(self._action.clone())
        
        if len(self.previous_position) > 1:
            self.last_position= self.previous_position[-2]
        if len(self.previous_actions) > 2:
            self.pastAction = th.cat(list(self.previous_actions)[:3], dim=-1)
            self.last_action = self.previous_actions[-2] #倒数第二个应该才是上一步的动作
        
        state = th.hstack([
            relative_pos / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        if not self.requires_grad:
            if self.visual:
                return TensorDict({
                    "index": self._next_target_i.clone().detach().cpu().numpy().reshape(-1, 1),
                    "state": state,
                    # "noise_state": noise_state,
                    # "pastAction": self.pastAction.cpu().numpy(),
                    # "noise_target": relative_pos.cpu().numpy(),
                    "vd": self.v_d.unsqueeze(1).cpu().numpy(),
                    "depth": self.sensor_obs["depth"],
                    # "latent": self.latent.cpu().numpy(),
                    # "mask": (self.image/255.0).astype(np.float32),
                })
            else:
                return TensorDict({
                    "index": self._next_target_i.clone().detach().cpu().numpy().reshape(-1, 1),
                    "state": state,
                    "vd": self.v_d.unsqueeze(1).cpu().numpy(),
                    # "noise_state": noise_state,
                    # "latent": self.latent.cpu().numpy(),
                    # "noise_target": relative_pos.cpu().numpy(),
                    # "pastAction": self.pastAction.cpu().numpy()
                })

