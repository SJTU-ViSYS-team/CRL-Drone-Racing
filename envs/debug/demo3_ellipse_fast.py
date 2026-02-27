import numpy as np
from envs.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from collections import deque
from utils.type import TensorDict
from scipy.spatial.transform import Rotation as R

PI = 3.14159
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
                                        {"position": {"mean": [2., 3., 1], "half": [.3, .3, 0.3]},
                                         "orientation": {"mean": [0., 0., 0.], "half": [0., 0., 0.]}},

                                },
                                # {
                                #     "class": "Uniform",
                                #     "kwargs":
                                #         {"position": {"mean": [10., 3., 1.], "half": [.2, .2, 0.2]},
                                #         "orientation": {"mean": [0., 0., -PI/2], "half": [0., 0., 0.]}},

                                # },
                                # {
                                #     "class": "Uniform",
                                #     "kwargs":
                                #         {"position": {"mean": [10., -3., 1], "half": [.2, .2, 0.2]},
                                #         "orientation": {"mean": [0., 0., -PI+0.01], "half": [0., 0., 0.]}},

                                # },
                                # {
                                #     "class": "Uniform",
                                #     "kwargs":
                                #         {"position": {"mean": [2., -3., 1], "half": [.2, .2, 0.2]},
                                #         "orientation": {"mean": [0., 0., PI /2], "half": [0., 0., 0.]}},

                                # },
                            ]
                        }
                    ]

                }
        }

        sensor_kwargs = [{
            "sensor_type": SensorType.DEPTH,
            "uuid": "depth",
            "position": [0.0, 0.0, -0.05],
            "resolution": [64, 64],
        }]
        # sensor_kwargs = []
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
            [4, 3, 1],    # 第一个门
            [8, 3, 1],    # 第二个门
            [10, 0, 1],   # 第三个门
            [8, -3, 1],   # 第四个门
            [4, -3, 1],   # 第五个门
            [2, 0, 1],  # 第六个门
        ])
        
        self.length_target = len(self.targets)
        self._next_target_num = 2
        self._next_target_i = th.zeros((self.num_envs,), dtype=th.int)
        self._past_targets_num = th.zeros((self.num_envs,), dtype=th.int)
        self._is_pass_next = th.zeros((self.num_envs,), dtype=th.bool)
        self.success_radius = 0.3
        
        self.v_d = 0.5 *th.ones((self.num_envs,),dtype=th.float)
        self.angle_diff = th.zeros((self.num_envs,),dtype=th.float)
        
        
        self.total_timesteps = 0
        self.target_update_interval = 500
        
        # self.observation_space["vd"] = spaces.Box(
        #     low=0.,
        #     high=30.,
        #     shape=(1,),
        #     dtype=np.float32
        # )
        
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
        # 输入四元数顺序为 (w, x, y, z)，需转为 (x, y, z, w)
        quat = self.orientation.cpu().numpy()
        quat_xyzw = quat[:, [1, 2, 3, 0]]  # 从 (w, x, y, z) 转为 (x, y, z, w)
        rotation = R.from_quat(quat_xyzw)
        rotation_matrix = th.tensor(rotation.as_matrix(), dtype=th.float32, device=self.device)
        
        # 世界到机体系：应用旋转矩阵转置
        relative_pos_body = th.einsum('bij,bjk->bik', rotation_matrix.transpose(1, 2), relative_pos_world.transpose(1, 2)).transpose(1, 2)
        return relative_pos_body
    
    def get_current_yaw(self):
        # 输入四元数顺序为 (w, x, y, z)，需转为 (x, y, z, w)
        quat = self.orientation[:, [1, 2, 3, 0]]  # 转为 (x, y, z, w)
        rotations = R.from_quat(quat.cpu().numpy())
        yaw = rotations.as_euler('zyx')[:, 0]  # 提取绕z轴的偏航角
        return th.from_numpy(yaw).to(self.device)
    
    
    def _choose_target(self, indices=None):
        indices = th.arange(self.num_envs) if indices is None else indices
        rela_poses = self.position - th.as_tensor([6,0,1])
        for index in indices:
            if rela_poses[index][0] < 0:
                if rela_poses[index][1] > 0:
                    self._next_target_i[index] = 0
                else:
                    self._next_target_i[index] = 5
            if rela_poses[index][0] > 0:
                if rela_poses[index][1] > 0:
                    self._next_target_i[index] = 2
                else:
                    self._next_target_i[index] = 3
                    
    # def get_reward(self) -> th.Tensor:
    #     lambda1 = 1.0
    #     lambda2 = 0.001
    #     lambda3 = 0.0025
    #     lambda4 = 0.0002
    #     lambda5 = 0.001
    #     # lambda6 = 0.0005
    #     lambda6 = 0.01
        
    #     _next_target_i_clamp = self._next_target_i.clamp_max(len(self.targets) - 1)
    #     r_prog1 = lambda1 * ((self.last_position - self.targets[_next_target_i_clamp]).norm(dim=1)-(self.position - self.targets[_next_target_i_clamp]).norm(dim=1))
    #     # r_prog2 = self._success * (self.max_episode_steps - self._step_count) * 1 / ((self.velocity-0).norm()+1)
    #     # r_perc = th.tensor(-lambda2 * np.exp(-np.power(self.compute_yaw_error(_next_target_i_clamp),4)))
    #     r_ori = -lambda2 * (self.orientation-self.orientations[_next_target_i_clamp]).norm(dim=1)
    #     # r_success = 10.0 * self.get_success() # no contribution to the reward
    #     r_cmd = -lambda3 * (self._action - 0).norm(dim=1) - lambda4 * (self._action - self.last_action).norm(dim=1)
    #     r_crash = -4.0  * self.is_collision
    #     r_v = -lambda6 * (self.velocity - 0).norm(dim=1) 
    #     r_col_avoid = -lambda5 * 1 / (self.collision_dis + 0.2) 
    #     # + (1-self.collision_dis ).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6+self.collision_dis)).relu() * -lambda6
    #     # r_pass = (1.0 -(self.position - self.targets[_next_target_i_clamp]).norm(dim=1))* self.is_pass_next
    #     r_pass = 5.0 * self.is_pass_next
    #     reward = r_prog1 + r_crash  + r_pass + r_cmd + r_ori + r_col_avoid
        
    #     return reward  


    def get_reward(self) -> th.Tensor:
        lambda1 = 0.9
        lambda2 = 0.05        
        lambda3 = 0.0005
        lambda4 = 0.0002
        lambda5 = 0.02
        lambda6 = 0.01
        lambda7 = 0.02
        # lambda6 = 0.01
        # self.yaw = th.atan2(2 * (self.orientation[:,0] * self.orientation[:,3] + self.orientation[:,1] * self.orientation[:,2]), 1 - 2 * (self.orientation[:,2]**2 + self.orientation[:,3]**2))
        # self.yaw = th.atleast_1d(th.atan2(2 * (self.orientation[:,0] * self.orientation[:,1] + self.orientation[:,2] * self.orientation[:,3]), 1 - 2 * (self.orientation[:,1] ** 2 + self.orientation[:,3] ** 2)))
        _next_target_i_clamp = self._next_target_i.clamp_max(len(self.targets) - 1)
        r_prog1 = lambda1 * ((self.last_position - self.targets[_next_target_i_clamp]).norm(dim=1)-(self.position - self.targets[_next_target_i_clamp]).norm(dim=1))
        # r_ori = -lambda2 * (self.orientation-self.orientations[_next_target_i_clamp]).norm(dim=1)
        # r_prog2 = self._success * (self.max_episode_steps - self._step_count) * 1 / ((self.velocity-0).norm()+1)
        # r_perc = th.tensor(lambda2 * np.exp(-np.power(self.compute_yaw_error(_next_target_i_clamp),2)))
        # r_perc = lambda2 * th.exp(-(self.yaw - self.yaw_target[_next_target_i_clamp]).norm(dim=1))
        # r_perc = lambda2 * th.exp(-self.theta**4)
        r_perc = lambda2 * th.exp(-(self.angle_diff).unsqueeze(1).norm(dim=1))
        r_cmd = -lambda3 * (self._action - 0).norm(dim=1) - lambda4 * (self._action - self.last_action).norm(dim=1)
        r_crash = -4.0  * self.is_collision
        r_anglev = -lambda5 * (self.angular_velocity - 0).norm(dim=1)
        r_v = -lambda7 * (self.velocity - 0).norm(dim=1) 
        # r_vel = -lambda6 * ((self.velocity).norm(dim=1)-self.v_d).abs()
        r_col_avoid = -lambda6 * 1 / (self.collision_dis + 0.2) 
        # + (1-self.collision_dis ).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6+self.collision_dis)).relu() * -lambda6
        # r_pass = (1.0 -(self.position - self.targets[_next_target_i_clamp]).norm(dim=1))* self.is_pass_next
        r_pass = 5.0 * self.is_pass_next
        # r_success = 10.0 * self.get_success()
        reward = r_prog1 + r_pass + r_perc + r_cmd + r_crash + r_anglev + r_col_avoid + r_v
        return reward 

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
        
        direction_vector = next_targets[:,0,:2]-self.position[:,:2]  # [x, y] 平面
        theta_target = th.atan2(direction_vector[:, 1], direction_vector[:, 0])  # 弧度
        theta_yaw = self.get_current_yaw()
        
        angle_diff = th.abs(theta_yaw - theta_target)
        self.angle_diff = th.min(angle_diff, 2 * PI - angle_diff)  # 确保范围[0, π]
        
        # 添加噪声
        # noise = th.randn_like(relative_pos) * 0.1  # 生成与 relative_pos 形状相同的噪声，标准差为 0.1
        # relative_pos_noisy = relative_pos + noise
        # current_target_body = relative_pos_world[:, 0, :]
        # dot_product = current_target_body[:, 0]
        # norms = th.norm(current_target_body, dim=1) + 1e-6
        # cos_theta = dot_product / norms
        # self.theta = th.acos(cos_theta)
        
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
                    # "vd": self.v_d.unsqueeze(1).cpu().numpy(),
                    "depth": self.sensor_obs["depth"],
                    # "latent": self.latent.cpu().numpy(),
                    # "mask": (self.image/255.0).astype(np.float32),
                })
            else:
                return TensorDict({
                    "index": self._next_target_i.clone().detach().cpu().numpy().reshape(-1, 1),
                    "state": state,
                    # "vd": self.v_d.unsqueeze(1).cpu().numpy(),
                    # "noise_state": noise_state,
                    # "latent": self.latent.cpu().numpy(),
                    # "noise_target": relative_pos.cpu().numpy(),
                    # "pastAction": self.pastAction.cpu().numpy()
                })

