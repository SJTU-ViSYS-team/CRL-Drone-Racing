import numpy as np
from envs.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
import cv2
from habitat_sim import SensorType
from gymnasium import spaces
from collections import deque
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
                                        {"position": {"mean": [1., 0., 1], "half": [0.3, 0.3, 0.3]}},

                                },
                                # {
                                #     "class": "Uniform",
                                #     "kwargs":
                                #         {"position": {"mean": [5., 0., 1], "half": [0.3, 0.3, 0.3]}},
                                # },
                                # {
                                #     "class": "Uniform",
                                #     "kwargs":
                                #         {"position": {"mean": [9., 0., 1], "half": [.2, .2, 0.2]}},
                                # },
                                # {
                                #     "class": "Uniform",
                                #     "kwargs":
                                #         {"position": {"mean": [12., 0., 1], "half": [.2, .2, 0.2]}},
                                # },
                            ]
                        }
                    ]
                }
        }
        sensor_kwargs = [{
            "sensor_type": SensorType.DEPTH,
            "uuid": "depth",
            "position": [0.0, 0.0, -0.1],
            "resolution": [64, 64],
        },]
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

        self.depth_tensor = th.empty((self.num_envs, self.num_envs, 64, 64), 
                                     dtype=th.float32, 
                                     device=self.device)
        self.noise_buffer = th.empty((self.num_envs, self.num_envs, 64, 64), 
                                     dtype=th.float32, 
                                     device=self.device)
        
        # pastactions and targets
        self.previous_position = deque(maxlen=2)  # 初始化上一步位32
        self.pastAction = th.zeros((self.num_envs, 12))  # 初始化过去动作
        self.previous_actions = deque(maxlen=4)  # 初始化动作队列
        self.last_action = th.zeros((self.num_envs, 4)) 
        self.last_position = th.zeros((self.num_envs, 3))
        self.v_d = 1*th.ones((self.num_envs,),dtype=th.float)
        
        self.first_frame_saved = False  # 添加一个标志变量                

        self.targets = th.as_tensor([
            [3, 0, 1],    
            [7, -1, 1],    
            [11, 1, 1],   
            [15, 0, 1],   
            [16, 0, 1],   
        ])

        self.yaw_errors = th.zeros((self.num_envs, 1),dtype=float)
        
        self.length_target = len(self.targets)
        self._next_target_num = 2
        self._next_target_i = th.zeros((self.num_envs,), dtype=th.int)
        self._past_targets_num = th.zeros((self.num_envs,), dtype=th.int)
        self._is_pass_next = th.zeros((self.num_envs,), dtype=th.bool)
        self.success_radius = 0.25
        
        self.total_timesteps = 0
        self.target_update_interval = 500
                
        self.observation_space["vd"] = spaces.Box(
            low=0.,
            high=30.,
            shape=(1,),
            dtype=np.float32
        )

        # add gate index to observation space
        self.observation_space["index"] = spaces.Box(
            low=0,
            high=self.length_target,
            shape=(1,),
            dtype=np.float32,
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
                    "index": self._next_target_i.cpu().numpy(),
                    "vd": self.v_d.cpu().numpy(),
                    # "pastAction": self.pastAction.cpu().numpy(),
                    # "noise_target" : self.noise_target.cpu().numpy(),
                    "latent": self.latent.cpu().numpy(),
                    "depth": self.sensor_obs["depth"],
                })
            else:
                obs = TensorDict({
                    "state": self.state.cpu().numpy(),
                    "index": self._next_target_i.cpu().numpy(),
                    "vd": self.v_d.cpu().numpy(),
                    # "noise_target" : self.noise_target.cpu().numpy(),
                    "latent": self.latent.cpu().numpy(),
                    # "pastAction": self.pastAction.cpu().numpy()
                })
        else:
            if self.visual:
                obs = TensorDict({
                    "state": self.state,
                    "index": self._next_target_i,
                    "vd": self.v_d,
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
        # self._next_target_i = self._next_target_i % len(self.targets)
        self._past_targets_num = self._past_targets_num + self._is_pass_next
        return self._next_target_i == len(self.targets)-1

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
                    
    def _choose_target(self, indices=None):
        indices = th.arange(self.num_envs) if indices is None else indices
        rela_poses = self.position - th.as_tensor([6,0,1])
        for index in indices:
            if rela_poses[index][0] < -3:
                self._next_target_i[index] = 0
            elif rela_poses[index][0] < 0:
                self._next_target_i[index] = 1
            elif rela_poses[index][0] < 4:
                self._next_target_i[index] = 2
            elif rela_poses[index][0] < 8:
                self._next_target_i[index] = 3
                
    def get_reward(self) -> th.Tensor:
        # lambda1 = 0.8
        lambda1 = 0.9
        lambda2 = 0.01
        lambda3 = 0.025
        lambda4 = 0.002
        lambda5 = 0.001
        # lambda6 = 0.0005
        lambda6 = 0.03
        lambda7 = 0.001
            
        _next_target_i_clamp = self._next_target_i.clamp_max(len(self.targets) - 1)
        target_pos = self.targets[_next_target_i_clamp]
        r_prog1 = lambda1 * ((self.last_position - self.targets[_next_target_i_clamp]).norm(dim=1)-(self.position - self.targets[_next_target_i_clamp]).norm(dim=1))
        r_ori = -lambda2 *  (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1)
        # r_prog2 = self._success * (self.max_episode_steps - self._step_count) * 1 / ((self.velocity-0).norm()+1)
        # r_perc = th.tensor(-lambda2 * np.exp(-np.power(self.compute_yaw_error(_next_target_i_clamp),4)))
        # r_success = 10.0 * self.get_success() # no contribution to the reward
        r_cmd = -lambda3 * (self._action - 0).norm(dim=1) - lambda4 * (self._action - self.last_action).norm(dim=1)
        r_crash = -4.0  * self.is_collision
        # r_v = -lambda7 * (self.velocity - 0).norm(dim=1) 
        r_vel = -lambda6 * ((self.velocity).norm(dim=1)-self.v_d).abs()
        r_col_avoid = -lambda5 * 1 / (self.collision_dis + 0.2) 
        
        # + (1-self.collision_dis ).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6+self.collision_dis)).relu() * -lambda6
        # r_pass = (1.0 -(self.position - self.targets[_next_target_i_clamp]).norm(dim=1))* self.is_pass_next
        r_pass = 6.0 * self.is_pass_next
        # r_vertical = -lambda7 * (self.position[:, 2] - target_pos[: ,2]).abs()
        # r_success = 10.0 * self.get_success()
        reward = r_prog1 + r_crash  + r_pass + r_cmd  + r_ori + r_col_avoid
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
    
    # def realsense_d435i_noise_model(self, depth_map):
    #     min_depth = 0.1
    #     max_depth = 10.0
        
    #     # 1. 深度值量化噪声 (16位转8位)
    #     depth_map_quantized = (depth_map * 1000).astype(np.uint16)  # 转换为毫米
    #     depth_map = depth_map_quantized.astype(np.float32) / 1000.0  # 转回米
        
    #     # 2. 高斯噪声 (与深度值成正比)
    #     depth_dependent_noise = depth_map * 0.01 * np.random.normal(0, 1, depth_map.shape)
    #     depth_map += depth_dependent_noise
        
    #     # 3. 边缘噪声 (边缘区域增加额外噪声)
    #     if depth_map.ndim == 2:
    #         # 检测边缘
    #         sobel_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    #         sobel_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    #         edge_mask = np.sqrt(sobel_x**2 + sobel_y**2) > 0.05
            
    #         # 在边缘添加额外噪声
    #         edge_noise = np.random.normal(0, 0.03, depth_map.shape)
    #         depth_map = np.where(edge_mask, depth_map + edge_noise, depth_map)
        
    #     # 4. 深度空洞噪声 (模拟传感器失效区域)
    #     hole_mask = np.random.random(depth_map.shape) < 0.005  # 0.5%的概率出现空洞
    #     depth_map[hole_mask] = 0.0
        
    #     # 5. 时间一致性噪声 (模拟帧间抖动)
    #     if hasattr(self, 'prev_depth'):
    #         temporal_noise = 0.7 * self.prev_depth + 0.3 * depth_map
    #         temporal_noise += np.random.normal(0, 0.005, depth_map.shape)
    #         depth_map = temporal_noise
    #     self.prev_depth = depth_map.copy()
        
    #     # 6. 近距离饱和噪声 (模拟近距离物体饱和)
    #     close_mask = depth_map < min_depth
    #     depth_map[close_mask] = min_depth + np.random.uniform(0, 0.01, size=np.sum(close_mask))
        
    #     # 7. 远距离精度下降 (模拟远距离精度降低)
    #     far_mask = depth_map > max_depth * 0.8
    #     depth_map[far_mask] += np.random.normal(0, 0.05, size=np.sum(far_mask))
        
    #     # 确保深度值在合理范围内
    #     depth_map = np.clip(depth_map, min_depth, max_depth)
        
    #     return depth_map
    

    # def visualize_depth(self, original_depth, noisy_depth):

    #     original_uint8 = (original_depth /np.max(original_depth) * 255).astype(np.uint8)
    #     noisy_uint8 = (noisy_depth /np.max(noisy_depth) * 255).astype(np.uint8)
    #     original_uint8 = np.squeeze(original_uint8)
    #     noisy_uint8 = np.squeeze(noisy_uint8)
        
    #     if original_uint8.ndim == 3:
    #         original_uint8 = original_uint8[0]
    #     if noisy_uint8.ndim == 3:
    #         noisy_uint8 = noisy_uint8[0]
        
    #     # 应用伪彩色增强以便更好地观察
    #     # original_color = cv2.applyColorMap(original_uint8, cv2.COLORMAP_JET)
    #     # noisy_color = cv2.applyColorMap(noisy_uint8, cv2.COLORMAP_JET)
        
    #     # 创建对比图像（左右并排）
    #     comparison = np.hstack((original_uint8, noisy_uint8))
        
    #     # # 添加标题
    #     # cv2.putText(comparison, "Original Depth", (10, 30), 
    #     #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #     # cv2.putText(comparison, "Noisy Depth", (original_uint8.shape[1] + 10, 30), 
    #     #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    #     # 显示图像
    #     cv2.imshow("Depth Visualization", comparison)
        
    #     # 等待按键（非阻塞方式）
    #     key = cv2.waitKey(1)
    #     if key == 27:  # ESC键
    #         cv2.destroyAllWindows()
    #         return False
    #     return True
    
    # real-world debug
    # def save_first_frame(self, noisy_depth):
    #     # 保存第一帧的 noisy_depth 数据
    #     save_path = "first_frame_noisy_depth.npy"
    #     np.save(save_path, noisy_depth)
    #     print(f"First frame noisy depth saved to {save_path}")
        
    def get_observation(
            self,
            indices=None
    ) -> Dict:

        self.total_timesteps += 1

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
                depth_data = self.sensor_obs["depth"]
                # # 使用预分配内存
                # depth_tensor = self.depth_tensor
                # noise_buffer = self.noise_buffer
                # depth_tensor.copy_(th.as_tensor(depth_data, device=self.device))
                
                # depth_tensor.clamp_(min=0.2, max=10.0)
                # depth_tensor.reciprocal_()  
                
                # # if self.step_count % 10 == 0:
                # noise_buffer.normal_(mean=0.0, std=0.03)
                
                # depth_tensor.add_(noise_buffer)
                
                # noisy_depth = depth_tensor.cpu().numpy()
                
                # noisy_depth = self.realsense_d435i_noise_model(original_depth)

                noisy_depth = 1.0 / th.as_tensor(depth_data).clamp(0.2, 10) + th.randn_like(th.as_tensor(depth_data)) * 0.03 
                
                # self.visualize_depth(original_depth, noisy_depth)
                # if not self.first_frame_saved:
                #     self.save_first_frame(noisy_depth)
                #     self.first_frame_saved = True
                    
                return TensorDict({
                    "index": self._next_target_i.clone().detach().cpu().numpy().reshape(-1, 1),
                    "state": state,
                    # "pastAction": self.pastAction.cpu().numpy(),
                    "vd": self.v_d.unsqueeze(1).cpu().numpy(),
                    "depth": noisy_depth,
                    # "latent": self.latent.cpu().numpy()
                })
            else:
                return TensorDict({
                    # "index": self._next_target_i.clone().detach().cpu().numpy().reshape(-1, 1),
                    "state": state,
                    "vd": self.v_d.unsqueeze(1).cpu().numpy()
                    # "latent": self.latent.cpu().numpy()
                    # "pastAction": self.pastAction.cpu().numpy()
                })
