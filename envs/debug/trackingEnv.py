
import numpy as np
from envs.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from utils.type import TensorDict
from utils.maths import Quaternion

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
            r_type=None,
    ):
        self.r_type = 0
        self.center = th.as_tensor([5, 0, 1])
        self.next_points_num = 10
        self.radius = 2
        self.radius_spd = 0.2 * th.pi / 1
        self.success_radius = 0.3
        self.height = 0.3

        random_kwargs = {
            "state_generator":
                {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [self.center[0], 0., self.center[2]],
                                      "half": [.2, .2, 0.2]}},
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
        )

        self.observation_space["state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 * (self.next_points_num - 1) + self.observation_space["state"].shape[0],),
            dtype=np.float32
        )
        self.dt = self.envs.dynamics.ctrl_dt
        self.target_ori = th.tensor([1,0,0,0])

        self.get_observation()

    def update_target(self):
        ts = self.t.repeat(self.next_points_num, 1).T + th.arange(self.next_points_num) * self.dt
        self.target = (th.stack([self.radius * th.cos(self.radius_spd * ts) + self.center[0],
                                 self.radius * th.sin(self.radius_spd * ts) + self.center[1],
                                 self.height * th.sin(self.radius_spd * ts) + self.center[2]
                                 ])
                       ).permute(1, 2, 0)
        self.target_ori = th.tensor([1,0,0,0])

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        self.update_target()
        # self.target = self.trajs[self.current_traj_index, target_index]
        diff_pos = self.target - self.position.unsqueeze(1)
        # consider target as next serveral waypoint
        diff_pos_flatten = diff_pos.reshape(self.num_envs, -1)

        state = th.hstack([
            diff_pos_flatten / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        return TensorDict({
            "state": state,
        })

    def get_success(self) -> th.Tensor:
        return th.full((self.num_agent,), False)
        # return (self.position - self.target).norm(dim=1) < self.success_radius

    def get_reward(self) -> th.Tensor:
        base_r = 0.1 * th.ones((self.num_envs,), dtype=th.float32)

        pos_factor = -0.1 * 1 / 9
        pos_r = (self.position - self.target[:, 0, :]).norm(dim=1) * pos_factor
        x_pos_r = (self.position[:, 0] - self.target[:, 0, 0]).abs() * pos_factor
        y_pos_r = (self.position[:, 1] - self.target[:, 0, 1]).abs() * pos_factor
        z_pos_r = (self.position[:, 2] - self.target[:, 0, 2]).abs() * pos_factor
        ori_r = (self.orientation - self.target_ori).norm(dim=1) * -0.00001
        vel_r = (self.velocity - 0).norm(dim=1) * -0.002
        ang_vel_r = (self.angular_velocity - 0).norm(dim=1) * -0.002
        # reward = (
        #         base_r +
        #         (self.position - self.target[:, 0, :]).norm(dim=1) * pos_factor +
        #         (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
        #         (self.velocity - 0).norm(dim=1) * -0.002 +
        #         (self.angular_velocity - 0).norm(dim=1) * -0.002
        # )
        if self.r_type == 0:
            diff_r = x_pos_r + y_pos_r + z_pos_r + ori_r + vel_r + ang_vel_r
            # diff_r = pos_r + ori_r + vel_r + ang_vel_r
            disc_r = 0 + base_r
        elif self.r_type == 1:
            diff_r = x_pos_r + y_pos_r + ori_r + vel_r + ang_vel_r
            disc_r = z_pos_r.detach() + base_r
        elif self.r_type == 2:
            diff_r = x_pos_r + ori_r + vel_r + ang_vel_r
            disc_r = y_pos_r.detach() + z_pos_r.detach() + base_r
        elif self.r_type == 3:
            diff_r = ori_r + vel_r + ang_vel_r
            disc_r = x_pos_r.detach() + y_pos_r.detach() + z_pos_r.detach() + base_r
        elif self.r_type == 4:
            diff_r = vel_r + ang_vel_r
            disc_r = x_pos_r.detach() + y_pos_r.detach() + z_pos_r.detach() + ori_r.detach() + base_r
        else:
            raise ValueError(f"Invalid reward type: {self.r_type}")
        reward = diff_r + disc_r
        # return {"reward":reward, "diff_r":diff_r, "disc_r":th.as_tensor(disc_r)}
        return reward


class TrackEnv2(TrackEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius_spd = self.radius_spd

    def update_target(self):
        ts = self.t.repeat(self.next_points_num, 1).T + th.arange(self.next_points_num) * self.dt
        self.target = (th.stack([2 * self.radius * th.cos(self.radius_spd * ts) + self.center[0],
                                 2 * self.radius * th.sin(self.radius_spd * ts) * th.cos(self.radius_spd * ts) + self.center[1],
                                 self.height * th.sin(self.radius_spd * ts) + self.center[2]
                                 ])
                       ).permute(1, 2, 0)

class AwareTrackEnv(TrackEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space["state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 * (self.next_points_num - 1) + 13+3,),
            dtype=np.float32
        )

    def get_reward(self) -> th.Tensor:
        base_r = 0.1 * th.ones((self.num_envs,), dtype=th.float32)
        target_vector = self.center - self.position
        normal_target_vector = target_vector / target_vector.norm(dim=1, keepdim=True)-0
        proj = ((self.direction.clone()-0) * normal_target_vector-0).sum(dim=1)
        # proj = (self.orientation-0).norm(dim=1) * 0.001  # Project the target vector onto the x-axis
        aware_r = proj * 0.05
        pos_factor = -0.1 * 1 / 9
        pos_r = (self.position - self.target[:, 0, :]).norm(dim=1) * pos_factor
        keep_pos_r = ((self.position - self.center).norm(dim=1)-1.0).abs() * pos_factor
        ori_r = (self.orientation - self.target_ori).norm(dim=1) * -0.00001
        vel_r = (self.velocity - 0).norm(dim=1) * -0.002
        acc_r = (self.envs.acceleration-0).norm(dim=1) * -0.001
        ang_acc_r = (self.envs.angular_acceleration-0).norm(dim=1) * -0.001
        ang_vel_r = (self.angular_velocity - 0).norm(dim=1) * -0.002

        diff_r = vel_r + ang_vel_r + aware_r + keep_pos_r #+ acc_r + ang_acc_r
        # diff_r = pos_r + ori_r + vel_r + ang_vel_r
        disc_r = th.zeros_like(diff_r)

        reward = diff_r + disc_r + base_r
        return {"reward":reward, "diff_r":diff_r, "disc_r":th.as_tensor(disc_r)}

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        self.update_target()
        # self.target = self.trajs[self.current_traj_index, target_index]

        diff_pos = self.target - self.position.unsqueeze(1)
        # consider target as next serveral waypoint
        diff_pos_flatten = diff_pos.reshape(self.num_envs, -1)
        rela_tar = self.center-self.position
        state = th.hstack([
            rela_tar,
            diff_pos_flatten / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        return TensorDict({
            "state": state,
        })


class AwareTrackEnv2(DroneGymEnvsBase):
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
            r_type=None,
    ):

        self.center = th.as_tensor([5, 0, 1.])
        self.radius = 2
        self.radius_spd = 0.2 * th.pi / 1
        self.success_radius = 0.3
        self.height = 0.3

        random_kwargs = {
            "state_generator":
                {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [self.center[0], 0., self.center[2]],
                                      "half": [1, 1, 0.2]}},
                    ]
                }
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

        self.update_target()

    def get_reward(self) -> th.Tensor:
        base_r = 0.1 * th.ones((self.num_envs,), dtype=th.float32)
        target_vector = self.target - self.position
        normal_target_vector = target_vector / target_vector.norm(dim=1, keepdim=True) - 0
        proj = ((self.direction.clone() - 0) * normal_target_vector - 0).sum(dim=1)
        aware_r = proj * 0.05
        pos_factor = -0.1 * 1 / 9
        pos_r = (self.position - self.target).norm(dim=1) * pos_factor
        keep_pos_r = ((self.position - self.target).norm(dim=1) - 1.0).abs() * pos_factor
        vel_r = (self.velocity - 0).norm(dim=1) * -0.002
        ang_vel_r = (self.angular_velocity - 0).norm(dim=1) * -0.002
        acc_r = (self.envs.acceleration - 0).norm(dim=1) * -0.001
        # ang_acc_r = (self.envs.angular_acceleration - 0).norm(dim=1) * -0.001

        diff_r = vel_r + ang_vel_r + aware_r + keep_pos_r  # + acc_r + ang_acc_r
        disc_r = th.zeros_like(diff_r)

        reward = diff_r + disc_r + base_r
        return reward

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        self.update_target()

        rela_tar = self.target - self.position
        # use local target
        orientation = self.envs.dynamics._orientation.clone()
        local_targets = orientation.inv_rotate(rela_tar.T).T
        state = th.hstack([
            local_targets / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        return TensorDict({
            "state": state,
        })

    def update_target(self):
        self.target = self.center
        self.target = th.stack([self.radius * th.cos(self.radius_spd * self.t) + self.center[0],
                                 self.radius * th.sin(self.radius_spd * self.t) + self.center[1],
                                 self.height * th.sin(self.radius_spd * self.t) + self.center[2]
                                 ]).T

    def  get_success(self) -> th.Tensor:
        return th.full((self.num_agent,), False)
