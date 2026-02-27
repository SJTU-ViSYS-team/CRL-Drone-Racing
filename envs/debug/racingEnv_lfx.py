import numpy as np
from envs.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from collections import deque
from utils.type import TensorDict
from scipy.spatial.transform import Rotation as R
is_pos_reward = True


class RacingEnv(DroneGymEnvsBase):
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
                    "class": "Union",

                    "kwargs": [
                        {"randomizers_kwargs":
                            [
                                {
                                    "class": "Uniform",
                                    "kwargs":
                                        {"position": {"mean": [2., 3., 1], "half": [.3, .3, 0.3]}},

                                },
                                {
                                    "class": "Uniform",
                                    "kwargs":
                                        {"position": {"mean": [7., 2., 2], "half": [.2, .2, 0.2]}},

                                },
                                {
                                    "class": "Uniform",
                                    "kwargs":
                                        {"position": {"mean": [7., -3., 1], "half": [.2, .2, 0.2]}},

                                },
                                {
                                    "class": "Uniform",
                                    "kwargs":
                                        {"position": {"mean": [1., -3., 1], "half": [.2, .2, 0.2]}},

                                },
                            ]
                        }
                    ]

                }
        }
        sensor_kwargs = [{
            "sensor_type": SensorType.DEPTH,
            "uuid": "depth",
            "resolution": [64, 64],
        }]
        sensor_kwargs = []
        dynamics_kwargs = {
            "dt": 0.01,
            "ctrl_dt": 0.03,
            "action_type": "thrust",
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
        self.target = th.as_tensor([
            [4, 3, 1.],
            [7, 0, 2.],
            [4, -3, 1.],
            [1, 0, 1.],
        ]) 

        self._next_target_num = 2
        self._next_target_i = th.zeros((self.num_envs,), dtype=th.int)
        self._past_targets_num = th.zeros((self.num_envs,), dtype=th.int)
        self._is_pass_next = th.zeros((self.num_envs,), dtype=th.bool)
        self.success_radius = 0.4
        self.observation_space["index"] = spaces.Box(
            low=0,
            high=len(self.target),
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

        obs = TensorDict({
            "state": self.state,
            "index": self._next_target_i,
        })

        if self.latent is not None:
            if not self.requires_grad:
                obs["latent"] = self.latent.cpu().numpy()
            else:
                obs["latent"] = self.latent

        return obs

    def get_success(self) -> th.Tensor:
        _next_target_i_clamp = self._next_target_i
        self._is_pass_next = ((self.position - self.target[_next_target_i_clamp]).norm(dim=1) <= self.success_radius)
        self._next_target_i = self._next_target_i + self._is_pass_next
        self._next_target_i = self._next_target_i % len(self.target)
        self._past_targets_num = self._past_targets_num + self._is_pass_next
        return th.zeros((self.num_envs,), dtype=th.bool)

    def reset_by_id(self, indices=None, state=None, reset_obs=None):
        indices = th.arange(self.num_envs) if indices is None else indices
        if reset_obs is not None:
            self._next_target_i = reset_obs["index"].to(self.device).squeeze()
        else:
            self._choose_target(indices=indices)
            # self._next_target_i[indices] = th.zeros((len(indices),), dtype=th.int)

        self._past_targets_num[indices] = th.zeros((len(indices),), dtype=th.int)
        self._is_pass_next[indices] = th.zeros((len(indices),), dtype=th.bool)

        obs = super().reset_by_id(indices, state, reset_obs)

        return obs

    def reset(self, state=None, obs=None, is_test=False):
        obs = super().reset(state)
        self._next_target_i = th.zeros((self.num_envs,), dtype=th.int)
        # self._past_targets_num = th.zeros((self.num_envs,), dtype=th.int)
        self._choose_target()
        return obs

    def _choose_target(self, indices=None):
        indices = th.arange(self.num_envs) if indices is None else indices
        rela_poses = self.position - th.as_tensor([4,0,1])
        for index in indices:
            if rela_poses[index][0] < 0:
                if rela_poses[index][1] > 0:
                    self._next_target_i[index] = 0
                else:
                    self._next_target_i[index] = 3
            else:
                if rela_poses[index][0] > 0:
                    self._next_target_i[index] = 1
                else:
                    self._next_target_i[index] = 2

    def get_reward(self) -> th.Tensor:
        if not is_pos_reward:
            _next_target_i_clamp = self._next_target_i
            dis_vector = (self.target[_next_target_i_clamp] - self.position)
            dis = (dis_vector-0).norm(dim=1, keepdim=True)
            dis_vector_norm = dis_vector / (dis+1e-6)
            # v_norm = (self.velocity-0).norm(dim=1)
            dis_v_product = ((self.velocity-0) * dis_vector).sum(dim=1, keepdim=True)
            approaching_v = (dis_v_product / (dis+1e-6)).clamp_max(15.)
            approaching_v_vector = dis_vector_norm * approaching_v
            away_v_vector = self.velocity - approaching_v_vector
            away_v = (away_v_vector-0).norm(dim=1) * (1/ (dis.squeeze() + 1))
            reward = approaching_v.squeeze() * 0.02 - away_v * 0.02 + self.is_pass_next * self.success_r
            reward = reward + (self.angular_velocity - 0).norm(dim=1) * -0.001
            return reward
        else:
            base_r = 0.1
            pos_factor = -0.1 * 1 / 9
            self.success_r = 20
            _next_target_i_clamp = self._next_target_i
            reward = (
                    base_r +
                    (self.position - self.target[_next_target_i_clamp]).norm(dim=1) * pos_factor +
                    (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
                    (self.velocity - 0).norm(dim=1) * -0.002 +
                    (self.angular_velocity - 0).norm(dim=1) * -0.002 +
                    self.is_pass_next * self.success_r
            )
            return reward


class RacingEnv2(RacingEnv):

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
            tensor_output: bool = False,
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
        )

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        _next_targets_i_clamp = th.stack([self._next_target_i + i for i in range(self._next_target_num)]).T % len(self.target)
        next_targets = self.target[_next_targets_i_clamp]
        relative_pos = (next_targets - self.position.unsqueeze(1)).reshape(self.num_envs, -1)
        state = th.hstack([
            relative_pos / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        return TensorDict({
            "state": state,
            "index": self._next_target_i.unsqueeze(1).clone().detach(),
        })