from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder


@register_env("ResetButton-v1", max_episode_steps=100)
class ResetButtonEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    button_xy = (0.0, 0.0)
    pedestal_half = (0.06, 0.06, 0.035)
    cap_radius = 0.05
    cap_half_length = 0.015
    cap_rest_gap = 0.02
    travel = 0.02
    trigger_frac = 0.6
    spring_stiffness = 1000.0
    spring_damping = 20.0
    approach_weight = 5.0
    press_weight = 2.0
    success_bonus = 20.0
    action_penalty = 5e-4

    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        kwargs.setdefault("control_mode", "pd_ee_delta_pos")
        kwargs.setdefault("reward_mode", "dense")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.7, 0.7, 0.6], target=[0.0, 0.0, 0.15])
        return [CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.button = self._build_button()

    def _build_button(self):
        builder = self.scene.create_articulation_builder()
        builder.set_initial_pose(
            sapien.Pose(p=[self.button_xy[0], self.button_xy[1], 0.0])
        )
        base = builder.create_link_builder()
        base.set_name("base")
        base.add_box_visual(
            pose=sapien.Pose(p=[0, 0, self.pedestal_half[2]]),
            half_size=list(self.pedestal_half),
            material=sapien.render.RenderMaterial(base_color=[0.15, 0.15, 0.17, 1]),
        )
        base.add_box_collision(
            pose=sapien.Pose(p=[0, 0, self.pedestal_half[2]]),
            half_size=list(self.pedestal_half),
        )

        cap = builder.create_link_builder(base)
        cap.set_name("cap")
        cap.set_joint_name("button_joint")
        q = [0.7071068, 0.0, 0.7071068, 0.0]
        anchor_z = 2 * self.pedestal_half[2] + self.cap_rest_gap
        cap.set_joint_properties(
            type="prismatic",
            limits=[[0.0, self.travel]],
            pose_in_parent=sapien.Pose(p=[0, 0, anchor_z], q=q),
            pose_in_child=sapien.Pose(p=[0, 0, -self.cap_half_length], q=q),
            damping=self.spring_damping,
        )
        cap.add_cylinder_visual(
            pose=sapien.Pose(p=[0, 0, 0]),
            radius=self.cap_radius,
            half_length=self.cap_half_length,
            material=sapien.render.RenderMaterial(base_color=[0.85, 0.1, 0.1, 1]),
        )
        cap.add_cylinder_visual(
            pose=sapien.Pose(p=[0, 0, -self.cap_half_length - 0.01]),
            radius=0.02,
            half_length=0.02,
            material=sapien.render.RenderMaterial(base_color=[0.3, 0.3, 0.32, 1]),
        )
        cap.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0]),
            half_size=[self.cap_radius, self.cap_radius, self.cap_half_length],
        )

        button = builder.build(name="button", fix_root_link=True)
        for joint in button.get_joints():
            if "prismatic" in joint.type:
                joint.set_drive_properties(self.spring_stiffness, self.spring_damping)
                joint.set_drive_target(0.0)
                joint.set_drive_velocity_target(0.0)
        return button

    def _depression(self):
        qpos = self.button.get_qpos().reshape(self.num_envs, -1)
        return torch.clamp(qpos[:, 0], 0.0, self.travel)

    def _button_top(self):
        root_p = self.button.pose.p
        top_lift = (
            2 * self.pedestal_half[2]
            + self.cap_rest_gap
            + 2 * self.cap_half_length
            - self._depression()
        )
        return torch.stack(
            [root_p[:, 0], root_p[:, 1], root_p[:, 2] + top_lift], dim=1
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self.button.set_qpos(torch.zeros((len(env_idx), 1), device=self.device))
        dist = torch.linalg.norm(self._button_top() - self.agent.tcp_pose.p, axis=1)
        depression = self._depression()
        if not hasattr(self, "prev_dist"):
            self.prev_dist = dist.clone()
            self.prev_depression = depression.clone()
        else:
            self.prev_dist[env_idx] = dist[env_idx].clone()
            self.prev_depression[env_idx] = depression[env_idx].clone()

    def _get_obs_extra(self, info: dict):
        return dict(
            tcp_to_button=self._button_top() - self.agent.tcp_pose.p,
            button_depression=(self._depression() / self.travel).unsqueeze(-1),
        )

    def evaluate(self):
        depression = self._depression()
        return {
            "success": depression >= self.trigger_frac * self.travel,
            "button_depressed": depression > 1e-4,
            "depression": depression,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        dist = torch.linalg.norm(self._button_top() - self.agent.tcp_pose.p, axis=1)
        depression = self._depression()
        reward = self.approach_weight * (self.prev_dist - dist)
        reward += self.press_weight * (depression - self.prev_depression)
        reward += self.success_bonus * info["success"].float()
        if action is not None:
            reward -= self.action_penalty * (action**2).sum(dim=-1)
        self.prev_dist = dist.clone()
        self.prev_depression = depression.clone()
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs, action, info) / self.success_bonus
