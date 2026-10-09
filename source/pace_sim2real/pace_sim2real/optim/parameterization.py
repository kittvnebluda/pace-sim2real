# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

"""Parameter layouts searched by :class:`CMAESOptimizer`, one per identification stage.

A layout owns the bounds, writes a population of physical parameter values to the simulator,
reports the encoder bias used in the position loss, and logs its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pace_sim2real.utils.actuators import actuator_groups
from pace_sim2real.utils.joint_properties import write_joint_properties_to_sim, write_joint_state_to_sim

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class PaceJointParams:
    """Stage 1, PACE: armature, viscous friction, Coulomb friction and encoder bias per joint, and one delay."""

    def __init__(self, bounds: torch.Tensor, joint_order: list[str]):
        self.bounds = bounds
        self.joint_order = joint_order
        num_joints = len(joint_order)
        self.armature_idx = slice(0, num_joints)
        self.damping_idx = slice(num_joints, 2 * num_joints)
        self.friction_idx = slice(2 * num_joints, 3 * num_joints)
        self.bias_idx = slice(3 * num_joints, 4 * num_joints)
        self.delay_idx = 4 * num_joints

    def encoder_bias(self, sim_params: torch.Tensor) -> torch.Tensor:
        return sim_params[:, self.bias_idx]

    def apply(self, articulation, joint_ids: torch.Tensor, initial_position: torch.Tensor, sim_params: torch.Tensor):
        env_ids = torch.arange(len(sim_params[:, self.armature_idx]), device=sim_params.device)
        write_joint_properties_to_sim(
            articulation,
            armature=sim_params[:, self.armature_idx],
            viscous_friction=sim_params[:, self.damping_idx],
            coulomb_friction=sim_params[:, self.friction_idx],
            joint_ids=joint_ids,
            env_ids=env_ids,
        )
        write_joint_state_to_sim(articulation, initial_position + sim_params[:, self.bias_idx], joint_ids)
        for actuator, drive_joint_idx in actuator_groups(articulation, joint_ids):
            actuator.update_encoder_bias(sim_params[:, self.bias_idx][:, drive_joint_idx])
            actuator.update_time_lags(sim_params[:, self.delay_idx].to(torch.int))
            actuator.reset(env_ids)

    def print_best(self, sim_params: torch.Tensor, index: int):
        print("Armature: ", sim_params[index, self.armature_idx].tolist())
        print("Viscous Friction: ", sim_params[index, self.damping_idx].tolist())
        print("Static/Dynamic Friction: ", sim_params[index, self.friction_idx].tolist())
        print("Bias: ", sim_params[index, self.bias_idx].tolist())
        print("Delay: ", sim_params[index, self.delay_idx].tolist())

    def log(self, writer: SummaryWriter, sim_params: torch.Tensor, best_index: int, iteration: int):
        for i in range(len(self.joint_order)):
            writer.add_histogram(
                "4_Bias/distribution_" + self.joint_order[i], sim_params[:, self.bias_idx][:, i], iteration
            )
            writer.add_histogram(
                "3_Static_Dynamic_Friction/distribution_" + self.joint_order[i],
                sim_params[:, self.friction_idx][:, i],
                iteration,
            )
            writer.add_histogram(
                "2_Viscous_Friction/distribution_" + self.joint_order[i],
                sim_params[:, self.damping_idx][:, i],
                iteration,
            )
            writer.add_histogram(
                "1_Armature/distribution_" + self.joint_order[i], sim_params[:, self.armature_idx][:, i], iteration
            )

            writer.add_scalar(
                "4_Bias/best_" + self.joint_order[i], sim_params[best_index, self.bias_idx][i].item(), iteration
            )
            writer.add_scalar(
                "3_Static_Dynamic_Friction/best_" + self.joint_order[i],
                sim_params[best_index, self.friction_idx][i].item(),
                iteration,
            )
            writer.add_scalar(
                "2_Viscous_Friction/best_" + self.joint_order[i],
                sim_params[best_index, self.damping_idx][i].item(),
                iteration,
            )
            writer.add_scalar(
                "1_Armature/best_" + self.joint_order[i], sim_params[best_index, self.armature_idx][i].item(), iteration
            )
        writer.add_histogram("0_Delay/distribution", sim_params[:, self.delay_idx], iteration)
        writer.add_scalar("0_Delay/best", sim_params[best_index, self.delay_idx].item(), iteration)
