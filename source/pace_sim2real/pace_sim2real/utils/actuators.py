# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

"""Access to the PACE actuator groups of an articulation, with tensors in PACE's ``joint_order``."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
from isaaclab.actuators import ActuatorBase

if TYPE_CHECKING:
    from isaaclab_newton.assets.articulation import Articulation as NewtonArticulation
    from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation


def actuator_groups(
    articulation: PhysxArticulation | NewtonArticulation, joint_ids: torch.Tensor
) -> Iterator[tuple[ActuatorBase, torch.Tensor]]:
    """Yield each actuator with the ``joint_order`` column of each of its joints.

    ``joint_ids`` are the articulation joint indices of ``joint_order``.

    Example:
        >>> joint_order = ["FR_hip", "FR_knee", "FL_hip", "FL_knee"]
        >>> joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order])
        >>> joint_ids
        tensor([1, 3, 0, 2])
        >>> for actuator, columns in actuator_groups(articulation, joint_ids):
        ...     print(actuator.joint_names, columns)
        ['FL_hip', 'FR_hip'] tensor([2, 0])
        ['FL_knee', 'FR_knee'] tensor([3, 1])
    """
    for actuator in articulation.actuators.values():
        drive_indices = actuator.joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0)
        yield actuator, torch.argmax(comparison_matrix.int(), dim=0)


def write_pd_gains(
    articulation: PhysxArticulation | NewtonArticulation, kp: torch.Tensor, kd: torch.Tensor, joint_ids: torch.Tensor
) -> None:
    """Set the actuator-model PD gains, one value per joint in ``joint_order``, for all environments."""
    for actuator, columns in actuator_groups(articulation, joint_ids):
        actuator.stiffness[:] = kp[columns]
        actuator.damping[:] = kd[columns]


def write_envelope(
    articulation: PhysxArticulation | NewtonArticulation,
    c_tau: torch.Tensor,
    c_omega: torch.Tensor,
    joint_ids: torch.Tensor,
) -> None:
    """Set the motor constants of every :class:`PaceDCMotorEnvelope`, shape (num_envs, len(joint_order))."""
    for actuator, columns in actuator_groups(articulation, joint_ids):
        if hasattr(actuator, "update_envelope"):
            actuator.update_envelope(c_tau[:, columns], c_omega[:, columns])


def read_delayed_effort(articulation: PhysxArticulation | NewtonArticulation, joint_ids: torch.Tensor) -> torch.Tensor:
    """Torque applied to the joints in the last step (after clip and delay), shape (num_envs, len(joint_order))."""
    effort = torch.zeros((articulation.num_instances, joint_ids.shape[0]), device=joint_ids.device)
    for actuator, columns in actuator_groups(articulation, joint_ids):
        effort[:, columns] = actuator.delayed_effort
    return effort


def read_clip_active(articulation: PhysxArticulation | NewtonArticulation, joint_ids: torch.Tensor) -> torch.Tensor:
    """Whether the actuator clip changed the PD torque in the last step, shape (num_envs, len(joint_order))."""
    active = torch.zeros((articulation.num_instances, joint_ids.shape[0]), dtype=torch.bool, device=joint_ids.device)
    for actuator, columns in actuator_groups(articulation, joint_ids):
        active[:, columns] = actuator.computed_effort != actuator.applied_effort
    return active


def read_clip_at_effort_limit(
    articulation: PhysxArticulation | NewtonArticulation, joint_ids: torch.Tensor
) -> torch.Tensor:
    """Whether the actuator clip held the torque at its effort limit in the last step.

    With :func:`read_clip_active`, this splits the clipped samples into those at the flat effort limit and
    those on a speed-dependent limit, such as a DC motor's torque-speed line. Shape (num_envs, len(joint_order)).
    """
    at_limit = torch.zeros((articulation.num_instances, joint_ids.shape[0]), dtype=torch.bool, device=joint_ids.device)
    for actuator, columns in actuator_groups(articulation, joint_ids):
        clipped = actuator.computed_effort != actuator.applied_effort
        at_limit[:, columns] = clipped & (actuator.applied_effort.abs() == actuator.actuator_effort_limit)
    return at_limit
