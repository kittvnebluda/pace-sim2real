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
