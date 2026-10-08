# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab_newton.assets.articulation import Articulation as NewtonArticulation
    from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation


def write_joint_properties_to_sim(
    articulation: PhysxArticulation | NewtonArticulation,
    armature: torch.Tensor,
    viscous_friction: torch.Tensor,
    coulomb_friction: torch.Tensor,
    joint_ids: torch.Tensor,
    env_ids: torch.Tensor,
) -> None:
    """Write the PACE joint properties to the simulation for any Isaac Lab physics backend.

    Coulomb friction is modeled with static friction equal to dynamic friction. PhysX has both
    properties; Newton (MuJoCo Warp) has a single friction loss and no dynamic friction.
    """
    articulation.write_joint_armature_to_sim_index(armature=armature, joint_ids=joint_ids, env_ids=env_ids)
    articulation.write_joint_viscous_friction_coefficient_to_sim_index(
        joint_viscous_friction_coeff=viscous_friction, joint_ids=joint_ids, env_ids=env_ids
    )
    if hasattr(articulation, "write_joint_dynamic_friction_coefficient_to_sim_index"):
        # static and dynamic friction are written together, so static is never below dynamic
        articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=coulomb_friction,
            joint_dynamic_friction_coeff=coulomb_friction,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )
    else:
        articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=coulomb_friction, joint_ids=joint_ids, env_ids=env_ids
        )


def write_joint_state_to_sim(
    articulation: PhysxArticulation | NewtonArticulation,
    position: torch.Tensor,
    joint_ids: torch.Tensor | None,
) -> None:
    """Write joint positions and zero joint velocities to the simulation for all environments."""
    articulation.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids)
    articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(position), joint_ids=joint_ids)
