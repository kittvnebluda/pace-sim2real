# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.actuators import DCMotor, IdealPDActuator, resolve_joint_parameter
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils import DelayBuffer

if TYPE_CHECKING:
    from .pace_actuator_cfg import PaceDCMotorCfg, PaceDCMotorEnvelopeCfg, PaceIdealPDActuatorCfg


class _PaceActuatorMixin:
    """Encoder bias and action delay on top of an Isaac Lab PD actuator.

    The controller receives joint positions in the encoder frame by adding a per-joint encoder
    bias to the true joint positions. In other words, the controller operates on biased (encoder)
    positions rather than the true joint positions.

    The torque command computed by the PD controller is applied after a configurable delay
    (in simulation steps) to represent latency between command calculation and actuation.

    The software implementation is inspired by DelayedPDActuator.
    """

    def __init__(self, cfg: PaceDCMotorCfg | PaceIdealPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if isinstance(cfg.encoder_bias, (list, tuple)):
            if len(cfg.encoder_bias) != self.num_joints:
                raise ValueError(
                    f"encoder_bias must have {self.num_joints} elements (one per joint), "
                    f"but got {len(cfg.encoder_bias)}: {cfg.encoder_bias}"
                )
        self.encoder_bias = resolve_joint_parameter(cfg.encoder_bias, 0.0, self._joint_names, self._num_envs, self._device)

        self.torques_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(cfg.max_delay, torch.arange(self._num_envs, device=self._device))

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # reset buffers
        self.torques_delay_buffer.reset(env_ids)

    def update_encoder_bias(self, encoder_bias: torch.Tensor):
        self.encoder_bias = encoder_bias

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(delay, env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # compute actuator model with encoder bias added to joint positions (joint position in encoder frame, not simulation frame)
        control_action_sim = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        control_action_sim.joint_efforts = self.torques_delay_buffer.compute(control_action_sim.joint_efforts)
        # torque that reaches the joint; Isaac Lab's applied_effort holds the clipped torque before the delay
        self.delayed_effort = control_action_sim.joint_efforts
        return control_action_sim


class PaceDCMotor(_PaceActuatorMixin, DCMotor):
    """Pace DC Motor actuator model with encoder bias and action delay."""

    cfg: PaceDCMotorCfg


class PaceIdealPDActuator(_PaceActuatorMixin, IdealPDActuator):
    """Pace ideal PD actuator (flat effort limit) with encoder bias and action delay."""

    cfg: PaceIdealPDActuatorCfg


class PaceDCMotorEnvelope(PaceDCMotor):
    """Pace DC motor whose torque-speed envelope is set by the motor constants.

    With bus voltage V, torque constant c_tau [V/(N m)] and back-EMF constant c_omega [V s/rad],
    the envelope is upper = min((V - c_omega * qd) / c_tau, effort limit) and
    lower = max((-V - c_omega * qd) / c_tau, -effort limit). This is :class:`PaceDCMotor` with
    saturation_effort = V / c_tau and actuator_velocity_limit = V / c_omega, so the clip, the
    encoder bias and the delay are unchanged; only the parameterization differs.
    """

    cfg: PaceDCMotorEnvelopeCfg

    def __init__(self, cfg: PaceDCMotorEnvelopeCfg, *args, **kwargs):
        super().__init__(
            cfg.replace(
                saturation_effort=_divide(cfg.v_bus, cfg.c_tau),
                actuator_velocity_limit=_divide(cfg.v_bus, cfg.c_omega),
            ),
            *args,
            **kwargs,
        )

    def update_envelope(self, c_tau: torch.Tensor, c_omega: torch.Tensor):
        """Set the motor constants, shape (num_envs, num_joints)."""
        self._saturation_effort = self.cfg.v_bus / c_tau
        self.actuator_velocity_limit = self.cfg.v_bus / c_omega
        # DCMotor caches the speed where the envelope meets the effort limit at construction
        self._vel_at_effort_lim = self.actuator_velocity_limit * (1 + self.actuator_effort_limit / self._saturation_effort)


def _divide(numerator: float, value: dict[str, float] | float) -> dict[str, float] | float:
    if isinstance(value, dict):
        return {name: numerator / v for name, v in value.items()}
    return numerator / value
