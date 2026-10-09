# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations
import torch
from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg
from pace_sim2real.utils import pace_actuator


@configclass
class PaceDCMotorCfg(DCMotorCfg):
    """Configuration for Pace DC Motor actuator model.

    This class extends the base DCMotorCfg with Pace-specific parameters.
    """
    class_type: type = pace_actuator.PaceDCMotor
    encoder_bias: dict[str, float] | float | None = 0.0
    max_delay: torch.int | None = 0


@configclass
class PaceIdealPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for Pace ideal PD actuator model.

    This class extends the base IdealPDActuatorCfg with Pace-specific parameters.
    """
    class_type: type = pace_actuator.PaceIdealPDActuator
    encoder_bias: dict[str, float] | float | None = 0.0
    max_delay: torch.int | None = 0


@configclass
class PaceDCMotorEnvelopeCfg(PaceDCMotorCfg):
    """Configuration for Pace DC motor with the envelope set by the motor constants.

    ``saturation_effort`` and ``actuator_velocity_limit`` are derived from ``v_bus``, ``c_tau``
    and ``c_omega`` by the actuator and must not be set; ``actuator_effort_limit`` is the peak torque.
    """
    class_type: type = pace_actuator.PaceDCMotorEnvelope
    saturation_effort: dict[str, float] | float | None = None  # derived: v_bus / c_tau
    v_bus: float = 27.0  # bus voltage [V]
    c_tau: dict[str, float] | float = MISSING  # torque constant at the joint [V/(N m)]
    c_omega: dict[str, float] | float = MISSING  # back-EMF constant at the joint [V s/rad]
