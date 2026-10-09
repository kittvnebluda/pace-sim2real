# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

# import math
from dataclasses import MISSING
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
# from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
# from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from . import mdp


##
# Scene definition
##


@configclass
class PaceSim2realSceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    env_spacing: float = 2.5
    num_envs: int = 4096

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False)  # actions = absolute joint position targets


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=False)


@configclass
class PacePhysicsCfg(PresetCfg):
    """Physics backends, selected on the command line with ``physics=<name>``."""
    isaacsim_physx = PhysxCfg()
    newton_mjwarp = NewtonCfg(solver_cfg=MJWarpSolverCfg(integrator="implicitfast"))  # same integrator as mjlab
    default = isaacsim_physx


@configclass
class CMAESOptimizerCfg:
    """CMA-ES optimizer configuration."""
    max_iteration: int = 200
    epsilon: float = 1e-2
    sigma: float = 0.5
    save_interval: int = 10
    save_optimization_process: bool = False  # consume more disk space if True, saves optimization process after finishing


@configclass
class ChirpCfg:
    """Excitation trajectory per joint (in ``joint_order``): target = center + amplitude * chirp, chirp in [-1, 1]."""
    center: list[float] = MISSING  # [rad]
    amplitude: list[float] = MISSING  # [rad], sign sets the direction


@configclass
class StepCfg:
    """Step excitation per joint (in ``joint_order``), for identifying the torque-speed envelope.

    Every ``hold`` seconds the target jumps to center + sign * scale * amplitude; the sign alternates
    and the scale cycles through ``scales``, so the steps have several sizes and reach several speeds.
    """
    center: list[float] = MISSING  # [rad]
    amplitude: list[float] = MISSING  # [rad]
    scales: list[float] = [1.0, 0.4, 0.8, 0.6]
    hold: float = 0.5  # [s]
    kp: float = 60.0  # P gain during the steps [Nm/rad]
    kd: float = 2.0  # D gain during the steps [Nm s/rad]


@configclass
class SyntheticParamsCfg:
    """Ground-truth parameters for synthetic data collection, applied to every joint."""
    armature: float = 0.1  # [kg m^2]
    viscous_friction: float = 4.5  # [Nm s/rad]
    coulomb_friction: float = 0.05  # [Nm]
    encoder_bias: float = 0.05  # [rad]
    delay: int = 5  # [sim steps]


@configclass
class PaceCfg:
    """Overall configuration for Pace Sim2Real task."""
    cmaes: CMAESOptimizerCfg = CMAESOptimizerCfg()
    chirp: ChirpCfg = ChirpCfg()
    synthetic: SyntheticParamsCfg = SyntheticParamsCfg()
    step: StepCfg | None = None  # step excitation, for identifying the torque-speed envelope

    robot_name: str = MISSING
    data_dir: str = MISSING
    joint_order: list = MISSING
    bounds_params: torch.Tensor = MISSING

##
# Environment configuration
##


@configclass
class PaceSim2realEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: PaceSim2realSceneCfg = PaceSim2realSceneCfg()
    # Basic settings
    actions: ActionsCfg = ActionsCfg()  # action = joint position targets (scale = 1.0 -> impedance control)

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    sim2real: PaceCfg = PaceCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 99999.0  # long episodes
        # viewer settings
        self.viewer.lookat = (0.0, 0.0, 0.8)
        self.viewer.eye = (2.0, 2.0, 1.5)
        # simulation settings
        self.sim.dt = 0.0025  # 400Hz simulation
        self.sim.render_interval = 4  # render at 100Hz
        # parse_env_cfg resolves the preset to one concrete PhysicsCfg; Isaac Lab types the field without presets
        self.sim.physics = PacePhysicsCfg()  # pyright: ignore[reportAttributeAccessIssue]
        # PaceDCMotor is a custom explicit actuator, so it needs the Python actuator path (Isaac Lab >= 3.0)
        if hasattr(self.sim, "use_newton_actuators"):
            self.sim.use_newton_actuators = False

        self.scene.robot.spawn.articulation_props.fix_root_link = True
