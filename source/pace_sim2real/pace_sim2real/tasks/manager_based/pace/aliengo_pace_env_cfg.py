# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

import os

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg
from pace_sim2real.tasks.manager_based.pace.pace_sim2real_env_cfg import EnvelopeCfg, StepCfg
from pace_sim2real.utils import PaceDCMotorEnvelopeCfg
from pace_sim2real.utils.paths import ASSETS_DIR

# The same MJCF that will be used in mjlab training,
# so this task and the mjlab port use one robot model.
ALIENGO_MJCF = os.environ.get(
    "PACE_ALIENGO_MJCF",
    str(ASSETS_DIR / "unitree_aliengo" / "aliengo.xml"),
)

ALIENGO_PACE_ACTUATOR_CFG = PaceDCMotorEnvelopeCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    # peak torque per group, from the URDF
    actuator_effort_limit={".*_hip_joint": 35.278, ".*_thigh_joint": 35.278, ".*_calf_joint": 44.4},
    # motor constants at 27 V: the synthetic ground truth, fixed in stage 1 and fitted in stage 2
    v_bus=27.0,
    c_omega={".*_hip_joint": 1.15, ".*_thigh_joint": 0.80, ".*_calf_joint": 1.45},  # [V s/rad]
    c_tau={".*_hip_joint": 0.60, ".*_thigh_joint": 0.55, ".*_calf_joint": 0.25},  # [V/(N m)]
    stiffness={".*": 40.0},  # P gain in Nm/rad
    damping={".*": 2.0},  # D gain in Nm s/rad
    encoder_bias={".*": 0.0},  # encoder bias in radians
    max_delay=10,  # max delay in simulation steps
)


@configclass
class AliengoPaceCfg(PaceCfg):
    """Pace configuration for the Unitree Aliengo robot."""

    robot_name: str = "aliengo_sim"
    data_dir: str = "aliengo_sim/chirp_data.pt"  # located in pace_sim2real/data/aliengo_sim/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((49, 2))  # 12 + 12 + 12 + 12 + 1 = 49 parameters to optimize
    joint_order: list[str] = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]

    def __post_init__(self):
        # hip +-0.3, thigh 0.5..1.1, calf -2.1..-0.9 rad: inside the MJCF limits (hip +-1.22, calf -2.78..-0.65);
        # thigh amplitude 0.3, not 0.5: at 0.5 the 10 Hz end clipped ~10% of thigh samples on the envelope
        self.chirp.center = [0.0, 0.8, -1.5] * 4
        self.chirp.amplitude = [0.3, 0.3, 0.6] * 4
        # ground truth for synthetic data, different from the MJCF defaults (armature 0.01, frictionloss 0.1)
        self.synthetic.armature = 0.02
        self.synthetic.viscous_friction = 0.5
        self.synthetic.coulomb_friction = 0.2
        self.synthetic.encoder_bias = 0.05
        self.synthetic.delay = 7
        # set bounds for parameters
        self.bounds_params[:12, 0] = 1e-4
        self.bounds_params[:12, 1] = 0.1  # armature between 1e-4 - 0.1 [kgm2]
        self.bounds_params[12:24, 1] = 2.0  # dof_damping between 0.0 - 2.0 [Nm s/rad]
        self.bounds_params[24:36, 1] = 1.0  # friction between 0.0 - 1.0 [Nm]
        self.bounds_params[36:48, 0] = -0.1
        self.bounds_params[36:48, 1] = 0.1  # bias between -0.1 - 0.1 [rad]
        self.bounds_params[48, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]

        # steps for stage 2, inside the MJCF limits; with Kp 60 the PD torque reaches the peak
        # torque for errors above 0.59 rad (hip, thigh) and 0.74 rad (calf). Kd 0.5, not 2: with Kd 2
        # the D term (19-52 N m above the corner speed) kept the thigh PD torque off the torque-speed line
        self.step = StepCfg(center=[0.0, 0.8, -1.5] * 4, amplitude=[0.5, 0.6, 0.6] * 4, kd=0.5)
        # stage 2: hip, thigh and knee constants, each shared by the 4 legs; CMA-ES starts at the bounds midpoints
        self.envelope = EnvelopeCfg(
            data_dir="aliengo_sim/step_data.pt",
            groups={"hip": ".*_hip_joint", "thigh": ".*_thigh_joint", "knee": ".*_calf_joint"},
            bounds_params=torch.tensor([
                [0.60, 1.35],  # c_omega hip [V s/rad]
                [0.60, 1.35],  # c_omega thigh
                [0.75, 1.70],  # c_omega knee
                [0.09, 0.77],  # c_tau hip [V/(N m)]
                [0.09, 0.77],  # c_tau thigh
                [0.07, 0.61],  # c_tau knee
            ]),
        )


@configclass
class AliengoPaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Aliengo robot in Pace Sim2Real environment."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.MjcfFileCfg(
            asset_path=ALIENGO_MJCF,
            # the importer fixes the base; fix_root_link fails on the converted (layered) USD
            fix_base=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={
                ".*R_hip_joint": -0.1,
                ".*L_hip_joint": 0.1,
                "F[LR]_thigh_joint": 0.8,
                "R[LR]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
        ),
        actuators={"legs": ALIENGO_PACE_ACTUATOR_CFG},
    )


@configclass
class AliengoPaceEnvCfg(PaceSim2realEnvCfg):
    scene: AliengoPaceSceneCfg = AliengoPaceSceneCfg()
    sim2real: PaceCfg = AliengoPaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # base is fixed by the MJCF importer (fix_base), see the scene config
        self.scene.robot.spawn.articulation_props.fix_root_link = None

        # robot sim and control settings, same 500 Hz as the MJCF timestep and the robot's hardware loop
        self.sim.dt = 0.002
        self.decimation = 1
