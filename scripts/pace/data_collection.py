# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Anymal-D-v0", help="Name of the task.")
parser.add_argument("--min_frequency", type=float, default=0.1, help="Minimum frequency for the chirp signal in Hz.")
parser.add_argument("--max_frequency", type=float, default=10.0, help="Maximum frequency for the chirp signal in Hz.")
parser.add_argument("--duration", type=float, default=20.0, help="Duration of the excitation signal in seconds.")
parser.add_argument("--excitation", type=str, default="chirp", choices=("chirp", "step"), help="Chirp (stage 1) or steps (stage 2).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments; the rest are Hydra overrides, e.g. physics=newton_mjwarp
args_cli, hydra_args = parser.parse_known_args()
# leave only the Hydra overrides in sys.argv, as Isaac Lab's own scripts do
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import re
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root
from pace_sim2real.utils.actuators import (
    actuator_groups,
    read_clip_active,
    read_clip_at_effort_limit,
    read_delayed_effort,
    write_pd_gains,
)
from pace_sim2real.utils.excitation import chirp, steps
from pace_sim2real.utils.joint_properties import write_joint_properties_to_sim, write_joint_state_to_sim


def report_joint_limits(
    joint_order: list[str],
    position: torch.Tensor,
    hardware_limits: dict[str, tuple[float, float]] | None,
    sim_limits: torch.Tensor,
) -> None:
    """Print each joint's position range against its hardware (URDF) and simulator limits, warn on violations.

    ``position`` holds the true joint positions, shape (num_steps, len(joint_order)); ``sim_limits`` has shape
    (len(joint_order), 2).
    """
    if hardware_limits is None:
        print("[WARN]: no hardware joint limits configured (sim2real.joint_limits), position range not checked")
        return
    print("[INFO]: joint position range vs limits [rad]")
    print(f"  {'joint':<16} {'q min':>7} {'q max':>7} {'URDF lo':>8} {'URDF hi':>8} {'margin':>7} {'sim lo':>8} {'sim hi':>8}")
    violations = []
    for j, name in enumerate(joint_order):
        matches = [limits for pattern, limits in hardware_limits.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f"joint {name} must match exactly one joint_limits pattern, matches {len(matches)}")
        low, high = matches[0]
        q_min, q_max = position[:, j].min().item(), position[:, j].max().item()
        margin = min(q_min - low, high - q_max)
        print(f"  {name:<16} {q_min:+7.3f} {q_max:+7.3f} {low:+8.3f} {high:+8.3f} {margin:+7.3f}"
              f" {sim_limits[j, 0].item():+8.3f} {sim_limits[j, 1].item():+8.3f}")
        if margin < 0:
            violations.append((name, -margin))
    for name, excess in violations:
        print(f"[WARN]: {name} exceeds its URDF limit by {excess:.3f} rad")


def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, overrides=hydra_args
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    articulation = env.unwrapped.scene["robot"]

    joint_order = env_cfg.sim2real.joint_order
    joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device)

    truth = env_cfg.sim2real.synthetic
    armature = torch.tensor([truth.armature] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
    damping = torch.tensor([truth.viscous_friction] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
    friction = torch.tensor([truth.coulomb_friction] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
    bias = torch.tensor([truth.encoder_bias] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
    time_lag = torch.tensor([[truth.delay]], dtype=torch.int, device=env.unwrapped.device)
    env.reset()

    write_joint_properties_to_sim(
        articulation,
        armature=armature,
        viscous_friction=damping,
        coulomb_friction=friction,
        joint_ids=joint_ids,
        env_ids=torch.arange(len(armature), device=env.unwrapped.device),
    )
    for actuator, drive_joint_idx in actuator_groups(articulation, joint_ids):
        actuator.update_time_lags(time_lag)
        actuator.update_encoder_bias(bias[:, drive_joint_idx])
        actuator.reset(torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device))

    # PD gains of the recording, in joint_order; the steps use their own gains
    if args_cli.excitation == "step":
        kp = torch.full((len(joint_ids),), env_cfg.sim2real.step.kp, device=env.unwrapped.device)
        kd = torch.full((len(joint_ids),), env_cfg.sim2real.step.kd, device=env.unwrapped.device)
        write_pd_gains(articulation, kp, kd, joint_ids)
    kp = torch.zeros(len(joint_ids), device=env.unwrapped.device)
    kd = torch.zeros(len(joint_ids), device=env.unwrapped.device)
    for actuator, columns in actuator_groups(articulation, joint_ids):
        kp[columns] = actuator.stiffness[0]
        kd[columns] = actuator.damping[0]

    data_dir = project_root() / "data" / env_cfg.sim2real.robot_name

    # Create the excitation signal for each action dimension

    duration = args_cli.duration  # seconds
    sample_rate = 1 / env.unwrapped.sim.get_physics_dt()  # Hz
    num_steps = int(duration * sample_rate)
    t = torch.linspace(0, duration, steps=num_steps, device=env.unwrapped.device)
    if args_cli.excitation == "chirp":
        signal = chirp(t, args_cli.min_frequency, args_cli.max_frequency, duration)
        shape = env_cfg.sim2real.chirp
    else:
        signal = steps(t, env_cfg.sim2real.step.hold, env_cfg.sim2real.step.scales)
        shape = env_cfg.sim2real.step

    trajectory = torch.zeros((num_steps, articulation.num_joints), device=env.unwrapped.device)
    center = torch.tensor(shape.center, device=env.unwrapped.device)
    amplitude = torch.tensor(shape.amplitude, device=env.unwrapped.device)
    trajectory[:, joint_ids] = center.unsqueeze(0) + amplitude.unsqueeze(0) * signal.unsqueeze(-1)

    write_joint_state_to_sim(articulation, trajectory[0, :].unsqueeze(0) + bias[0, joint_ids], joint_ids=None)

    counter = 0
    # simulate environment
    dof_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    dof_target_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    dof_vel_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    tau_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    clip_active_buffer = torch.zeros(num_steps, len(joint_ids), dtype=torch.bool, device=env.unwrapped.device)
    clip_at_limit_buffer = torch.zeros_like(clip_active_buffer)
    time_data = t
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute actions
            dof_pos_buffer[counter, :] = env.unwrapped.scene.articulations["robot"].data.joint_pos[0, joint_ids] - bias[0]
            dof_vel_buffer[counter, :] = articulation.data.joint_vel.torch[0, joint_ids]
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions = trajectory[counter % num_steps, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # apply actions
            obs, _, _, _, _ = env.step(actions)
            dof_target_pos_buffer[counter, :] = env.unwrapped.scene.articulations["robot"]._data.joint_pos_target[0, joint_ids]
            # torque applied during this step (synthetic tau_est), and whether the envelope clipped the PD torque
            tau_buffer[counter, :] = read_delayed_effort(articulation, joint_ids)[0]
            clip_active_buffer[counter, :] = read_clip_active(articulation, joint_ids)[0]
            clip_at_limit_buffer[counter, :] = read_clip_at_effort_limit(articulation, joint_ids)[0]
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter/sample_rate} seconds")
            if counter >= num_steps:
                break

    # close the simulator
    env.close()

    from time import sleep
    sleep(1)  # wait a bit for everything to settle

    # envelope coverage per joint: the clip must stay inactive on chirps; on steps, only the samples on the
    # torque-speed line carry information about the motor constants, the ones at the effort limit do not
    on_line_buffer = clip_active_buffer & ~clip_at_limit_buffer
    print(f"[INFO]: {args_cli.excitation}: envelope coverage per joint")
    print(f"  {'joint':<16} {'peak |qd|':>9} {'corner':>7} {'on speed line':>14} {'|qd| on line':>13} {'at effort limit':>16}")
    for actuator, columns in actuator_groups(articulation, joint_ids):
        if not hasattr(actuator, "_saturation_effort"):
            continue
        corner = actuator.actuator_velocity_limit[0] * (1 - actuator.actuator_effort_limit[0] / actuator._saturation_effort[0])
        for j, col in enumerate(columns.tolist()):
            on_line = on_line_buffer[:, col]
            speeds = dof_vel_buffer[on_line, col].abs()
            speed_range = f"{speeds.min().item():.1f}-{speeds.max().item():.1f}" if on_line.any() else "-"
            print(f"  {joint_order[col]:<16} {dof_vel_buffer[:, col].abs().max().item():9.2f} {corner[j].item():7.2f}"
                  f" {int(on_line.sum()):6d} ({on_line.float().mean().item():5.1%}) {speed_range:>13}"
                  f" {int(clip_at_limit_buffer[:, col].sum()):7d} ({clip_at_limit_buffer[:, col].float().mean().item():5.1%})")
    print("  speeds in rad/s; corner: speed above which the torque-speed line is below the effort limit")
    # recorded positions are in the encoder frame (q - bias)
    report_joint_limits(
        joint_order,
        dof_pos_buffer + bias[0],
        env_cfg.sim2real.joint_limits,
        articulation.data.joint_pos_limits.torch[0, joint_ids],
    )

    (data_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "time": time_data.cpu(),
        "dof_pos": dof_pos_buffer.cpu(),
        "des_dof_pos": dof_target_pos_buffer.cpu(),
        "dof_vel": dof_vel_buffer.cpu(),
        "tau_est": tau_buffer.cpu(),
        "clip_active": clip_active_buffer.cpu(),
        "clip_at_effort_limit": clip_at_limit_buffer.cpu(),
        "kp": kp.cpu(),
        "kd": kd.cpu(),
        "excitation": args_cli.excitation,
        "joint_order": joint_order,
    }, data_dir / f"{args_cli.excitation}_data.pt")

    import matplotlib.pyplot as plt

    for i in range(len(joint_ids)):
        plt.figure()
        plt.plot(t.cpu().numpy(), dof_pos_buffer[:, i].cpu().numpy(), label=f"{joint_order[i]} pos")
        plt.plot(t.cpu().numpy(), dof_target_pos_buffer[:, i].cpu().numpy(), label=f"{joint_order[i]} target", linestyle='dashed')
        plt.title(f"Joint {joint_order[i]} Trajectory")
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
