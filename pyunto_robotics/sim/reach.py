"""Putting a hand where you want it.

The office robot never needed this: a door handle is at a known height and a fixed
(shoulder_pitch, elbow) pair puts the gripper on it. Laundry is different -- a towel corner is
wherever it has fallen, the drum is a cavity the hand has to enter, and the fold target moves
as the sheet does. So the arm has to be aimed at a world POINT, not at a remembered pose.

This is damped least-squares IK on MuJoCo's own Jacobian, run on a scratch copy of the state so
the search never disturbs the live simulation. It is deliberately small: the arm is a 4-DoF
chain (three shoulder axes and an elbow), the targets are all within a metre, and anything more
elaborate would be solving a problem this robot does not have.

Damping matters. An undamped pseudo-inverse blows up near the edge of the workspace -- asking
for a point 5 cm beyond the arm's reach produces joint velocities in the hundreds of radians,
and the servo then slams the arm into its stops. The damping term trades a little accuracy for
a solution that degrades gracefully into "stretched out toward it as far as I go", which is
also the behaviour a caller wants when a target turns out to be too far away.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

log = logging.getLogger(__name__)

# Joints the solver is allowed to move, per side. The waist and legs are excluded on purpose:
# letting IK move the base turns every reach into a whole-body motion, and a robot that leans
# to pick something up while its gait controller is also holding it upright fights itself.
ARM_JOINTS = ("sh_pitch_{s}", "sh_roll_{s}", "sh_yaw_{s}", "elbow_{s}")

# Damping for the least-squares step. 0.05 was picked by measurement, not taste: at 0.01 a
# target just out of reach produced joint steps above 20 rad, and at 0.2 the solver needed
# roughly three times as many iterations to converge on reachable targets.
DAMPING = 0.05

# Stop when the hand is this close. 8 mm is well inside the gripper's tolerance -- the fingers
# span 6 cm -- and chasing more precision than that just burns iterations on a target that is
# itself moving, because cloth sags while you reach for it.
TOLERANCE_M = 0.008

# How far in front of the base the hand can actually be PUT, as opposed to how far the joint
# chain extends. These are not the same number and the difference matters:
#
#   forward   final hand error after 4 closed-loop passes
#   0.20 m    0.023 m     <- works
#   0.25 m    0.029 m     <- works
#   0.30 m    0.194 m     <- fails
#   0.40 m    0.230 m     <- fails
#
# The kinematics say 0.40 m and the pure IK residual agrees, but past 0.25 m the extended arm
# loads the torso, the body yields, and the hand settles a fifth of a metre short no matter how
# many correction passes are run -- the error grew pass over pass rather than converging.
#
# So this is the number callers must plan around: stand within 0.25 m of what you intend to
# pick up. Reaching further is not a matter of trying harder, it is walking closer.
WORKING_REACH_M = 0.25


class ArmSolver:
    """Solves arm joint angles that put a gripper on a world point."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        # A scratch MjData so the search never touches the live state. Running IK in-place and
        # rewinding afterwards is possible but easy to get wrong -- it leaves contacts and
        # warm-start data from poses the robot never actually adopted.
        self._scratch = mujoco.MjData(model)

    def joint_ids(self, side: str) -> list[int]:
        """Model joint ids for one arm, in shoulder-to-hand order."""
        ids = []
        for pattern in ARM_JOINTS:
            joint = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, pattern.format(s=side)
            )
            if joint >= 0:
                ids.append(joint)
        return ids

    def solve(
        self,
        data: mujoco.MjData,
        target: np.ndarray,
        side: str = "r",
        max_iters: int = 120,
    ) -> tuple[dict[str, float], float]:
        """Joint angles that put the `side` gripper on `target`.

        Returns the angles by actuator name, and the residual distance in metres. The residual
        is the honest part: it is how the caller learns the target was out of reach, rather
        than finding out by watching the hand stop short.

        The starting pose is whatever the robot is in now, so a small correction stays a small
        motion instead of swinging the arm through the workspace to an unrelated solution.
        """
        joints = self.joint_ids(side)
        if not joints:
            return {}, float("inf")

        site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"grip_{side}")
        if site < 0:
            return {}, float("inf")

        scratch = self._scratch
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0.0

        qpos_adr = [self.model.jnt_qposadr[j] for j in joints]
        dof_adr = [self.model.jnt_dofadr[j] for j in joints]
        limits = [self.model.jnt_range[j] for j in joints]

        target = np.asarray(target, dtype=float)
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))

        residual = float("inf")
        for _ in range(max_iters):
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)

            error = target - scratch.site_xpos[site]
            residual = float(np.linalg.norm(error))
            if residual < TOLERANCE_M:
                break

            mujoco.mj_jacSite(self.model, scratch, jac_pos, jac_rot, site)
            # Only the arm's own columns. Including the free joint would let the solver
            # "reach" by teleporting the whole robot, which is a perfectly good least-squares
            # answer and completely useless.
            jac = jac_pos[:, dof_adr]

            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e
            jjt = jac @ jac.T + (DAMPING**2) * np.eye(3)
            step = jac.T @ np.linalg.solve(jjt, error)

            # Cap the per-iteration motion. Without this the first step of a far reach is large
            # enough to swing the arm past the solution and oscillate.
            largest = np.abs(step).max()
            if largest > 0.25:
                step *= 0.25 / largest

            for k, adr in enumerate(qpos_adr):
                lo, hi = limits[k]
                scratch.qpos[adr] = float(np.clip(scratch.qpos[adr] + step[k], lo, hi))

        angles = {
            ARM_JOINTS[k].format(s=side): float(scratch.qpos[adr])
            for k, adr in enumerate(qpos_adr)
        }
        return angles, residual

    def reachable(self, data: mujoco.MjData, target: np.ndarray, side: str = "r") -> bool:
        """Whether the hand can actually get to a point from where the robot stands now.

        Callers use this to decide whether to step closer BEFORE committing to a reach, which
        is the difference between "walk up and pick it up" and an arm waving at something a
        metre away.
        """
        _, residual = self.solve(data, target, side)
        return residual < 0.05

    def correction(
        self, data: mujoco.MjData, target: np.ndarray, side: str = "r"
    ) -> tuple[dict[str, float], float]:
        """Re-solve from where the hand ACTUALLY is, not from where it was asked to be.

        Solving once and driving to the answer is not enough, and the reason is worth stating:
        the solve runs on a scratch copy holding the current pose, but as the arm extends, the
        torso yields under the load and the servos settle a little short of their commands.
        Measured on the first reach: IK residual 0.024 m, hand 0.20 m from the target once the
        physics settled. Stiffening the shoulder cut the joint error from 0.204 to 0.065 rad
        but did not close the gap, because the remaining error is the body moving, not the
        joint sagging.

        So a reach is a loop, not a shot. Each pass measures the real hand position and asks
        for the correction, which is what a real arm controller does for the same reason.
        """
        return self.solve(data, target, side)
