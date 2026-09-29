#!/usr/bin/env python3
"""Your own robot, from its URDF, in a Pyunto diary.

MuJoCo reads URDF directly, so a robot described for ROS can be simulated here without
converting it. This example loads a URDF, gives every joint a position motor, and exposes a few
skills to the diary: raise, lower, wave, and report where the joints are.

    pip install 'pyunto-robotics[llm]'
    python examples/urdf_robot.py                        # the bundled two-joint arm
    python examples/urdf_robot.py --urdf path/to/robot.urdf
    python examples/urdf_robot.py --check                # run the skills offline and exit

A QR code appears (unless --check). Scan it in the Pyunto app, choose a diary, and write
"@<robot name> raise" or "@<robot name> wave".

Works the same on Linux, Windows and macOS: nothing here needs Apple silicon. `--view` opens
the MuJoCo window (on macOS run it with `mjpython` instead of `python`).

Three things to know when you bring your own URDF:

* Mesh files. Relative <mesh filename="..."> paths resolve against the URDF's folder. A
  `package://` path from ROS will not resolve; rewrite it to a relative path.
* A fixed base collides. MuJoCo merges a fixed base link into its world body, and contacts with
  the world body are not filtered the way parent/child contacts are. If the base's collision
  shape touches the first link, that joint cannot move. Drop or shrink the base collision.
* No motors in URDF. URDF describes joints, not actuators, so this adds a position servo per
  joint (`add_position_motors`). Tune `kp` for your robot's masses.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco

from pyunto_robotics.api import SkillResult

HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE / "urdf" / "arm.urdf"


def load_urdf(path: Path, kp: float = 40.0, kv: float = 4.0) -> mujoco.MjModel:
    """Compile a URDF into a MuJoCo model with a position motor on every hinge/slide joint."""
    spec = mujoco.MjSpec.from_file(str(path))
    add_position_motors(spec, kp=kp, kv=kv)
    return spec.compile()


def add_position_motors(spec: mujoco.MjSpec, kp: float, kv: float) -> None:
    for joint in spec.joints:
        if joint.type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        motor = spec.add_actuator()
        motor.name = joint.name
        motor.target = joint.name
        motor.trntype = mujoco.mjtTrn.mjTRN_JOINT
        motor.set_to_position(kp=kp, kv=kv)


class URDFRobot:
    """Skills over a MuJoCo model built from URDF. Satisfies `pyunto_robotics.api.RobotSkills`.

    The rules from examples/my_robot.py hold: never raise, and answer in sentences -- the
    message goes into somebody's diary.
    """

    def __init__(self, model: mujoco.MjModel, viewer=None):  # noqa: ANN001
        self.model = model
        self.data = mujoco.MjData(model)
        self.viewer = viewer
        self.joints = [model.actuator(i).name for i in range(model.nu)]

    # -- motion -----------------------------------------------------------------------------

    def _limits(self, i: int) -> tuple[float, float]:
        jid = self.model.actuator_trnid[i, 0]
        if self.model.jnt_limited[jid]:
            lo, hi = self.model.jnt_range[jid]
            return float(lo), float(hi)
        return -math.pi, math.pi

    def move_to(self, targets: list[float], seconds: float = 1.5) -> None:
        for i, target in enumerate(targets):
            lo, hi = self._limits(i)
            self.data.ctrl[i] = min(max(target, lo), hi)
        steps = int(seconds / self.model.opt.timestep)
        for n in range(steps):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and n % 8 == 0:
                self.viewer.sync()
                time.sleep(self.model.opt.timestep * 8)

    def angles(self) -> list[float]:
        return [float(self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]])
                for i in range(self.model.nu)]

    # -- skills -----------------------------------------------------------------------------

    def run(self, action, argument=None, where=None, expect=None) -> SkillResult:  # noqa: ANN001
        if not self.joints:
            return SkillResult(False, "This robot has no joints I can move.")
        if action in ("raise", "up", "lift"):
            self.move_to([self._limits(i)[1] * 0.8 for i in range(self.model.nu)])
            return SkillResult(True, "I raised my arm.", data=self._state())
        if action in ("lower", "down", "rest", "home"):
            self.move_to([0.0] * self.model.nu)
            return SkillResult(True, "I am back at rest.", data=self._state())
        if action == "wave":
            first_lo, first_hi = self._limits(0)
            for _ in range(2):
                self.move_to([first_hi * 0.6] + [0.0] * (self.model.nu - 1), 0.6)
                self.move_to([first_lo * 0.6] + [0.0] * (self.model.nu - 1), 0.6)
            self.move_to([0.0] * self.model.nu, 0.6)
            return SkillResult(True, "I waved.", data=self._state())
        if action in ("where", "report", "status", "describe"):
            parts = ", ".join(f"{n} {math.degrees(a):.0f}°" for n, a in zip(self.joints, self.angles()))
            return SkillResult(True, f"My joints: {parts}.", data=self._state())
        return SkillResult(False, f"I do not know how to '{action}'. I can raise, lower, wave, "
                                  "and report where my joints are.")

    def _state(self) -> dict:
        return {name: round(math.degrees(a), 1) for name, a in zip(self.joints, self.angles())}


class DirectAgent:
    """Maps the first word of an entry to a skill: "raise", "wave", "report".

    Enough for a first run. For sentences instead of commands, give your skills a Domain and
    use RobotAgent with DomainLLMPlanner, as the bundled robots do (pyunto_robotics/robots).
    """

    def __init__(self, skills: URDFRobot, name: str):
        self.skills, self.name = skills, name.lstrip("🤖").strip().lower()

    def execute(self, text: str, report=None):  # noqa: ANN001, ANN201 - RobotBackend passes report
        words = [w.strip(".,!?").lower() for w in text.split() if not w.startswith("@")]
        words = [w for w in words if w and w != self.name]
        result = self.skills.run(words[0] if words else "report")
        return type("Execution", (), {"reply": lambda _self=None, r=result: r.message})()


def check(robot: URDFRobot) -> int:
    """Run every skill once without a network. Used by the tests and CI."""
    for action in ("raise", "report", "wave", "lower", "report", "fly"):
        result = robot.run(action)
        print(f"{action:7} {'ok ' if result.ok else 'no '} {result.message}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--name", default="Arm", help="the robot's display name in the diary")
    ap.add_argument("--operator", default="", help="who runs it; shown before people approve")
    ap.add_argument("--view", action="store_true", help="open the MuJoCo window")
    ap.add_argument("--check", action="store_true", help="run the skills offline and exit")
    args = ap.parse_args()

    model = load_urdf(args.urdf)
    viewer = None
    if args.view:
        import mujoco.viewer  # noqa: PLC0415

        data_holder = URDFRobot(model)
        viewer = mujoco.viewer.launch_passive(model, data_holder.data)
        robot = data_holder
        robot.viewer = viewer
    else:
        robot = URDFRobot(model)
    print(f"loaded {args.urdf.name}: joints {', '.join(robot.joints) or '(none)'}")

    if args.check:
        return check(robot)

    from pyunto_agent.bridge import Bridge  # noqa: PLC0415
    from pyunto_agent.pairing import (  # noqa: PLC0415
        encode_payload,
        pairing_payload,
        render_qr,
        wait_for_scan,
    )

    from pyunto_robotics.backend import RobotBackend  # noqa: PLC0415
    from pyunto_robotics.connect import connect  # noqa: PLC0415

    connection = connect(display_name=args.name)
    payload = pairing_payload(
        user_id=connection.user_id,
        display_name=connection.identity.display_name,
        public_key=connection.identity_store.public_key_b64,
        operator=args.operator,
        runtime="self_hosted",
    )
    print()
    print(render_qr(encode_payload(payload)) or encode_payload(payload))
    print(f"\nScan this in the Pyunto app to let {connection.identity.display_name} into a diary.")
    print("waiting for the scan… (Ctrl-C to stop)")
    try:
        space_id = wait_for_scan(connection.client)
    except KeyboardInterrupt:
        return 0
    if space_id is None:
        print("Nobody scanned it. Run this again when you are ready.")
        return 1
    print("paired ✓ — open that space in the app once so the robot is given the key.")
    bridge = Bridge(connection.client,
                    RobotBackend(DirectAgent(robot, connection.identity.display_name)),
                    persona="", space_ids={space_id}, history=2)
    try:
        bridge.run()
    except KeyboardInterrupt:
        bridge.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
