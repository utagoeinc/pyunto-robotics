#!/usr/bin/env python3
"""Run one instruction headlessly and write everything Artefacts should show.

`run_robot.py --say ... --view` is the interactive form: it needs a window, and on macOS a
mjpython interpreter to own it. Neither exists on a cloud runner, so this is the same run
with the window replaced by a recording.

    python scripts/artefacts_run.py --say "..." --output out

Writes into the output directory:

    video.mp4        the run, from a camera that follows the robot
    frames/*.png     the same footage as stills, for when mp4 encoding is unavailable
    metrics.json     pass/fail plus step counts, picked up as Artefacts metrics
    tests_junit.xml  pass/fail as Artefacts reads it (it does not look at the exit code)
    run.json         plan, reply, and per-step outcome
    run.log          the stdout/stderr of the run

The instruction comes from --say, or from the ARTEFACTS_SAY environment variable when the
job sets it as a scenario parameter (a null-framework job turns params into env vars).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyunto_robotics.agent import RobotAgent  # noqa: E402
from pyunto_robotics.brain.planner import LLMPlanner, RulePlanner  # noqa: E402
from pyunto_robotics.perception.grounding import ColorGrounder, VLMGrounder  # noqa: E402
from pyunto_robotics.sim.robot import Robot  # noqa: E402

log = logging.getLogger("artefacts_run")

# Recording at the control rate would produce a frame every few milliseconds of sim time and
# an enormous file. 20fps of wall-clock-ish playback is enough to see what the robot did.
VIDEO_FPS = 20


def _count_wall_contacts(robot: Robot) -> int:
    """Whether the robot is touching a wall or door frame right now.

    Scraping along a wall is invisible in pass/fail -- the errand still finishes -- so it needs
    its own number or it goes unnoticed. Counted here rather than in a probe script because
    wrapping robot.step from outside perturbs the run enough to change its outcome.
    """
    import mujoco  # noqa: PLC0415 - only needed for this introspection

    m, d = robot.model, robot.data
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist >= 0:
            continue
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if "floor" in n1 or "floor" in n2:
            continue
        if n1.startswith(("wall", "frame", "part")) or n2.startswith(("wall", "frame", "part")):
            return 1
    return 0


class Recorder:
    """Captures a follow-camera frame every so many control steps.

    Wraps Robot.step the same way the viewer does in run_robot.py, for the same reason: the
    agent drives the robot through many layers, and there is no single loop here to hook.
    Also tallies wall contact, which needs the same hook and must not cost a second one.
    """

    def __init__(self, robot: Robot, out_dir: Path, camera: str, every: int) -> None:
        self._robot = robot
        self._dir = out_dir
        self._camera = camera
        self._every = max(1, every)
        self._n = 0
        self.steps = 0
        self.contact_steps = 0
        self.frames: list[Path] = []
        self._failed_once = False

        real_step = robot.step

        def step_and_record(*a, **kw):
            real_step(*a, **kw)
            self.steps += 1
            self.contact_steps += _count_wall_contacts(robot)
            if self._n % self._every == 0:
                self._capture()
            self._n += 1

        robot.step = step_and_record  # type: ignore[method-assign]

    def _capture(self) -> None:
        try:
            obs = self._robot.look(camera=self._camera)
        except Exception as e:  # noqa: BLE001 - a bad camera name must not kill the run
            if not self._failed_once:
                log.warning("recording from %r failed (%s); video will be short", self._camera, e)
                self._failed_once = True
            return
        from PIL import Image

        path = self._dir / f"{len(self.frames):05d}.png"
        Image.fromarray(obs.rgb).save(path)
        self.frames.append(path)


def _encode_video(frames: list[Path], dest: Path) -> bool:
    """Stitch the stills into an mp4. Returns False if no encoder is available.

    The stills are kept either way — the dashboard renders them, and they are the fallback
    when a runner image has no ffmpeg.
    """
    if not frames:
        return False
    try:
        import imageio.v2 as imageio
    except ImportError:
        log.info("imageio not installed; keeping frames only")
        return False
    try:
        with imageio.get_writer(dest, fps=VIDEO_FPS, macro_block_size=None) as w:
            for f in frames:
                w.append_data(imageio.imread(f))
    except Exception as e:  # noqa: BLE001 - missing ffmpeg surfaces here
        log.warning("could not encode %s (%s); keeping frames only", dest.name, e)
        return False
    return True


def _write_junit(dest: Path, name: str, ok: bool, reply: str, seconds: float) -> None:
    """Record the outcome where Artefacts looks for it.

    The null-framework runner decides pass/fail by parsing tests_junit.xml, and treats a run
    with no such file as a pass. Without this, a robot that never found the door still shows
    up green on the dashboard.
    """
    from xml.etree import ElementTree as ET

    suite = ET.Element("testsuite", name="pyunto-robotics", tests="1",
                       failures="0" if ok else "1", errors="0", time=f"{seconds:.2f}")
    case = ET.SubElement(suite, "testcase", classname="artefacts_run", name=name,
                         time=f"{seconds:.2f}")
    if not ok:
        ET.SubElement(case, "failure", message=reply).text = reply
    ET.ElementTree(suite).write(dest, encoding="utf-8", xml_declaration=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--say", default=os.getenv("ARTEFACTS_SAY"),
                    help="the instruction to carry out (default: $ARTEFACTS_SAY)")
    ap.add_argument("--output", default=os.getenv("ARTEFACTS_OUTPUT", "output"),
                    help="directory to write results into")
    ap.add_argument("--scene", default=os.getenv("ARTEFACTS_SCENE", "office.xml"))
    ap.add_argument("--keyframe", default=os.getenv("ARTEFACTS_KEYFRAME", "lobby"))
    ap.add_argument("--llm", action="store_true",
                    default=os.getenv("ARTEFACTS_LLM", "").lower() in ("1", "true", "yes"),
                    help="plan with Gemma 4 instead of the rule matcher")
    ap.add_argument("--vlm", action="store_true",
                    default=os.getenv("ARTEFACTS_VLM", "").lower() in ("1", "true", "yes"))
    ap.add_argument("--camera", default=os.getenv("ARTEFACTS_CAMERA", "head_cam"),
                    help="which camera to record from")
    ap.add_argument("--record-every", type=int, default=10,
                    help="capture one frame per N control steps")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if not args.say:
        print("ERROR: no instruction. Pass --say TEXT or set ARTEFACTS_SAY.", file=sys.stderr)
        return 2

    out = Path(args.output)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out / "run.log", encoding="utf-8")],
    )
    for noisy in ("httpx", "urllib3", "engineio", "socketio", "huggingface_hub", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    grounder = VLMGrounder() if args.vlm else ColorGrounder()
    planner = LLMPlanner() if args.llm else RulePlanner()
    log.info("planner : %s", type(planner).__name__)
    log.info("grounder: %s", type(grounder).__name__)
    log.info("scene   : %s (starting %s)", args.scene, args.keyframe)
    log.info("instruction: %r", args.say)

    robot = Robot(args.scene, keyframe=args.keyframe)
    recorder = Recorder(robot, frames_dir, args.camera, args.record_every)

    started = time.monotonic()
    try:
        agent = RobotAgent(robot, grounder, planner=planner)
        execution = agent.execute(args.say)
    finally:
        robot.close()
    elapsed = time.monotonic() - started

    log.info("plan  : %s", execution.plan)
    log.info("reply : %s", execution.reply())
    log.info("result: %s", "ok" if execution.ok else "FAILED")

    has_video = _encode_video(recorder.frames, out / "video.mp4")

    steps = [str(s) for s in execution.plan.steps]

    # The narrowest a door was left when the robot walked through it. This is the number that
    # separates a real pass from a door that "opened" 17 degrees onto a 4 cm gap, and a run
    # that never touched a door reports nothing rather than a misleading zero.
    swings = [d["swing_degrees"] for d in execution.data if "swing_degrees" in d]
    (out / "run.json").write_text(json.dumps({
        "instruction": args.say,
        "scene": args.scene,
        "keyframe": args.keyframe,
        "planner": type(planner).__name__,
        "grounder": type(grounder).__name__,
        "plan": steps,
        "steps_measured": execution.data,
        "messages": execution.messages,
        "reply": execution.reply(),
        "ok": execution.ok,
        "video": "video.mp4" if has_video else None,
        "frames": len(recorder.frames),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Flat numbers only: Artefacts charts these per run, so they have to be scalars.
    (out / "metrics.json").write_text(json.dumps({
        "success": 1 if execution.ok else 0,
        "planned_steps": len(steps),
        "frames_recorded": len(recorder.frames),
        "duration_s": round(elapsed, 2),
        # Scraping along a wall does not fail the errand, so without this it never shows up.
        "wall_contact_steps": recorder.contact_steps,
        "wall_contact_pct": round(100.0 * recorder.contact_steps / max(recorder.steps, 1), 1),
        **({"min_door_swing_deg": round(min(swings), 1)} if swings else {}),
    }, indent=2), encoding="utf-8")

    _write_junit(out / "tests_junit.xml", "right_door_then_leftmost_room",
                 execution.ok, execution.reply(), elapsed)

    return 0 if execution.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
