# pyunto-robotics

A humanoid robot in a simulated office that you control by messaging its **Pyunto** account.

> Send 「オフィスのドアを開けて」 from the Pyunto app → the robot works out what you meant,
> finds the door from camera images alone with no prior map, walks across the office, pushes it
> open, goes through, and messages you back what happened.

Everything runs on one MacBook. No CUDA, no cloud inference, no pre-built map.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Project scaffolding | done |
| 1 | Pyunto client (login, E2EE messages, realtime) | done |
| 2 | Custom humanoid + office simulation (MuJoCo) | done |
| 3 | Perception + mapless navigation | done |
| 4 | Door opening + instruction understanding (Gemma 4) | done |
| 5 | End-to-end integration | done |
| 6 | RL locomotion instead of kinematic gait | optional |

## Setup

```bash
uv venv --python 3.12 .venv        # 3.12 required: mlx-vlm has no 3.13+ support
VIRTUAL_ENV=$PWD/.venv uv pip install -e .
cp .env.example .env               # then fill in PYUNTO_PASSWORD

# optional: local Gemma 4 for open-vocabulary instructions (~3 GB download)
VIRTUAL_ENV=$PWD/.venv uv pip install -e '.[llm]'
```

## Run it

**To watch the robot, launch with `mjpython`, not `python`.** Everything works under plain
`python` too, but headless -- the simulation runs and reports back in text with no window.
(macOS binds the viewer to the Cocoa event loop on the main thread, and only `mjpython`, which
ships with the mujoco wheel, sets that up.)

```bash
# Watch it carry out one instruction
./.venv/bin/mjpython scripts/run_robot.py --view --say "右のドアを開けて"
./.venv/bin/mjpython scripts/view_sim.py --say "左のドアを開けて"     # same thing, sim-only

# Chained instructions need --llm: the rule matcher only ever produces one step, and will
# silently do the wrong one. It prints a warning when it spots a chained instruction.
./.venv/bin/mjpython scripts/run_robot.py --view --llm \
    --say "右のドアを開けて、その後、一番左の部屋に入って"

# Headless: no window, just the result
./.venv/bin/python scripts/run_robot.py --say "オフィスのドアを開けて"
./.venv/bin/python scripts/run_robot.py --llm --say "会議室のドアを開けて中に入って"

# Connected: listen on Pyunto and act on whatever you message
./.venv/bin/mjpython scripts/run_robot.py --view          # with a window
./.venv/bin/python scripts/run_robot.py                   # without
./.venv/bin/python scripts/run_robot.py --join INVITE_CODE  # first time, to join your space
```

`--speed` sets playback rate (default 3x; `--speed 1` is real time). A trip to a far door
takes over a minute at 1x.

Other tools:

```bash
./.venv/bin/mjpython scripts/view_sim.py --walk        # canned route: ignores the camera,
                                                       # only checks the gait and collisions
./.venv/bin/python scripts/view_sim.py --shot out.png  # stills; plain python is fine
./.venv/bin/python scripts/test_comms.py --listen      # Pyunto connection only
./.venv/bin/python -m pytest tests/ -q                 # 112 tests
```

## What it understands

English and Japanese, via pattern matching (instant) or Gemma 4 (open-vocabulary):

| You say | It does |
|---|---|
| 「オフィスのドアを開けて」 / "open the door" | walks over and pushes it open |
| 「右のドアを開けて」 / "open the door on the left" | picks that specific door of the three |
| 「右のドアを開けて、その後左の部屋に」 | multi-step, with Gemma 4 (see limitation below) |
| 「ドアまで行って」 / "go to the whiteboard" | navigates to it |
| 「周りを見て」 / "look around" | turns in place, reports what it saw |
| 「何が見える？」 / "what do you see" | describes the current view |
| 「どこにいる」 / "where are you" | names the room it is in |

## Design

Three loops at different rates, so the language model never sits in the control path:

```
Pyunto (Socket.IO)
      │  instruction text
      ▼
 ~1Hz    Gemma 4 E2B (MLX)      "open the door" → [open(door)]
      ▼
 ~5Hz    perception + mapless nav   RGB-D → where is the door → velocity command
      ▼
 50Hz    gait controller            step(vx, vy, wz)     ← the seam that matters
      ▼
         MuJoCo 3.11
```

`step(vx, vy, wz)` is the only movement primitive anything above the simulator uses. The gait
behind it is currently kinematic — it cannot fall over, which is what makes a live demo
survivable — and swapping in a trained RL policy changes nothing upstream.

**Navigation is mapless by construction.** There is no occupancy grid and no SLAM; the robot's
memory of the world is exactly one camera frame deep. Depth is read as per-bearing clearances
rather than a metric grid, because a grid would be a map by another name and would drift with
odometry. Obstacle avoidance sits *underneath* target-seeking, so a confident but wrong
detection still cannot walk the robot into a wall.

## Measured on this machine (M5 Max, 128 GB)

| | |
|---|---|
| Physics, single thread | 81,000 steps/s (406× realtime) |
| Physics, 16 processes × 4 envs | 92,800 env-steps/s |
| Camera RGB 224×224 | 1.4–2.0 ms (500–700 fps) |
| Camera depth 224×224 | 0.6–0.9 ms |
| Gemma 4 E2B load / plan | 2.4 s / 0.11–0.27 s |
| Full "open the door" run | ~350 control steps, ~1 s wall clock |
| Detour to a far door | ~1400 control steps |

RL locomotion training is viable here despite the lack of CUDA: rollouts collect at ~72k
env-steps/s on CPU and MPS gradient updates are ~24× faster than CPU, putting 100–200M steps
within a few hours. MJX/JAX is *not* the route — `jax-metal` is unmaintained, and MJX is slower
than plain MuJoCo for a single robot anyway.

## Opening doors: push and pull

The robot can both push a door open and grasp the handle to pull it. Pulling came later, and
the reason is worth recording: the first design could only push, because a push needs the hand
somewhere on the leaf rather than precisely on a 3.6 cm handle. That was a reasonable
simplification and a bad long-term choice -- a door pushed into a room swings across the way
back out, so a push-only robot has no way to leave a room it entered.

The hardware was never the limitation. The gripper opens to 6.4 cm and closes to 4.2 cm against
a 3.6 cm handle, with high-friction fingers and `condim=4`. It was simply never asked to grasp
anything.

A friction grasp does not hold in MuJoCo -- the fingers slip off long before the arm can swing
a 20 kg leaf -- so a closed hand is modelled as a weld, which is standard practice. The
relative pose is written into `eq_data` at the moment of contact; without that the solver
enforces the compiled offset and the door teleports into the hand.

### Leaving a room is planned, not reactive

`leave_room` is the one manoeuvre here that runs a fixed sequence instead of steering frame by
frame, and it has to be. In a doorway the robot has under 0.5 m of clearance in every
direction, so the obstacle-avoiding controller that works everywhere else has no single-step
move that improves anything -- forward is the leaf, back is the room, and it just oscillates.

Instead `open_door` records the pose on the corridor side of the threshold as it goes through,
and leaving replays it: turn to face back, pull the leaf clear if the robot is against it, then
drive to the remembered spot without re-planning. Short, blind, and reliable.

Two details that mattered:

- The recorded pose is where the robot was *standing*, not the doorway itself. Recording the
  doorway put the target at y=1.27 for a threshold at y=1.0 -- inside the room -- so returning
  to it never left.
- Whether the leaf is in the way is decided by *contact*, not by forward clearance. In the
  pantry the robot ends up pressed against the door with chest, thigh and foot while the depth
  camera still reads 1.8 m ahead: the door is beside it, not in front.

Getting out of the last room needed one more thing: the doors are spring-loaded, so a robot
that stops in the opening gets squeezed -- measured closing from 21 degrees to 13 while it
stood there. Backing off and strafing only ever lost ground. The fix is to brace an arm against
the leaf and keep walking, which turns the robot into its own doorstop; the door gives way and
it goes through. It also keeps pushing for a moment after contact breaks, because stopping the
instant the leaf lets go leaves it still in the opening for the spring to close on again.

All three rooms: entered 3/3, left 3/3.

### Choosing a second room: go back to where you were told

「最初にいる位置からみて、三つ見えるドアのうち、右のドアを…今度は一番左の部屋に」 names its own
frame of reference, and the robot now uses it. `return_home` walks back to where it was standing
when it got the instruction, which is both what the user was describing from and the only spot
with all three doors in frame.

Without it the robot opens whichever door it happens to be beside. Measured: one door visible
from y=0.9, two from y=0.67, all three only from the starting position. So "the leftmost" from
next to the pantry means the meeting room.

One subtlety cost a while: standing on the exact starting spot facing the exact starting
heading still showed one door instead of three. The waist joint had drifted to -30 degrees and
stayed there -- the head camera hangs off the torso, so the view was aimed 30 degrees away from
where the body pointed. Its servo is deliberately weak (kp=1) so the torso stays compliant while
walking, and stiffening it to fix this broke the gait badly enough to fail four tests, so the
joint is reset directly instead.

**Known limitation:** going home reliably aims the robot at the correct side of the office, but
reaching and opening that far door afterwards is not yet dependable -- it stops around x=-2.9,
short of a door at x=-4. The test asserts the choice rather than the arrival.

Plans need the step explicitly: `open(right) -> leave -> home -> open(left)`. The prompt tells
Gemma 4 to emit it.

## Notes on the Pyunto backend

The server is Node/TypeScript + Express + Socket.IO (not FastAPI, despite older docs), and
`pyunto-server/api.md` is out of date. Behaviour this client depends on:

- Login returns `token` (not `access_token`); no refresh endpoint, so re-login on 401.
- Message bodies are AES-256-GCM encrypted with a per-space key from
  `GET /api/chat-spaces/:uuid/key`. That endpoint returns the key in the clear and is slated
  for removal in E2EE Phase 3 — hence the `SpaceKeyProvider` seam in `comms/keys.py`.
- REST is snake_case; Socket.IO payloads are camelCase inside `data`.
- The server echoes your own posts back to you — the listener filters on sender uuid, or the
  robot would answer itself forever.
- Thread membership (not space membership) governs visibility.
- `GET /api/messages/:threadId` omits `chat_space_id`; it is recovered from
  `encryption_metadata`.

## Gotchas worth knowing

- **MuJoCo orders `qpos` by the body tree**, so torso and arms come *before* the legs. Writing
  the standing keyframe leg-first put knee targets on shoulders.
- **Gemma 4 will not load through `mlx_lm`.** It is multimodal, so its weights live under
  `language_model.*` and the text-only loader rejects every tensor. Use `mlx_vlm`.
- **Use the 8-bit Gemma build.** Gemma 4's per-layer embeddings quantise badly at 4-bit and
  community 4-bit weights are reported to produce garbage.
- **MuJoCo's renderer is thread-bound.** Calling it off the thread that created it aborts the
  process with a Metal assertion on macOS — so the simulator owns the main thread and
  Socket.IO gets its own.
- `python-socketio` needs the `[client]` extra for `websocket-client`, or the sync client
  silently falls back to polling and the websocket transport fails.
- **The interactive viewer needs `mjpython` on macOS**, not `python`. Note that `sys.executable`
  still reports `python3` under mjpython, so detect it via `mujoco.viewer._MJPYTHON` instead.

## Layout

```
pyunto_robotics/
  agent.py      message → plan → act → reply
  comms/        auth, crypto (AES-256-GCM), keys, client (REST + Socket.IO)
  sim/          robot (step/look/arm), gait (kinematic, RL stub)
  perception/   depth geometry, object grounding (colour + VLM)
  nav/          mapless navigation state machine
  brain/        planner (rules + Gemma 4), skills
assets/
  pyunto_h1.xml custom humanoid: 1.38 m, 23 actuators, grippers, head camera
  office.xml    3 rooms, corridor, lobby, 3 hinged doors
scripts/        run_robot.py, view_sim.py, test_comms.py
tests/          99 tests
```
