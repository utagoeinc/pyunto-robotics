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

### Tracking a chosen target by where it is, not where it looks

The navigator used to follow a target by its bearing, and swapped doors whenever the view
changed. It now records where the target is in the world and re-acquires it by position, which
a changing viewpoint cannot disturb.

Making that work turned up a systematic error worth recording. MuJoCo's depth buffer holds
distance *along the view axis*, not distance to the point, and `target_offset` was using it
directly. For anything off-centre those differ by 1/cos(bearing): three doors at 4.0, 5.66 and
5.66 m all read as 3.90 m, so the two at 45 degrees landed 1.76 m from where they actually
were. `free_space` had always corrected for this in its clearance columns; `target_offset`
never had. With the correction the estimates come out 0.10-0.34 m from ground truth, and
picking a named door went from 2/3 to 3/3.

The other fix was squaring up to a doorway *before* pushing rather than after. Approaching the
left door leaves the robot at 167 degrees for an opening that faces 90, and a body turned 80
degrees across a 0.98 m gap does not fit through it.

**Known limitation:** resolving "the leftmost door" from the starting viewpoint is now correct
-- it picks the door at x=-3.6 rather than whichever is nearest -- but walking there afterwards
is not yet dependable. The tracked position is dropped during a long detour and the robot
re-acquires a nearer door. The test asserts the choice rather than the arrival.

### Landmarks: knowing which door is which

`perception/landmarks.py` gives each door an identity. Every sighting is matched against what
has been seen before -- by world position, the one thing about a door that does not change --
and either updates an existing landmark or starts a new one. A landmark's position is the
average of its own sightings, which is far steadier than any single frame: individual estimates
are 0.1-0.34 m out and the error swings with viewpoint.

This is not SLAM and not a prior map. Nothing is loaded from disk, nothing is built ahead of
time, and the robot still cannot navigate to a door it has not seen. It is the smallest amount
of memory that makes "that one, not the other one" expressible.

Three filters keep it honest, each answering a failure that actually happened:

- **Three sightings before a landmark counts.** A bad range reading lands far enough from a
  real door to start its own landmark; those are usually seen once or twice.
- **Detections merged within a frame, and landmarks consolidated across frames.** Close up,
  colour matching splits one door into two blobs at its edges -- measured one door held as two
  entries 0.89 m apart, both with 160-odd sightings.
- **Landmarks that break the line the others form are dropped.** Doors sit along a wall; a
  phantom from an edge-on range reading does not. The line is refitted without its worst
  outlier first, because a single phantom drags the fit far enough to push a real door out
  (measured a true door at 0.83 against a 0.7 threshold while the phantom sat at 1.56).

Walking the corridor now produces three landmarks within 0.3 m of the real doorways.

### Why the robot still walks near the wall

It heads straight at its target from first sighting, so a door across the office is approached
diagonally and the robot ends up alongside the corridor wall -- measured at mean y=+0.26 in a
corridor whose centre is y=0. That reads badly and it is why the logs are full of
`following the wall`.

A centring bias to fix it was written twice and abandoned twice. It works as steering: mean
corridor y goes from +0.26 to -0.16. But walking down the middle keeps all three doors in frame
the whole way, and with per-frame position estimates 0.1-0.34 m noisy the tracked anchor creeps
from one door to the next -- traced sliding from the left door to the middle one over a dozen
frames. Picking a named door drops from 3/3 to 1/3. Gains from 0.5 down to 0, gates from 2.0 m
down to 0.5 m, and smoothing the anchor were all tried; none separated the two effects.

Hugging the wall is, perversely, what makes the current tracking reliable: it narrows the view
so only the target door is in it.

The landmark map was built to break that dependency -- a stable identity to hold onto instead
of a noisy point -- and it does produce good landmarks. Switching the approach loop over to
track by landmark id did not work: the arrival test started firing on the wrong detection
almost immediately (reporting "arrived at 0.85 m" while the target was 5.4 m away), and the
robot stopped in the lobby. The map is committed and used to accumulate knowledge; the
approach loop still tracks by position. Connecting the two is unfinished work.

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
