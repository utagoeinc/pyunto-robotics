# pyunto-robotics

Four robots in four simulated worlds, each controlled by messaging its **Pyunto** account.

> Send 「オフィスのドアを開けて」 from the Pyunto app → the robot works out what you meant,
> finds the door from camera images alone with no prior map, walks across the office, pushes it
> open, goes through, and messages you back what happened.

Everything runs on one MacBook. No CUDA, no cloud inference, no pre-built map.

| Robot | World | What it does |
|---|---|---|
| **H1** humanoid, 1.38 m | office, 3 rooms | opens doors, finds a named one of three |
| **Momo** humanoid, 1.41 m | home laundry room | opens a washer, takes a towel out, folds it |
| **Q1** quadruped, 21 kg | outdoor site + building | patrols the perimeter, climbs steps |
| **R1** rover, 6 wheels | lunar south pole | drives cratered regolith at 1.62 m/s² |

All four share one seam — `step(vx, vy, wz)` — so the perception, navigation and agent code is
written once and does not know whether it is driving legs, four legs, or wheels.

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
| 7 | Momo + home laundry room (cloth) | done, one gap |
| 8 | Q1 quadruped + outdoor patrol site | done |
| 9 | R1 rover + lunar south pole | done |

The one gap in phase 7 is named in the [Momo](#momo-and-the-laundry) section below: folding a
towel that has just come out of the drum is not reliable, because it lands bunched. Folding a
towel that is lying flat works every time.

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

### The other three robots

Every robot takes the same flags as `run_robot.py` -- `--say`, `--llm`, `--view`, `--speed`,
`--join`, `--scene`, `--keyframe` -- so anything above works with any of them.

```bash
# ---- Momo, the home assistant ------------------------------------------------
./.venv/bin/mjpython scripts/run_home.py --view --say "洗濯機を開けて"
./.venv/bin/mjpython scripts/run_home.py --view --keyframe counter --say "タオルを畳んで"
./.venv/bin/python   scripts/run_home.py --llm --say "タオルを洗濯機から出して畳んで"

# ---- Q1, the patrol quadruped ------------------------------------------------
./.venv/bin/mjpython scripts/run_patrol.py --view --say "ビルの周りを1周して"
./.venv/bin/mjpython scripts/run_patrol.py --view --say "階段を上って"
./.venv/bin/python   scripts/run_patrol.py --say "2番の地点に行って"

# ---- R1, the lunar rover -----------------------------------------------------
./.venv/bin/mjpython scripts/run_lunar.py --view --speed 6 --say "ビーコンまで行って"
./.venv/bin/mjpython scripts/run_lunar.py --view --speed 6 --say "クレーターの縁まで行って"
./.venv/bin/python   scripts/run_lunar.py --say "周りを見て"

# ---- connected: listen on Pyunto and act on whatever you message --------------
./.venv/bin/mjpython scripts/run_home.py   --view
./.venv/bin/mjpython scripts/run_patrol.py --view
./.venv/bin/mjpython scripts/run_lunar.py  --view
./.venv/bin/python   scripts/run_home.py --join INVITE_CODE   # first time only
```

Starting positions, via `--keyframe`:

| Robot | Keyframes |
|---|---|
| `run_home.py` | `start` (at the washer), `middle` (centre of room), `counter` (at the counter) |
| `run_patrol.py` | `start` (south of the building), `corner` (SE corner), `steps` (at the stairs) |
| `run_lunar.py` | `plain` (open surface, the default), `start` (beside the lander) |

The lunar rover is slow by design -- lunar gravity gives it a sixth of Earth's traction -- so
`--speed 6` makes a drive across the site watchable.

Other tools:

```bash
./.venv/bin/mjpython scripts/view_sim.py --walk        # canned route: ignores the camera,
                                                       # only checks the gait and collisions
./.venv/bin/python scripts/view_sim.py --shot out.png  # stills; plain python is fine
./.venv/bin/python scripts/test_comms.py --listen      # Pyunto connection only
./.venv/bin/python -m pytest tests/ -q                 # 188 tests (~3 min)
./.venv/bin/python -m pytest tests/ -q -m "not slow"   # the fast ones only (~10 s)
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

### Checking what the user said before acting on it

「最初にいる位置からみて、三つ見えるドアのうち、右のドアを…」 states a count, and the count is
checkable. It now reaches the robot: the planner puts it on the step as `expect`, and before
resolving any left/right the robot confirms it can actually see that many. If it cannot, it
repositions; if it still cannot, it says so rather than guessing:

> You said there were 3 doors, but I can only see 1 from here, so I am not sure which one you
> mean.

That matters because "the leftmost" resolved against one visible door is not a smaller answer,
it is a wrong one -- the robot would confidently open something the user did not ask for.
Gemma 4 emits the count once the prompt shows it an example.

### Seeing sideways

The robot has three head cameras, not one: `head_cam` forward, plus `look_left` and
`look_right` turned 60 degrees out with 90-degree fields of view. Between them they cover about
180 degrees. Anything behind is checked by turning the head.

The forward camera alone cannot notice what the robot is brushing against. A wall it is walking
alongside sits at 90 degrees, outside a 75-degree view, so the clearance ahead stays comfortable
the whole way down a corridor while the robot scrapes along the side of it -- measured 568 steps
of wall contact across three errands. `Robot.side_clearance()` reads the two side cameras and
answers "how much room is there either side", which the front camera cannot.

**Steering on it is unfinished.** A keep-off term that turns away from whichever side is closest
does work -- with a 0.85 m personal space, wall contact drops from 568 steps to 115 -- but the
wider berth changes the route enough that the robot loses its target and opens the middle door
when asked for the left one (3/3 correct down to 2/3). At 0.55 m it holds 3/3 but only trims
contact to 513, and it fights the robot at doorways, which are narrow by nature. The cameras and
the measurement are committed; the steering that uses them is not.

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

## Momo and the laundry

A 1.41 m companion robot in a home laundry room, folding towels that are real MuJoCo flex
cloth rather than boxes on a hinge.

The appearance is a product requirement, not decoration: this robot is the face of a consumer
app, and what a user sees first is what it looks like. It is built entirely from MuJoCo
primitives -- spheres, capsules, boxes, ellipsoids, no meshes and nothing loaded from disk --
so "cute" had to come out of proportion. Head radius 0.115 m on a 1.41 m body is about 1:5.4
against roughly 1:7.5 for an adult; the eyes are large, set wide and BELOW the skull's midline,
which is the infant-schema cue that does most of the work; every joint cap is a sphere; and a
single strand of hair stands up, which is the cheapest thing in the model and the one that
stops the silhouette reading as a helmet.

Two things about the appearance were bugs rather than choices, and both are worth knowing:

- **A default-class `rgba` silently overrides every geom's `material`.** The whole robot
  rendered bone white until the default was removed. Defaults set physics; materials set
  appearance, and mixing them loses the materials without an error.
- **`geom_rbound` is a bounding SPHERE.** Solving the standing height from it put the feet
  9 cm off the floor, because for a 0.10 × 0.046 × 0.0225 foot box the bounding sphere is far
  bigger than the box. Measure from `geom_size`.

### Everything the room contains is placed against a measured reach

The arm's *kinematic* span is 0.40 m. Its *working* reach -- how far in front of the base the
hand can actually be PUT -- is 0.25 m, and the two are not the same number:

| commanded forward | hand error after 4 closed-loop passes |
|---|---|
| 0.20 m | 0.023 m |
| 0.25 m | 0.029 m |
| 0.30 m | 0.194 m |
| 0.40 m | 0.230 m |

Past 0.25 m the extended arm loads the torso, the body yields, and the hand settles a fifth of
a metre short *however many correction passes are run* -- the error grew pass over pass rather
than converging. Stiffening the shoulder from kp=150 to kp=500 cut the joint tracking error
from 0.204 rad to 0.065 but did not close the gap, because what remains is the body moving,
not the joint sagging. Reaching further is not a matter of trying harder; it is walking closer.

The hand also bottoms out at z≈0.70 and Momo has no crouch. So every surface in the room sits
inside z=0.70..1.40, and each one moved there because the task failed first:

- the drum opening is at z=0.91, not the realistic height of a front-loader, because at the
  realistic height the towel was simply below the arm;
- the basket stands on a plinth with its floor at z=0.62 -- on the floor, its contents were
  unreachable by 0.40 m;
- the towel starts toward the drum *mouth* rather than centred in the cavity, which is both
  0.07 m more reachable and where laundry actually ends up after a cycle.

### Cloth is grasped by welding a vertex, not by pinching

A towel has no pose. It has 63 vertex positions, and which one the hand is near is the whole
question. Friction does not hold a 4 mm sheet any more than it holds a door handle, so a closed
hand is a weld to one vertex -- the same compromise the office robot makes on doors. Measured:
welding a corner and raising the hand to z=0.72 lifts the sheet to z=0.665.

Which vertex matters. Lifting a towel by its middle gathers it into a bundle; a corner is the
furthest part of the sheet from a robot standing square on. `take_out` grasps the nearest
*perimeter* vertex, `fold` uses corners, and both search a patch of floor for a spot the target
is genuinely reachable from rather than trusting a "stand 0.22 m back and face it" rule -- that
rule kept reporting 0.19 m misses on grasps that work at 0.03 m from a spot 20 cm away.

Three failures worth recording because each looked like something else:

- **The washer cabinet was a solid box**, so the drum's floor and back were buried inside it.
  The towel started embedded in solid matter and was ejected onto the floor on the first step.
  The cabinet is now a shell around the cavity.
- **A closed door flush in its frame is 63 interpenetrating contacts**, and the solver simply
  threw it open -- measured swinging to -97° with nobody touching it. Same fix as the office:
  `<contact><exclude>` the leaf from the world.
- **An open door hangs across its own opening.** Every reach into the drum stopped dead at
  y=1.20 for a towel at y=1.32, the only contact being a finger against the door. The door now
  opens to 105-113° and the robot sidesteps to a spot with a clear line in.

Releasing needs care too: the arm has to move away BEFORE waiting, or the towel drapes over the
forearm and a perfectly aimed drop into the basket reads as a miss.

### Folding, and the one thing that does not work

Folding is measured, not asserted. A flat 0.40 × 0.30 sheet spans about 0.50 m corner to
corner; folded once it should span appreciably less, and `fold` reports both numbers.

The lift height while carrying a corner across turned out to be a narrow window:

| lift | outcome |
|---|---|
| 0.07 m | corner drags, nothing folds |
| 0.10 m | still nothing (span unchanged at 0.50 m) |
| 0.13 m | folds and stays put: **0.50 → 0.28 m**, sheet at z=0.811 |
| 0.16 m | peels the whole sheet off the counter; towel ends on the floor |

That 0.16 m case is the one worth dwelling on, because it *passed*. A towel gathered on the
floor has a small span too -- 0.28 → 0.24 m -- so a span-only test called it a successful fold.
The skill now also checks the towel is still at surface height, which is what a measurement is
for.

Folding is two-handed by nature, and this arm cannot do it that way: the two corners of an edge
are 0.40 m apart, and an exhaustive sweep of standing positions still left the worse hand
0.22 m from its corner against a 0.075 m tolerance. There is no spot where both are in reach,
because the sheet is wider than the span the hands share. So each corner is carried across on
its own and the robot walks between them, which is also what a person does with a bath towel.

**Known limitation.** Folding a towel that has just come out of the drum is not reliable. Every
step individually works -- `open_washer`, `take_out` and `put_on_counter` all pass, and `fold`
is deterministic and repeatable on a flat towel (0.50 → 0.28 m, twice from the `counter`
keyframe) -- but a towel carried out of the washer lands *bunched*, and folding a bunched sheet
drags the gathered mass off the counter instead of folding it. `_spread` was written to flatten
it first and does not do enough. The chain gets three steps in and then reports honestly that
it pulled the towel off the surface. The fix is a proper two-handed spread: pin one corner and
drag the opposite one, which needs the arms to work together in a way nothing else here does.

## Q1 and the patrol

A 21 kg quadruped walking a route around a building, over grass, and up three 0.16 m steps.
Four legs rather than two because the route has stairs: a kinematic biped handles a flat floor
and has nothing sensible to do with a step, whereas four contact points make a stair a question
of foot placement rather than balance.

The trot is kinematic, like the humanoid's gait, and implements the same `Gait` protocol, so
navigation and skills are unchanged. Measured: forward 0.54 m/s against 0.6 commanded, turning
±0.73 rad/s against ±0.8, strafe correct.

Two findings:

- **Writing both the qpos quaternion and `qvel[5]` double-integrates a turn.** A commanded
  +0.8 rad/s over three seconds came out as **-2.63 rad** where +2.40 was wanted -- which reads
  exactly like a sign error and is not. Setting the heading through qpos alone gives +2.400.
  (The humanoid's gait sets both and gets away with it, because its planted stance resists the
  extra rotation.)
- **The trunk must be held at a height above the LOWEST FOOT, not above world zero.** Holding a
  fixed world height makes the robot fight the terrain: on a step it hauls itself back down to
  lawn level. With the ground-relative hold it climbs all three rises and ends on the landing,
  trunk rising 0.365 → 0.831 m.

The patrol route is read from the scene -- sites named `waypoint_1..4` -- so moving the building
in the XML moves the patrol with it. A lap visits all four corners in about 10 s of simulated
time and reports what the camera saw along the way.

Climbing needed one thing the rest of the site does not: **obstacle avoidance turned off for
the last two metres.** The steps *are* an obstacle by any clearance measure, so the avoider slid
the robot sideways along the building and it arrived at x=-2.98 for a staircase at x=0, having
never touched a step. The approach now stops short, squares up, and drives the last stretch
blind -- and the standoff has to sit *inside* the bollard line, or the blind run spends its
whole budget pushing against a post.

## R1 and the Moon

A six-wheeled rocker-bogie rover on 60 × 50 m of cratered regolith at 1.62 m/s². Rocker-bogie
because it is the suspension every planetary rover uses, and for one property: both arms pivot
freely and the two sides are linked by a differential, so all six wheels stay on the ground over
terrain rougher than the wheels are tall.

Measured on flat ground at lunar gravity:

| command | achieved |
|---|---|
| vx +0.6 m/s | 0.57 m/s |
| vx -0.5 m/s | 0.48 m/s |
| wz ±0.5 rad/s | ±0.24 rad/s |

The turn shortfall is not a bug to tune out. A six-wheeler turns by scrubbing its wheels
sideways, the force available to do that is proportional to weight, and on the Moon the rover
weighs a sixth of what it would on Earth. Skills plan around it by closing the loop on heading
rather than assuming a commanded rate arrives.

Getting it to move at all took four fixes, each of which presented as something else:

1. **The rocker and bogie struts were colliding with the ground** and carrying the rover's
   weight, so the wheels barely turned -- 0.19 rad/s against a commanded 3.0, the whole vehicle
   moving 0.04 m/s. Struts are structure, not undercarriage; only wheels touch ground now.
2. **Skid steering alone cannot turn a six-wheeler at 1/6 g.** The two middle wheels sit near
   the turn centre where they cannot roll through the turn, and they ploughed: stalled at
   0.05 rad/s with their actuators saturated at the full 45 N·m. Four corner steering joints
   fixed it, which is what real rovers do for the same reason.
3. **The steering angle and the wheel differential act in opposite senses.** Getting the sign
   wrong makes them cancel almost exactly: +0.5 rad/s commanded arrived as -0.18.
4. **The obstacle threshold was above what empty ground reads.** A mast camera 0.9 m up on
   rolling terrain always has ground in the lower half of its view, so open plain reads about
   2.0 m of forward clearance -- and a 2.2 m threshold declared the rover permanently blocked.

Two things about the terrain are compromises, and both are stated in the XML where they are
made. The relief is 1.4 m rather than 3.2: at 3.2 the crater walls were simply unclimbable for
0.4 m wheels -- dropped into one beside the beacon, the rover managed 0.0-1.4 m in each of the
four compass directions before stopping. And the Sun is lifted to about 20° from the 1-3° that
is physically right at the pole, with a little bounce light added, because the physically
correct version rendered a scene so dark that the craters, the lander and the rover were all
barely discernible to a human eye. What survives the compromise is what matters: light still
arrives from low and to one side, shadows are still long, and crater floors are still much
darker than their rims.

Target positions are measured rather than chosen by eye. The beacon was originally on a crater
rim with 0.78 m of local relief, and a rover that reached it could then drive 0.1 m before
stopping -- arriving stranded it. It now sits on ground with 0.35 m of relief and can leave in
any direction. All three named targets are reachable in one run.

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
  agent.py          message → plan → act → reply (drives any robot's skills)
  runner.py         the shared body of every run_*.py script
  comms/            auth, crypto (AES-256-GCM), keys, client (REST + Socket.IO)
  sim/
    robot.py        step/look/arm/reach -- the boundary everything else talks to
    gait.py         humanoid kinematic gait, RL stub
    quad_gait.py    quadruped trot + closed-form leg IK
    wheel_drive.py  rover skid steering + corner steering
    reach.py        damped least-squares arm IK
    cloth.py        grasping MuJoCo flex cloth by welding a vertex
    terrain.py      procedural heightfields (lawn, lunar regolith)
  perception/       depth geometry, object grounding (colour + VLM)
  nav/              mapless navigation state machine
  brain/
    planner.py      office planner (rules + Gemma 4)
    domains.py      per-robot vocabularies and LLM prompts
    skills.py       office humanoid
    laundry.py      Momo
    patrol.py       Q1
    lunar.py        R1
assets/
  pyunto_h1.xml   humanoid: 1.38 m, 23 actuators, grippers, head camera
  office.xml      3 rooms, corridor, lobby, 3 hinged doors
  momo.xml        companion humanoid: 1.41 m, 24 DoF, neck pitch, face and hair
  home.xml        laundry room: drum washer, raised basket, counter, 2 flex towels
  pyunto_q1.xml   quadruped: 21 kg, 12 DoF, mast cameras incl. a downward one
  campus.xml      building, lawn heightfield, trees, hedges, bollards, 3 steps
  pyunto_r1.xml   rover: 6 wheels, rocker-bogie, differential, 4 corner steers
  lunar.xml       60 × 50 m cratered regolith, lander, beacon, ice, 1.62 m/s²
scripts/          run_robot.py, run_home.py, run_patrol.py, run_lunar.py,
                  view_sim.py, test_comms.py
tests/            188 tests (37 of them for the three newer robots)
```
