# pyunto-robotics

A humanoid robot in a simulated office that you control by sending natural-language messages to
its **Pyunto** account.

> Send "オフィスのドアを開けて" from the Pyunto app → the robot understands the instruction,
> finds the door from camera images alone (no pre-built map), walks to it, and opens it.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Project scaffolding | done |
| 1 | Pyunto client (login, E2EE messages, realtime) | done |
| 2 | Custom humanoid + office simulation (MuJoCo) | next |
| 3 | Perception + mapless navigation | |
| 4 | Instruction understanding (Gemma 4 E2B) | |
| 5 | End-to-end integration | |
| 6 | (optional) RL locomotion | |

## Setup

```bash
uv venv --python 3.12 .venv        # 3.12 required: mlx-vlm does not support 3.13+
VIRTUAL_ENV=$PWD/.venv uv pip install -e .
cp .env.example .env               # then fill in PYUNTO_PASSWORD
```

## Verify the Pyunto connection

```bash
./.venv/bin/python scripts/test_comms.py                # list spaces, threads, read history
./.venv/bin/python scripts/test_comms.py --send "hello" # post a message
./.venv/bin/python scripts/test_comms.py --listen       # stream incoming messages
./.venv/bin/python scripts/test_comms.py --join CODE    # join a space via invite code

./.venv/bin/python -m pytest tests/ -q
```

## Design

Three loops at different rates, so the language model never sits in the control path:

```
Pyunto (Socket.IO)
      │  instruction text
      ▼
 ~1Hz    Gemma 4 E2B (MLX)      "open the door" → plan: [find(door), approach, open, report]
      ▼
 ~5Hz    perception + mapless nav   RGB-D → where is the door → velocity command
      ▼
200Hz    gait controller            step(vx, vy, wz)     ← the one interface that matters
      ▼
         MuJoCo 3.11
```

`step(vx, vy, wz)` is the seam: the gait behind it can be kinematic or a trained RL policy, and
nothing above it changes.

## Notes on the Pyunto backend

The server is Node/TypeScript + Express + Socket.IO (not FastAPI, despite older docs), and
`pyunto-server/api.md` is out of date. Behaviour this client depends on:

- Login returns `token` (not `access_token`); no refresh endpoint, so re-login on 401.
- Message bodies are AES-256-GCM encrypted with a per-space key from
  `GET /api/chat-spaces/:uuid/key`. That endpoint returns the key in the clear and is slated for
  removal in E2EE Phase 3 — hence the `SpaceKeyProvider` seam in `comms/keys.py`.
- REST is snake_case; Socket.IO payloads are camelCase inside `data`.
- The server echoes your own posts back to you — the listener filters on sender uuid, otherwise
  the robot would reply to itself forever.
- Thread membership (not space membership) governs visibility.

## Layout

```
pyunto_robotics/
  comms/      auth, crypto (AES-256-GCM), keys, client (REST + Socket.IO)
  sim/        robot model, gait                    (phase 2)
  perception/ grounding, depth                     (phase 3)
  nav/        mapless exploration                  (phase 3)
  brain/      planner, skills                      (phase 4)
assets/       MJCF: humanoid + office              (phase 2)
scripts/      test_comms.py, view_sim.py, run_robot.py
tests/
```
