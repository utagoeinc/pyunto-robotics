# Pyunto Robotics

Message a robot from the Pyunto diary app, and watch it act.

```bash
pip install pyunto-robotics
pyunto-robotics demo --pair K3F9QZ
```

A window opens with a humanoid standing in an office. Write "walk to the door" in the diary on
your phone, and it walks to the door and tells you what it did.

The robot runs on **your** computer. Pyunto never sees the room, the camera, or anything the
robot does — the diary is end-to-end encrypted, and decryption happens on your machine.

---

## The demonstration, step by step

1. In the Pyunto app, open a premium space and choose **Invite a robot**. A six-character
   pairing code appears.
2. On your computer:
   ```bash
   pyunto-robotics demo --pair <code>
   ```
3. Open that space in the app once. Because the diary is end-to-end encrypted, a member has to
   let the robot in — the server cannot hand out a key on its own. The robot says hello when it
   is through.
4. Write an entry. The robot acts and replies in the same thread.

Four robots and four worlds ship with the package:

| `--robot` | Machine | World |
|---|---|---|
| `office` (default) | H1 humanoid | three-room office, opens doors |
| `home` | Momo humanoid | laundry room, empties a washer and folds |
| `patrol` | Q1 quadruped | outdoor site, walks a perimeter and climbs steps |
| `lunar` | R1 rover | lunar south pole, drives cratered regolith |

```bash
pyunto-robotics robots      # what is installed
pyunto-robotics whoami      # this robot's account and the spaces it is in
```

---

## What the robot tells you

A robot that accepts an instruction, goes quiet, and posts one sentence a minute later is
indistinguishable from a robot that has crashed. So it narrates, in the same thread the
instruction arrived in.

Write 「オフィスのドアを開けて」 and the diary fills in as it happens:

```
🤖 Understood: “オフィスのドアを開けて”
   I will: goto door → open door
✅ goto door — Walked to the door. It is 0.4 m ahead. (38s)
✅ open door — Pushed it to 109 degrees and went through.
[a photograph of the room beyond the door]
```

Three things are worth noticing.

The plan arrives **before the robot moves**, so a misunderstanding is caught in the two
seconds before it walks off, not after. Each step reports **as it finishes**, with ⚠️ rather
than ✅ when it did not work and a plain sentence saying why. And the picture at the end is
the robot's own camera, so a claim that a door is open comes with the evidence.

An instruction the robot cannot parse is answered too, rather than ignored:

```
🤖 I did not understand “make me a coffee”.
   I know how to: describe, face, goto, home, lower_arm, open, raise_arm, wave, where…
```

This is the most common outcome of all, and the one where silence does the most damage.

Simple gestures work, which is what people try first:

| You write | The robot does |
|---|---|
| 「右手を挙げて」 / "raise your right hand" | lifts the right hand and holds it up |
| 「左手を振って」 / "wave your left hand" | raises the left arm, waves, lowers it |
| 「手を下ろして」 / "put your hand down" | returns the arm to its side |

Add `--no-photos` to report in words only.

---

## Bringing your own robot

The SDK is not four robots; it is a way to attach *any* robot to a diary. One class, one method:

```python
from pyunto_robotics.api import SkillResult

class MyRobot:
    def run(self, action, argument=None, where=None, expect=None) -> SkillResult:
        if action == "goto":
            ok = my_control_stack.move_to(argument)
            return SkillResult(ok, f"I went to {argument}." if ok
                                   else f"I could not reach {argument}.")
        return SkillResult(False, f"I do not know how to '{action}' yet.")
```

That is the whole contract. You get the encrypted transport, space membership, message
handling, planning and replies; you write what your machine does. It works for real hardware,
for another simulator (Newton, Isaac, Gazebo), or for a robot that is only an HTTP API.

A runnable version is in [`examples/my_robot.py`](examples/my_robot.py) — about forty lines.
The full contract, including the optional `RobotBody` interface for reusing our navigation and
door-opening skills on your own machine, is documented in
[`pyunto_robotics/api.py`](pyunto_robotics/api.py).

To ship your robot as a package others can install, declare an entry point:

```toml
[project.entry-points."pyunto_robotics.robots"]
acme = "acme_robot:setup"
```

After `pip install acme-robot`, `pyunto-robotics demo --robot acme` works with no change here.

---

## Requirements

- macOS on Apple silicon (Windows and Linux are not verified yet)
- Python 3.11 or newer
- The Pyunto app, and a premium space to invite the robot into

The simulator window is owned by `mjpython` on macOS; `pyunto-robotics` re-executes itself under
it automatically, so the command above works as typed.

Optional extras:

```bash
pip install 'pyunto-robotics[llm]'   # plan with a local language model (Apple silicon)
```

Without it, instructions are understood by rule matching in English and Japanese, which covers
the examples above and needs no model download.

---

## What is private, and what is not

- Diary entries are end-to-end encrypted. They are decrypted **on the computer running the
  robot** and nowhere else. Pyunto's servers cannot read them.
- Whoever controls that computer can read everything written in that space. The app says so
  when you invite a robot, and again in the diary where every member can see it.
- The robot only ever reads the spaces it was invited to.
- Its account and keys live in `~/.pyunto-robot`. Delete that and it becomes a different robot,
  and has to be invited again.

## Licence

The Pyunto robots, scenes and SDK code are ours. MuJoCo (Apache-2.0) is a dependency. The
optional Asimov-1 model fetched by `scripts/fetch_asimov.py` is third-party (CERN-OHL-S-2.0)
and is not distributed with this package.
