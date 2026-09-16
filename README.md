# Pyunto Robotics

Message a robot from the Pyunto mobile app, and watch it act.

[**Pyunto for iPhone and iPad**](https://apps.apple.com/app/id6755097890) ·
[**Pyunto for Android**](https://play.google.com/store/apps/details?id=com.pyunto.app)

Install it, scan a square with the app, and a window opens on a robot parked in a carport with
a solar panel on its back. Write "go and find some sunlight, and bring back power" in the diary
on your phone, and it drives out, finds sunlight by measuring what the panel receives, charges,
comes home, and turns the house lights on with what it collected.

The robot runs on **your** computer. Pyunto never sees the room, the camera, or anything the
robot does — the diary is end-to-end encrypted, and decryption happens on your machine.

---

## Quick start

**Use Python 3.11 or 3.12.** `mlx-vlm`, which runs the language model, has no build for 3.13
or newer — and on those versions the install below quietly skips it rather than failing, so
the first sign of trouble is the second command refusing to run.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
```

Then three commands, and the only one to remember is the first:

```bash
pip install 'pyunto-robotics[llm] @ git+https://github.com/utagoeinc/pyunto-robotics'
python -m pyunto_robotics.download_model   # so the robot reads what you write (~5.5 GB, once)
pyunto-robotics showqr                     # a square appears in the terminal
```

Not on PyPI yet, so the install comes from git — one command either way. It brings
`pyunto-agent` with it.

Scan that square with the Pyunto app. The app asks which diary to let the robot into and shows
who runs it; when you approve, the terminal asks which robot to open and starts it — no second
command, nothing to copy back.

```
waiting for the scan… (Ctrl-C to stop)
paired ✓

Which robot would you like to open?
  1. solar    S1 (solar errand robot)
             e.g. "go and find some sunlight, and bring back power"
  2. hotel    H1 (hotel cleaner)
  ...
Number, or a name [1-6, Enter for 1]:

Opening S1 (solar errand robot):
    pyunto-robotics demo --robot solar

listening — message the robot from the Pyunto app. Ctrl-C to stop.
```

It names the command it is running, so opening the same robot again later is a matter of
copying that line. `--robot solar` on `showqr` skips the question entirely.

Then write in the diary, in your own words:

```
go and find some sunlight, and bring back power
```

The robot says how it understood you, drives out, finds the sun by measuring, charges, comes
home, and turns the house lights on — reporting each step in the same thread.

Open that space in the app once after pairing. The diary is end-to-end encrypted, so a member
has to hand the robot a key; the server cannot do it alone.

### Choosing a robot

```bash
pyunto-robotics robots                  # what is installed
pyunto-robotics showqr --robot pet      # pair and open a particular one
pyunto-robotics demo --robot watch      # if already paired
pyunto-robotics whoami                  # this robot's account and its spaces
```

---

## The robots

Six machines and six worlds ship with the package. Every picture below is the scene as it
opens, rendered from the simulator itself.

### `solar` — fetches its own energy

![The S1 parked under a carport, solar panel on its back](docs/images/solar.png)

The demonstration the SDK leads with, and the only one where the robot finds something by
measuring rather than by being told where it is. It drives out, reads what the panel is
receiving as it goes, stops when the measurement says it is in sunlight, charges, comes home,
and turns the house lights on with what it collected.

> *"go and find some sunlight, and bring back power"*
> *"I think we're running low on power"*
> *"how much charge have you got?"*

### `watch` — a house that watches, with no robot in it

![A one-bedroom flat seen from above, an older person asleep in bed, a wall clock](docs/images/watch.png)

There is no machine to command here. The flat itself watches an older person living alone —
floor sensors in each room, a bed sensor, temperature and humidity, the lock, the doorphone —
and writes what it sees into a diary a family member reads from another city.

**No cameras indoors.** The person being watched did not ask to be; the only camera is the
doorphone, and it faces the street. It also turns out to be the better sensor: three
unanswered callers on a day she did not get up is corroboration a motion sensor cannot give.

> *"how is she doing?"*
> *"what has she done today?"*
> *"has anyone been to the door?"*
> *"it feels stuffy in there, can you do something?"*

### `pet` — a camera that has to aim, not just drive

![A flat with a ginger cat on the windowsill and the P1 on its dock](docs/images/pet.png)

The opposite case, and the one that shows why the flat above has no camera. Here the only
human is the one holding the phone, in their own home, looking for their own cat — so a camera
is the right instrument, and the demo is about aiming it.

The cat's four usual places are at four different heights: under the sofa, the windowsill, the
cat tree, the top of the bookshelf. A fixed forward-facing lens finds none of them.

> *"where is the cat?"*
> *"have you seen her anywhere?"*
> *"look around the flat"*
> *"point the camera upwards a bit"*

### `mars` — driving by camera alone

![The R1 rover beside its lander on the Martian surface](docs/images/mars.png)

The clearest demonstration of mapless navigation: there is no map of Mars in the robot, and
the targets are found by looking. Six wheels, a camera mast, and a channel to follow.

> *"drive to the sample"*
> *"head over to that rock"*
> *"how steep is the ground?"*

### `orchard` — four legs and a load

![The Q1 quadruped standing by the shed, an apple tree behind](docs/images/orchard.png)

Walks the rows on four legs and carries a crate. Where the wheeled robots need a surface, this
one handles the ground an orchard actually has.

> *"fetch the crate of apples"*
> *"take them to the shed"*
> *"what are you carrying?"*

### `hotel` — cleans rooms and rides the lift

![The H1 humanoid in a hotel corridor, the lift ahead](docs/images/hotel.png)

Two floors, guest rooms, and a lift the robot has to call, board and ride. The multi-storey
case: getting somewhere is a task in itself, not just a drive.

> *"clean the rooms on both floors"*
> *"clean this corridor"*
> *"which floor are you on?"*

---

## Writing in your own words

There are no commands to learn. Write what you mean, and a local language model reads it:

```
"I think we're running low on power"           ->  find_sun -> goto(park)
"have you seen the cat anywhere?"              ->  patrol
"point the camera upwards a bit"               ->  tilt
"it feels stuffy in there"                     ->  temperature
"the guests have checked out, sort the rooms"  ->  clean
```

None of those are in any list, and that is the point: a keyword table only matches the
phrasings somebody thought to write down, and every miss needs another pattern, in every
language the product ships in. There is no end to that table.

The model runs on your machine, so the diary is never sent anywhere to be understood. This is
the default and needs no flag — run `python -m pyunto_robotics.download_model` once.

It needs Apple silicon. Everywhere else, and until that download has run, the robot matches
commands instead and says so in one line at startup rather than refusing to open.

---

## Command mode

Some sites want the opposite: a closed vocabulary. Equipment with its own command set, an
operator who types the same six instructions all day, a safety case that will not accept a
model deciding what was meant.

```bash
pyunto-robotics demo --robot pet --commands mysite.json
```

```json
{
  "verbs": {
    "find":  ["FIND-TGT", "locate the animal"],
    "photo": ["CAM-SNAP"],
    "home":  ["RTB", "return to base"]
  },
  "objects": {
    "sill": ["POS-03"]
  }
}
```

Only the actions you name are overridden — everything else keeps its built-in phrasings, so
you change the two verbs your equipment spells differently and inherit the rest. Matching is
case-insensitive, so write codes the way your manual writes them. An action the robot does
not have is refused at startup, naming what it does have, rather than becoming a command that
can never fire.

`--command-mode` on its own uses the built-in lists without the model. A runnable example is
in [`examples/commands.example.json`](examples/commands.example.json).

---

## What the robot tells you

A robot that accepts an instruction, goes quiet, and posts one sentence a minute later is
indistinguishable from a robot that has crashed. So it narrates, in the same thread the
instruction arrived in.

This is a real thread, on a phone, after writing "move to the sunlight":

<img src="docs/images/app-thread.png" alt="A Pyunto thread: the robot repeats what it
understood, reports the step with its measurements, and posts a photograph from its own
camera" width="380">

Three things are worth noticing.

The plan arrives **before the robot moves**, so a misunderstanding is caught in the seconds
before it drives off, not after. Each step reports **as it finishes**, with its measurements
(`travelled 16.40 m, irradiance w m2 312`) and ⚠️ rather than ✅ when it did not work. And the
pictures are the robot's own camera, so "I found sunlight 16 m from where I started" comes
with the evidence.

An instruction the robot cannot parse is answered too, rather than ignored:

```
🤖 I did not understand “make me a coffee”.
   I know how to: check, find, go, home, look, pan, patrol, photo, tilt…
```

This is the most common outcome of all, and the one where silence does the most damage.

Each robot answers in its own vocabulary — asking the pet camera to raise its hand gets the
list above, not a shrug. A few that people try first:

| You write | The robot does |
|---|---|
| "where is the cat?" (`pet`) | drives the flat, aims the camera at each of her places, reports where she is |
| "look up" (`pet`) | tilts the lens up without moving the robot |
| "how is she doing?" (`watch`) | reads the sensors and says where she is and whether she is up |
| "has anyone been to the door?" (`watch`) | the day's doorphone callers, and whether she answered |

Add `--no-photos` to report in words only.

---


## Bringing your own robot

The SDK is not six robots; it is a way to attach *any* robot to a diary. One class, one method:

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

### Shipping it as a package

The above is enough to run your own robot on your own machine. This part is only needed if
you want to hand it to *other people* as something they can `pip install`, and have
`pyunto-robotics` discover it without being told.

Say your company is shipping a warehouse vehicle. You would lay the package out like this:

```
mycompany-agv/            ← your project, a separate repository from this one
├── pyproject.toml        ← the file below
└── mycompany/
    └── agv.py            ← your robot: the class above, plus a `setup()` that describes it
```

`pyproject.toml` is the file every Python package has at its root. It tells `pip` the
package's name, its dependencies, and — the part that matters here — what it offers to other
packages. Add this to yours:

```toml
[project.entry-points."pyunto_robotics.robots"]
warehouse-agv = "mycompany.agv:setup"
```

Three parts:

| | |
|---|---|
| `warehouse-agv` | what `--robot` will be called |
| `mycompany.agv` | the module it lives in |
| `setup` | a `RobotSetup`, or anything callable that returns one |

Then, on any machine with both packages installed:

```bash
pip install mycompany-agv
pyunto-robotics robots              # warehouse-agv is in the list
pyunto-robotics demo --robot warehouse-agv
```

You never edit `pyunto-robotics` itself. It asks Python which installed packages have declared
themselves under `pyunto_robotics.robots` and registers whatever it finds, so your robot sits
alongside the bundled six. A plugin that fails to load is logged and skipped rather than
taking the others down with it.

`RobotSetup` is the small record that says what your robot is called, which scene to open (or
none, for real hardware), and which skills to use — see
[`pyunto_robotics/registry.py`](pyunto_robotics/registry.py).

---

## Requirements

- macOS on Apple silicon (Windows and Linux are not verified yet)
- Python 3.11 or newer — but not 3.13+ if you want the language model, which `mlx-vlm` does
  not build for yet
- The Pyunto app, and a premium space to invite the robot into

The simulator window is owned by `mjpython` on macOS; `pyunto-robotics` re-executes itself under
it automatically, so the command above works as typed.

### Installing

Neither package is published to PyPI yet, so both come from git — that is the line in
[Quick start](#quick-start), and it brings `pyunto-agent` with it.

The `[llm]` extra is what lets the robot read sentences rather than match commands, and it
installs nothing at all off Apple silicon, so the same line is safe everywhere. Drop it if you
only ever want [command mode](#command-mode):

```bash
pip install 'pyunto-robotics @ git+https://github.com/utagoeinc/pyunto-robotics'
```

To work on the SDK itself, clone it and install in place:

```bash
git clone https://github.com/utagoeinc/pyunto-robotics
cd pyunto-robotics
python3 -m venv .venv && .venv/bin/pip install -e '.[llm,dev]'
```

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

The Pyunto robots, scenes and SDK code are ours. MuJoCo (Apache-2.0) is a dependency.
