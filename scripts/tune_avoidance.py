#!/usr/bin/env python3
"""Search the obstacle-avoidance constants for a setting that stops scraping walls.

    python scripts/tune_avoidance.py --generations 8 --population 16

An evolutionary search rather than reinforcement learning, because the thing being improved
is not a policy the robot executes. The gait writes base velocity directly (sim/gait.py:
"Drive the base by writing VELOCITY, never position"), so there is no torque-to-contact
mapping to learn and nothing for an RL agent to act on. What is actually costing contact is a
handful of hand-set thresholds in nav/explore.py -- how close counts as blocked, how far to
lean off a wall, how long to commit to a detour -- tuned one at a time by hand, each in
isolation, on a metric nobody could see until this session added it. That is exactly the
shape of problem a black-box search is for: ten coupled scalars, an expensive but honest
fitness function, no gradient.

Fitness is the errands themselves, and getting it right took two failed searches. Scored
against a single errand, the winner improved that one and made every other route worse.
Scored with completion as a currency, the winner bought a completion by scraping. So: four
errands per evaluation, completion as a floor rather than a score, and contact ranking what
is left. Each evaluation runs in a subprocess, so a candidate that wedges the simulator
cannot take the search down with it, and a held-out set of routes the search never sees is
what decides whether a result is real.

WHAT IT FOUND, run properly: nothing. Six generations, 84 evaluations, and the winner was
the values already in the file, gene for gene. That is a result worth keeping rather than a
wasted afternoon -- the constants each landed by hand, one at a time, on a metric that did
not exist until this session, and it was a fair question whether they were anywhere near
each other's optimum. They are: random genomes that clear the completion floor still scrape
eight times as much (56% of control steps against 7%), so the basin is real and the hand
tuning sits in it.

Which also says where the remaining contact is NOT. Attributing it by call site puts most of
what is left inside the door manipulation itself -- turning on the spot, holding a leaf open,
pulling one clear -- and no threshold in this file reaches that. Re-run this after changing
anything in the avoidance stack; do not expect it to find gains that are not there.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# name -> (low, high). Bounds are wide enough to be interesting and narrow enough to stay
# physical: a standoff wider than a doorway, or a detour that never commits, is not a
# candidate worth spending 37 seconds on.
GENOME: dict[str, tuple[float, float]] = {
    "CLEARANCE_MARGIN_M": (0.02, 0.30),
    "CLEARANCE_DISABLE_M": (1.0, 3.0),
    "DETOUR_TRIGGER_M": (0.5, 1.2),
    "DETOUR_CLEAR_M": (1.0, 2.5),
    "DETOUR_MIN_STEPS": (10, 80),
    "WALL_STANDOFF_M": (0.4, 0.9),
    "WALL_PUSH_OFF": (0.05, 0.35),
    "WALL_REFLEX_BACK": (0.0, 0.35),
    "WALL_REFLEX_STEPS": (8, 40),
    "SIDE_LOOK_DISABLE_M": (1.5, 4.0),
}
INTEGER = {"DETOUR_MIN_STEPS", "WALL_REFLEX_STEPS"}

# What one evaluation runs, in a subprocess. Constants are patched in memory before the
# navigator is imported by anything that captures them at import time.
EVAL = textwrap.dedent("""
    import json, sys, logging
    sys.path.insert(0, {root!r})
    logging.basicConfig(level=logging.CRITICAL)
    genome = json.loads(sys.argv[1])

    import pyunto_robotics.nav.explore as E
    for key, value in genome.items():
        # Step counts index range(); they have to arrive as ints, not as the floats JSON
        # gives back for a whole number.
        setattr(E, key, int(value) if key in {intkeys} else value)

    from pyunto_robotics.sim.robot import Robot
    from pyunto_robotics.brain.skills import Skills
    from pyunto_robotics.perception.grounding import ColorGrounder

    # skills.py binds WALL_REFLEX_BACK at import time, so patching the navigator's module
    # alone leaves the skills-side reflex on the old value.
    import pyunto_robotics.brain.skills as S
    if "WALL_REFLEX_BACK" in genome:
        S.WALL_REFLEX_BACK = genome["WALL_REFLEX_BACK"]

    # Several errands, not one. Tuned against the four-step errand alone, the search found a
    # genome that improved it from 5.2% contact to 4.7% and made everything else worse --
    # 9.2% to 14.8% across errands it had never seen, with one going from 4.9% to 29.6%.
    # Thresholds that suit a single route are not obstacle avoidance, they are a memorised
    # path, and only a fitness that spans routes can tell the two apart.
    CASES = [
        ("errand", "lobby", [("open", ("door", "right", 3)), ("leave", (None,)),
                             ("home", (None,)), ("open", ("door", "left", 3))]),
        ("middle", "lobby", [("open", ("door", "middle", 3))]),
        ("left-out", "lobby", [("open", ("door", "left", 3)), ("leave", (None,))]),
        ("from-start", "start", [("open", ("door", "right", None))]),
    ]

    done = 0
    wanted = 0
    contact = 0
    total = 0
    for _name, keyframe, plan in CASES:
        robot = Robot("office.xml", keyframe=keyframe)
        skills = Skills(robot, ColorGrounder())
        touching = [0]
        steps = [0]
        real = robot.step

        def step(*a, _t=touching, _s=steps, _r=real, _rob=robot, **k):
            _r(*a, **k)
            _s[0] += 1
            if _rob.wall_contact_side() is not None:
                _t[0] += 1

        robot.step = step
        wanted += len(plan)
        for name, args in plan:
            if not skills.run(name, *args).ok:
                break
            done += 1
        robot.close()
        contact += touching[0]
        total += steps[0]
    print(json.dumps({{"done": done, "wanted": wanted,
                       "contact": contact, "steps": total}}))
""")


def evaluate(genome: dict[str, float], timeout: float = 300.0) -> dict:
    """Run one errand with these constants. Never raises: a bad genome scores badly."""
    try:
        out = subprocess.run(
            [sys.executable, "-c", EVAL.format(root=str(ROOT), intkeys=repr(INTEGER)),
             json.dumps(genome)],
            capture_output=True, text=True, timeout=timeout, cwd=ROOT, check=False,
        )
        line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
        if not line:
            return {"done": 0, "wanted": 8, "contact": 10**6, "steps": 1}
        return json.loads(line[-1])
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return {"done": 0, "wanted": 8, "contact": 10**6, "steps": 1}


def fitness(result: dict, baseline_done: int) -> float:
    """Lower is better. Completing at least as much as today, then contact, then time.

    Completion is a floor, not a currency. Scoring it linearly -- 100 points a step against a
    contact rate worth about 10 -- let the search buy a completion by scraping: it found a
    genome that finished 8 of 8 instead of 7 while nearly tripling contact (7.1% to 18.9%),
    which is the opposite of what this is for. So anything that regresses completion is simply
    out, and among the rest the ranking is contact, with a small term on step count to break
    ties toward the brisker route.

    Improving completion beyond the baseline earns a modest credit -- worth having, not worth
    scraping for.
    """
    done = result["done"]
    if done < baseline_done:
        return 10_000.0 + (baseline_done - done)
    rate = result["contact"] / max(result["steps"], 1)
    return rate * 100.0 - (done - baseline_done) * 0.5 + result["steps"] / 100_000.0


def random_genome(rng: random.Random) -> dict[str, float]:
    return {k: _quantise(k, rng.uniform(*bounds)) for k, bounds in GENOME.items()}


def _quantise(key: str, value: float) -> float:
    low, high = GENOME[key]
    value = min(max(value, low), high)
    return float(round(value)) if key in INTEGER else round(value, 4)


def mutate(genome: dict[str, float], rng: random.Random, scale: float) -> dict[str, float]:
    """Gaussian jitter on each gene, sized to that gene's own range."""
    child = {}
    for key, value in genome.items():
        low, high = GENOME[key]
        child[key] = _quantise(key, rng.gauss(value, (high - low) * scale))
    return child


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="avoidance_tuned.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Seed the population with what is in the file today, so the search starts from a known
    # working point and can only be judged an improvement on it.
    import pyunto_robotics.nav.explore as E  # noqa: PLC0415 - needs sys.path set up first
    baseline = {k: float(getattr(E, k)) for k in GENOME}
    population = [baseline] + [random_genome(rng) for _ in range(args.population - 1)]

    # What today's constants complete. Nothing that finishes less than this is a candidate,
    # however little it touches -- a genome that avoids walls by avoiding the errand is not
    # an improvement.
    baseline_done = evaluate(baseline)["done"]
    print(f"baseline completes {baseline_done} steps; candidates must match it", flush=True)

    best: tuple[float, dict, dict] | None = None
    for generation in range(args.generations):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(evaluate, population))
        scored = sorted(zip(population, results),
                        key=lambda pair: fitness(pair[1], baseline_done))

        top_genome, top_result = scored[0]
        if best is None or fitness(top_result, baseline_done) < best[0]:
            best = (fitness(top_result, baseline_done), top_genome, top_result)
        rate = 100.0 * top_result["contact"] / max(top_result["steps"], 1)
        print(f"gen {generation}: best done={top_result['done']}/{top_result.get('wanted', 8)} "
              f"contact={rate:.1f}% steps={top_result['steps']} "
              f"(fitness {fitness(top_result, baseline_done):.2f})", flush=True)

        # Elitist: keep the top quarter, refill by crossover and mutation. The mutation scale
        # decays so early generations explore and later ones settle.
        keep = max(2, args.population // 4)
        parents = [g for g, _ in scored[:keep]]
        scale = 0.25 * (1.0 - generation / max(args.generations - 1, 1)) + 0.05
        population = list(parents)
        while len(population) < args.population:
            a, b = rng.choice(parents), rng.choice(parents)
            population.append(mutate(crossover(a, b, rng), rng, scale))

    assert best is not None
    score, genome, result = best
    print("\nbest genome:")
    for key in GENOME:
        print(f"  {key} = {genome[key]}   (was {baseline[key]})")
    print(f"\ndone={result['done']}/{result.get('wanted', 8)}  contact="
          f"{100.0 * result['contact'] / max(result['steps'], 1):.1f}%  "
          f"steps={result['steps']}  fitness={score:.2f}")
    Path(args.out).write_text(json.dumps(
        {"genome": genome, "result": result, "baseline": baseline}, indent=2))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
