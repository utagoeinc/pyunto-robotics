#!/usr/bin/env python3
"""Fetch the language model the robots use to understand what you wrote.

Without it the robots match keywords, which works for the phrasings somebody thought to list
and fails for everything else. Asked to "move somewhere sunny" the keyword matcher went
looking for a landmark called "somewhere sunny", because the phrase says "move to" rather than
"search for" and only the latter was in the table. Every such failure needs a new pattern, and
there is no end to them -- least of all across languages.

With the model, the robot reads the sentence. "I think we're running low on power" becomes an
errand to go and fetch some, and no list contains that.

    python -m pyunto_robotics.download_model     # about 5.5 GB, once
    pyunto-robotics demo                 # the model is used by default

Runs on Apple Silicon. On other hardware the robots keep using the keyword matcher, which is
why `--llm` is a flag rather than the default.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys

# Gemma 4 E2B, 8-bit. The 8-bit build rather than 4-bit because Gemma 4's per-layer embeddings
# quantise badly below it -- 4-bit plans are noticeably worse at picking the right verb. E2B
# rather than a larger model because this runs beside a physics simulation on a laptop.
MODEL_ID = "lmstudio-community/gemma-4-E2B-it-MLX-8bit"
APPROX_GB = 5.5


def supported() -> tuple[bool, str]:
    """Whether this machine can run the model, and why not if it cannot."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False, (
            f"The local model runs on Apple Silicon; this is {platform.system()} "
            f"{platform.machine()}. The robots will use keyword matching instead."
        )
    if sys.version_info >= (3, 13):
        # Say what to do, not only what is wrong. Reaching this message means an environment
        # has already been built on the wrong version -- pip installed the [llm] extra's
        # markers as "nothing to do" and reported success -- so the fix is a new environment,
        # and it is worth spelling out rather than leaving as an exercise.
        return False, (
            f"mlx-vlm has no build for Python {sys.version_info.major}."
            f"{sys.version_info.minor} yet, so the model cannot be installed here.\n"
            "\n"
            "Build the environment on 3.11 or 3.12 instead:\n"
            "    python3.12 -m venv .venv && source .venv/bin/activate\n"
            "    pip install 'pyunto-robotics[llm] @ "
            "git+https://github.com/utagoeinc/pyunto-robotics'\n"
            "\n"
            "The robots still run here without it, matching commands instead of reading "
            "sentences. See `--command-mode` in the README."
        )
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL_ID, help="model to fetch")
    parser.add_argument("--check", action="store_true",
                        help="report whether the model is ready, and download nothing")
    args = parser.parse_args()

    ok, why = supported()
    if not ok:
        print(why)
        return 0 if args.check else 1

    free_gb = shutil.disk_usage(".").free / 1e9
    if not args.check and free_gb < APPROX_GB + 1.0:
        # Say so before starting rather than failing part-way through a 5 GB download.
        print(f"Not enough disk space: {free_gb:.1f} GB free, about "
              f"{APPROX_GB:.1f} GB needed.")
        return 1

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install the model extra first:\n\n    pip install -e '.[llm]'\n")
        return 1

    if args.check:
        try:
            path = snapshot_download(args.model, local_files_only=True)
            print(f"Model is ready: {path}")
            return 0
        except Exception:  # noqa: BLE001 - "not downloaded" is the expected answer here
            print("Model is not downloaded. Run: python -m pyunto_robotics.download_model")
            return 1

    print(f"Downloading {args.model} (about {APPROX_GB:.1f} GB).")
    print("This happens once; it is cached in ~/.cache/huggingface.\n")
    path = snapshot_download(args.model)
    print(f"\nDone: {path}")
    print("\nNow run a robot with --llm:\n")
    print("    pyunto-robotics demo --robot solar --llm\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
