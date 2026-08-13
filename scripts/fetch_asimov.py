#!/usr/bin/env python3
"""Fetch the Asimov 1 simulation model from its public repository.

    python scripts/fetch_asimov.py

Asimov 1 is Menlo Research's open-source humanoid: 1.2 m, 35 kg, 25 actuated joints plus two
passive toes, published with its MuJoCo model at github.com/asimovinc/asimov-1 under
CERN-OHL-S-2.0. This downloads that model and its 28 STL link meshes into assets/asimov/.

Fetched rather than vendored, for two reasons. The meshes are 45 MB of binary, which is what
.gitignore already keeps out of this repository; and a copy of someone else's hardware model
goes stale silently, where a fetch always names its upstream. Nothing here modifies the
upstream file -- the adaptation this project needs (a head camera, a gripper) lives beside it
in assets/asimov_r1.xml, which includes it.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "asimovinc/asimov-1"
API = f"https://api.github.com/repos/{REPO}/contents"
MODEL = "sim-model/xmls/asimov.xml"
MESHES = "sim-model/assets/meshes"
DEST = Path(__file__).resolve().parent.parent / "assets" / "asimov"


def _get(url: str) -> bytes:
    """Fetch a URL, through `gh` when it is available.

    The GitHub API allows 60 anonymous requests an hour and this needs 30, so a second run in
    the same hour fails on rate limit. `gh` carries the user's token and raises that to 5000;
    plain urllib remains the fallback for a machine without it.
    """
    if url.startswith(API) and shutil.which("gh"):
        done = subprocess.run(["gh", "api", url[len("https://api.github.com/"):]],
                              capture_output=True, check=False)
        if done.returncode == 0:
            return done.stdout
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed host
        return response.read()


def _download(path: str, into: Path) -> int:
    """Fetch one repository file. Returns bytes written."""
    entry = json.loads(_get(f"{API}/{path}"))
    if entry.get("content"):
        blob = base64.b64decode(entry["content"])
    else:
        # Files over 1 MB come back without inline content, only a download URL.
        blob = _get(entry["download_url"])
    into.parent.mkdir(parents=True, exist_ok=True)
    into.write_bytes(blob)
    return len(blob)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(DEST), help="where to put the model")
    ap.add_argument("--force", action="store_true", help="re-download files already present")
    args = ap.parse_args()
    dest = Path(args.dest)

    # asimov.xml declares meshdir="../assets/meshes", and MuJoCo resolves that against the
    # file doing the including -- assets/asimov_r1.xml -- not against the included file. So the
    # meshes go where that path lands from there: assets/../assets/meshes, i.e. assets/meshes.
    try:
        model = dest / "xmls" / "asimov.xml"
        if args.force or not model.exists():
            size = _download(MODEL, model)
            print(f"asimov.xml  {size / 1024:.0f} KB")
        else:
            print("asimov.xml  (already here)")

        meshes = json.loads(_get(f"{API}/{MESHES}"))
    except urllib.error.URLError as e:
        print(f"ERROR: could not reach github.com ({e.reason})", file=sys.stderr)
        return 1

    total = 0
    for i, entry in enumerate(meshes, 1):
        target = dest.parent / "meshes" / entry["name"]
        if target.exists() and not args.force:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_get(entry["download_url"]))
        total += entry["size"]
        print(f"  [{i}/{len(meshes)}] {entry['name']}  {entry['size'] / 1024:.0f} KB", flush=True)

    print(f"\n{len(meshes)} meshes in {dest.parent / 'meshes'}"
          + (f", {total / 1048576:.1f} MB fetched" if total else ", all already present"))
    print("\nAsimov 1 model (c) Menlo Research, CERN-OHL-S-2.0.")
    print("Run it with:  python scripts/run_robot.py --scene office_asimov.xml --say '...'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
