"""Compile the stationary Dyssol units of a delivery into shared libraries.

The delivery holds the units as C++ source only (``<local>/partner/<delivery>/Units/<unit>/``,
confidential, never committed). This script builds the three finished units with MSBuild against
the SDK shipped with Dyssol (``<DYSSOL_HOME>/ModelsCreatorSDK``) and writes them, with a manifest
of digests, to ``<runs>/units/``. The fourth unit of the delivery (a heat exchanger with area and
heat-transfer coefficient) is unfinished and not built; no flowsheet of the delivery uses it.

Nothing is written into the Dyssol installation or the delivery: the sources are copied to
``<runs>/build/src`` and intermediate files go to ``<runs>/build/obj``.

Usage: ``tools/run_here.sh scripts/build_dyssol_units.py`` (environment as in
``scripts/run_dyssol_steady.py``; ``$MSBUILD`` names MSBuild.exe if it is not found).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import git  # noqa: E402

from twinlink_ml.dyssol_steady.materials import sha256_of  # noqa: E402
from twinlink_ml.dyssol_steady.study import load_config  # noqa: E402

REPO_ROOT = SCRIPTS.parent
LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", REPO_ROOT / "local"))
DYSSOL_HOME = Path(os.environ.get("DYSSOL_HOME", "C:/Program Files/Dyssol"))
UNITS = ("DryerTwinLink", "HeaterTwinLink", "HeatrecoveryTwinLink")


def find_msbuild() -> Path:
    if os.environ.get("MSBUILD"):
        return Path(os.environ["MSBUILD"])
    root = Path("C:/Program Files/Microsoft Visual Studio")
    found = sorted(root.glob("*/*/MSBuild/Current/Bin/MSBuild.exe"))
    if not found:
        raise SystemExit("MSBuild.exe not found; set MSBUILD")
    return found[-1]


def main() -> None:
    branch = git("rev-parse", "--abbrev-ref", "HEAD") or "detached"
    runs = Path(os.environ.get("TWINLINK_RUNS_DIR", Path("C:/twinlink_runs") / branch))
    source = LOCAL / "partner" / str(load_config().raw["delivery"]) / "Units"
    sdk = DYSSOL_HOME / "ModelsCreatorSDK"
    out = runs / "units"
    manifest_path = out / "manifest.json"
    files = sorted(p for u in UNITS for p in (source / u).iterdir() if p.suffix in {".cpp", ".h"})
    sources = {f"{p.parent.name}/{p.name}": sha256_of(p) for p in files}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["sources"] == sources:
            print(f"{manifest_path} is up to date")
            return
        raise SystemExit(f"{out} was built from other sources -- use a new TWINLINK_RUNS_DIR")
    msbuild = find_msbuild()
    build = runs / "build"
    if build.exists():
        shutil.rmtree(build)
    shutil.copytree(source, build / "src")
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SolutionDir": str(sdk) + "\\"}
    for unit in UNITS:
        command = [
            str(msbuild),
            str(build / "src" / unit / f"{unit}.vcxproj"),
            "/nologo",
            "/v:minimal",
            "/p:Configuration=Release",
            "/p:Platform=x64",
            "/p:BuildProjectReferences=false",
            f"/p:OutDir={build / 'out'}\\",
            f"/p:IntDir={build / 'obj' / unit}\\",
        ]
        done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            print(done.stdout[-4000:])
            raise SystemExit(f"build of {unit} failed")
        shutil.copy2(build / "out" / f"{unit}.dll", out / f"{unit}.dll")
        print(f"built {unit}.dll")
    version = subprocess.run(
        [str(msbuild), "/nologo", "/version"], capture_output=True, text=True, check=False
    )
    manifest = {
        "delivery": str(load_config().raw["delivery"]),
        "sources": sources,
        "libraries": {f"{u}.dll": sha256_of(out / f"{u}.dll") for u in UNITS},
        "msbuild": str(msbuild),
        "msbuild_version": version.stdout.strip(),
        "sdk": str(sdk),
        "configuration": "Release|x64",
        "git/commit": git("rev-parse", "HEAD"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
