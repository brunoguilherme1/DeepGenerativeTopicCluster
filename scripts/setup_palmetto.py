#!/usr/bin/env python
"""Downloads and installs real Palmetto + the Wikipedia_bd coherence
index, so `--protocol ecrtm_hicot`'s C_V (metrics/palmetto.py::
palmetto_cv) stops reporting N/A (see docs/methodological_notes.md #10:
without this, `ecrtm_hicot` runs correctly record C_V as unavailable
rather than silently substituting a different number - this script is
what makes it actually available).

Installs exactly the two artifacts metrics/palmetto.py's own
DEFAULT_JAR/DEFAULT_WIKI_INDEX already point at - nothing to configure
in code after this script finishes:

    tools/palmetto/palmetto.jar             <- the jar itself (~5.9MB,
                                                checked into
                                                HoangTran223/HiCOT's own
                                                repo directly, no build
                                                step needed)
    tools/palmetto/wiki_data/wikipedia_bd/  <- the Wikipedia coherence
                                                index (a Lucene index,
                                                ~5.1GB compressed,
                                                official DICE/AKSW host -
                                                verified reachable and
                                                sized via a HEAD request
                                                before writing this
                                                script, not guessed)

Idempotent: re-running skips whatever is already present and verified,
unless --force is given. Prints a step-by-step debug trail throughout
(download URL, bytes-so-far, where each file landed, extraction
progress) since this is expected to run unattended on a Colab/SLURM
session someone is watching remotely, not interactively.

Usage:
    python scripts/setup_palmetto.py
    python scripts/setup_palmetto.py --jar-only
    python scripts/setup_palmetto.py --wiki-only
    python scripts/setup_palmetto.py --force
    python scripts/setup_palmetto.py --keep-zip   # don't delete the 5.1GB
                                                   # zip after extracting
                                                   # (e.g. to move it to
                                                   # Google Drive after)

Run from the repository root (same convention as scripts/run_experiment.py) -
paths above are relative to the current working directory, matching
metrics/palmetto.py's own DEFAULT_JAR/DEFAULT_WIKI_INDEX exactly.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

JAR_URL = "https://raw.githubusercontent.com/HoangTran223/HiCOT/main/evaluations/pametto.jar"
WIKI_ZIP_URL = "https://files.dice-research.org/users/mroeder/palmetto/Wikipedia_bd.zip"
# From a HEAD request against WIKI_ZIP_URL (2026-09) - not enforced as a
# hard integrity check (a mirror could legitimately serve a different
# exact byte count), only used to size the progress log / disk-space
# warning sensibly when the server doesn't report Content-Length itself.
WIKI_ZIP_APPROX_BYTES = 5_484_636_886

TOOLS_DIR = Path("tools/palmetto")
JAR_PATH = TOOLS_DIR / "palmetto.jar"
WIKI_DATA_DIR = TOOLS_DIR / "wiki_data"
# Matches metrics/palmetto.py::DEFAULT_WIKI_INDEX exactly. Verified via
# the zip's own central directory (an HTTP-range read of the remote
# file, without downloading it in full) that extracting the archive
# produces a top-level `wikipedia_bd/` dir plus a `wikipedia_bd.
# histogram` sibling file - both land here when extracted to
# WIKI_DATA_DIR.
WIKI_INDEX_DIR = WIKI_DATA_DIR / "wikipedia_bd"


def _log(msg: str) -> None:
    print(f"[setup_palmetto] {msg}", flush=True)


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _check_java() -> bool:
    _log("Checking for a Java runtime (`java -version`)...")
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result = None

    # A missing `java` doesn't always raise FileNotFoundError - macOS's
    # own `java` shim runs fine (no exception) but prints "Unable to
    # locate a Java Runtime" to stderr and exits non-zero. Check
    # returncode too, not just whether the subprocess call itself raised.
    if result is None or result.returncode != 0:
        _log(
            "Java not found on PATH. Palmetto needs a JVM to actually COMPUTE coherence later "
            "(not needed for this download step itself). On Colab/Debian/Ubuntu: "
            "`apt-get install -y default-jre-headless`."
        )
        return False

    output = (result.stderr or result.stdout).strip()
    version_line = output.splitlines()[0] if output else "(no output)"
    _log(f"Found Java: {version_line}")
    return True


def _disk_space_check(path: Path, needed_bytes: int) -> None:
    check_path = path if path.exists() else path.parent
    check_path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(check_path)
    _log(f"Free disk space at {check_path}: {_human_bytes(usage.free)} (this step needs ~{_human_bytes(needed_bytes)})")
    if usage.free < needed_bytes:
        _log(
            f"WARNING: free space ({_human_bytes(usage.free)}) looks smaller than what this step needs "
            f"(~{_human_bytes(needed_bytes)}) - continuing anyway, but this may fail partway through. "
            "Consider --keep-zip=False (the default) so the zip is removed right after extraction, or "
            "free up space first."
        )


def _download(url: str, dest: Path, label: str, approx_bytes: int = 0) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Downloading {label} from {url}")
    start = time.perf_counter()
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0) or approx_bytes
        downloaded = 0
        next_log_at = 0
        with open(tmp_dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_log_at:
                    pct = f" ({100 * downloaded / total:.1f}%)" if total else ""
                    total_str = f" / {_human_bytes(total)}" if total else ""
                    _log(f"  {label}: {_human_bytes(downloaded)}{total_str}{pct} downloaded")
                    next_log_at = downloaded + 200 * 1024 * 1024  # log roughly every 200MB
    tmp_dest.rename(dest)
    elapsed = time.perf_counter() - start
    _log(f"Finished downloading {label}: {_human_bytes(dest.stat().st_size)} in {elapsed:.1f}s -> {dest}")


def setup_jar(force: bool = False) -> bool:
    if JAR_PATH.exists() and not force:
        _log(f"palmetto.jar already present at {JAR_PATH} ({_human_bytes(JAR_PATH.stat().st_size)}) - "
             "skipping (pass --force to re-download).")
        return True
    _download(JAR_URL, JAR_PATH, "palmetto.jar")
    return True


def setup_wiki_index(force: bool = False, keep_zip: bool = False) -> bool:
    if WIKI_INDEX_DIR.is_dir() and any(WIKI_INDEX_DIR.iterdir()) and not force:
        _log(f"Wikipedia index already present at {WIKI_INDEX_DIR} - skipping "
             "(pass --force to re-download/re-extract).")
        return True

    zip_path = TOOLS_DIR / "Wikipedia_bd.zip"
    _disk_space_check(TOOLS_DIR, WIKI_ZIP_APPROX_BYTES * 2)  # zip + extracted, roughly, at peak

    if zip_path.exists() and not force:
        _log(f"Found existing {zip_path} ({_human_bytes(zip_path.stat().st_size)}) - reusing instead of "
             "re-downloading (pass --force to re-download).")
    else:
        _download(WIKI_ZIP_URL, zip_path, "Wikipedia_bd.zip", approx_bytes=WIKI_ZIP_APPROX_BYTES)

    zip_size = zip_path.stat().st_size
    _log(f"Extracting {zip_path} ({_human_bytes(zip_size)}) to {WIKI_DATA_DIR} - "
         "this can take a few minutes for an archive this size...")
    WIKI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        _log(f"Archive contains {len(members)} entries (expects a top-level 'wikipedia_bd/' index dir "
             "plus a 'wikipedia_bd.histogram' sibling file).")
        for i, member in enumerate(members, start=1):
            zf.extract(member, WIKI_DATA_DIR)
            if i % 20 == 0 or i == len(members):
                _log(f"  extracted {i}/{len(members)} entries...")
    _log(f"Extraction finished -> {WIKI_DATA_DIR}")

    if keep_zip:
        _log(f"Keeping {zip_path} as requested (--keep-zip).")
    else:
        _log(f"Removing {zip_path} to free disk space (pass --keep-zip to keep it instead, "
             "e.g. to move it to Google Drive for reuse across sessions).")
        zip_path.unlink()

    if not WIKI_INDEX_DIR.is_dir():
        _log(
            f"ERROR: expected {WIKI_INDEX_DIR} to exist after extraction but it does not - the archive's "
            f"internal layout may have changed upstream. Check what actually got extracted under {WIKI_DATA_DIR}."
        )
        return False
    _log(f"Wikipedia index verified at {WIKI_INDEX_DIR}.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jar-only", action="store_true", help="Only download palmetto.jar, skip the Wikipedia index.")
    parser.add_argument("--wiki-only", action="store_true", help="Only download/extract the Wikipedia index, skip the jar.")
    parser.add_argument("--force", action="store_true", help="Re-download/re-extract even if already present.")
    parser.add_argument("--keep-zip", action="store_true", help="Don't delete Wikipedia_bd.zip after extracting.")
    args = parser.parse_args()

    _log("Starting Palmetto setup - see docs/methodological_notes.md #10 for what this unblocks "
         "(--protocol ecrtm_hicot's C_V, currently N/A without it).")

    _check_java()

    ok = True
    if not args.wiki_only:
        ok = setup_jar(force=args.force) and ok
    if not args.jar_only:
        ok = setup_wiki_index(force=args.force, keep_zip=args.keep_zip) and ok

    from vaebm_benchmark.metrics.palmetto import palmetto_available

    available = palmetto_available()
    _log(f"palmetto_available() -> {available}")
    if ok and available:
        _log("Palmetto is ready. Re-run your --protocol ecrtm_hicot command - C_V should now compute for "
             "real instead of N/A.")
    else:
        _log("Palmetto is NOT ready yet - see the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
