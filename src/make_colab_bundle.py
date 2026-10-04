"""
Pack everything Colab needs for Phase 3 into one zip to upload to Google Drive.

    python src/make_colab_bundle.py

Writes colab_bundle.zip (git-ignored) containing the Phase 3 scripts,
data/clips_selected.csv, the selected audio from data/audio_subset/, and a
VERSION file with the current git commit so results can be traced to the code.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["src/transcribe.py", "src/score.py", "src/accent_groups.py"]


def main():
    clips = pd.read_csv(ROOT / "data" / "clips_selected.csv", dtype=str)["clip_id"]
    audio_dir = ROOT / "data" / "audio_subset"
    missing = [c for c in clips if not (audio_dir / c).exists()]
    if missing:
        sys.exit(f"{len(missing)} clips missing from {audio_dir}. Run src/extract_clips.py first.")

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *SCRIPTS], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            commit += "-modified"
            print("Note: scripts have uncommitted changes; commit them so results trace to code.")
    except Exception:
        commit = "unknown"

    out = ROOT / "colab_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # mp3s don't compress further
        for s in SCRIPTS:
            z.write(ROOT / s, s)
        z.write(ROOT / "data" / "clips_selected.csv", "data/clips_selected.csv")
        for c in clips:
            z.write(audio_dir / c, f"data/audio_subset/{c}")
        z.writestr("VERSION", commit + "\n")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.0f} MB, {len(clips):,} clips, code {commit})")


if __name__ == "__main__":
    main()
