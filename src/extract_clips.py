"""
Copy only the selected clips out of Common Voice, ready to upload to Google Drive
for Phase 3. Works from an extracted corpus folder OR straight from the .tar.gz,
so you never have to unpack the whole English release.

    # from an extracted folder (the one containing clips/)
    python src/extract_clips.py --cv-dir data/raw/cv-corpus-23.0/en --zip

    # straight from the downloaded archive (one slow pass, no full extraction)
    python src/extract_clips.py --archive ~/Downloads/<download>.tar.gz --zip

Output: data/audio_subset/<clip>.mp3 (git-ignored), plus audio_subset.zip with --zip.
"""

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def from_folder(cv_dir: Path, wanted: set, out: Path) -> set:
    found = set()
    for name in sorted(wanted):
        src = cv_dir / "clips" / name
        if src.exists():
            shutil.copy2(src, out / name)
            found.add(name)
    return found


def from_archive(archive: Path, wanted: set, out: Path) -> set:
    found = set()
    # "r|*" streams the archive front to back, so memory stays small.
    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and "/clips/" in member.name and name in wanted:
                with tar.extractfile(member) as src, open(out / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                found.add(name)
                if len(found) % 250 == 0:
                    print(f"  {len(found):,}/{len(wanted):,} clips extracted")
                if found == wanted:
                    break
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cv-dir", help="Extracted corpus folder that contains clips/")
    src.add_argument("--archive", help="The downloaded .tar.gz")
    p.add_argument("--selection", default=str(PROJECT_ROOT / "data" / "clips_selected.csv"))
    p.add_argument("--out", default=str(PROJECT_ROOT / "data" / "audio_subset"))
    p.add_argument("--zip", action="store_true", help="Also write <out>.zip for Drive upload")
    args = p.parse_args()

    wanted = set(pd.read_csv(args.selection, dtype=str)["clip_id"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {len(wanted):,} clips to {out} ...")

    if args.cv_dir:
        found = from_folder(Path(args.cv_dir), wanted, out)
    else:
        found = from_archive(Path(args.archive).expanduser(), wanted, out)

    missing = wanted - found
    print(f"Done: {len(found):,} clips copied, {len(missing):,} missing.")
    if missing:
        print("  e.g. " + ", ".join(sorted(missing)[:5]))
    if args.zip:
        zip_path = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"Wrote {zip_path}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
