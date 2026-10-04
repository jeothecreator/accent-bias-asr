"""
Phase 3 scoring: normalize text, compute WER per clip, and combine every model
into one results file.

    python src/score.py                 # score everything in results/raw/
    python src/score.py --show 10       # also print 10 random clips per model to check by hand

Reads results/raw/<model>.csv (from transcribe.py) and data/clips_selected.csv.
Writes results/results.csv with one row per clip x model.

Both the reference and the prediction go through Whisper's EnglishTextNormalizer
before scoring, so "Mr." vs "mister", "colour" vs "color", "$20" vs "twenty
dollars" and punctuation/case differences don't count as errors.

WER per clip = (substitutions + deletions + insertions) / words in the reference.
The per-clip error counts are saved too, so Phase 4 can compute pooled WER
(total errors / total reference words), which isn't dominated by short clips.
"""

import argparse
import sys
from pathlib import Path

import jiwer
import pandas as pd
from whisper.normalizers import EnglishTextNormalizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe import HF_MODELS, WHISPER_MODELS, read_transcripts  # noqa: E402
import accent_groups as ag  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ORDER = WHISPER_MODELS + list(HF_MODELS)

normalize = EnglishTextNormalizer()


def score_pair(reference: str, prediction: str) -> dict:
    ref = normalize(reference)
    pred = normalize(prediction)
    n_ref = len(ref.split())
    if n_ref == 0:
        return {"reference_norm": ref, "prediction_norm": pred, "wer": float("nan"),
                "substitutions": 0, "deletions": 0, "insertions": len(pred.split()),
                "ref_words": 0}
    out = jiwer.process_words(ref, pred)
    return {"reference_norm": ref, "prediction_norm": pred, "wer": out.wer,
            "substitutions": out.substitutions, "deletions": out.deletions,
            "insertions": out.insertions, "ref_words": n_ref}


def score_all(raw_dir: Path, clips: pd.DataFrame) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"), key=lambda f: (
        MODEL_ORDER.index(f.stem) if f.stem in MODEL_ORDER else 99, f.stem))
    if not files:
        sys.exit(f"No transcripts found in {raw_dir}. Run src/transcribe.py first.")

    frames = []
    for f in files:
        raw = read_transcripts(f)
        merged = clips.merge(raw[["clip_id", "model", "prediction"]], on="clip_id", how="inner")
        scores = pd.DataFrame([score_pair(r, p) for r, p in
                               zip(merged["sentence"], merged["prediction"])], index=merged.index)
        frames.append(pd.concat([merged, scores], axis=1))
        status = "complete" if len(merged) == len(clips) else f"INCOMPLETE ({len(merged):,}/{len(clips):,})"
        print(f"  {f.stem:<10} {status}")

    res = pd.concat(frames, ignore_index=True)
    return res.rename(columns={"sentence": "reference"})[[
        "clip_id", "accent_group", "speaker", "model", "reference", "prediction",
        "reference_norm", "prediction_norm", "wer", "substitutions", "deletions",
        "insertions", "ref_words"]]


def summary_table(res: pd.DataFrame) -> pd.DataFrame:
    """Pooled WER (%) per group x model: total errors / total reference words."""
    res = res.assign(errors=res["substitutions"] + res["deletions"] + res["insertions"])
    g = res.groupby(["accent_group", "model"])[["errors", "ref_words"]].sum()
    table = (100 * g["errors"] / g["ref_words"]).unstack("model").round(1)
    table = table[[m for m in MODEL_ORDER if m in table.columns]]
    return table.loc[[k for k in ag.GROUPS if k in table.index]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default=str(PROJECT_ROOT / "results" / "raw"))
    p.add_argument("--clips", default=str(PROJECT_ROOT / "data" / "clips_selected.csv"))
    p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "results.csv"))
    p.add_argument("--show", type=int, default=0,
                   help="Print this many random clips per model for a hand check")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    clips = pd.read_csv(args.clips, dtype=str, keep_default_na=False)
    print("Scoring transcripts:")
    res = score_all(Path(args.raw_dir), clips)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    print("\nPooled WER (%) by accent group and model "
          "(a first look; confidence intervals come in Phase 4):\n")
    print(summary_table(res).to_string())

    empty = res[res["prediction_norm"] == ""].groupby("model").size()
    if len(empty):
        print("\nClips with an empty transcript: " +
              ", ".join(f"{m} {n}" for m, n in empty.items()))

    if args.show:
        for model, sub in res.groupby("model", sort=False):
            print(f"\n--- {model}: {args.show} random clips ---")
            for _, r in sub.sample(min(args.show, len(sub)), random_state=args.seed).iterrows():
                print(f"[{r['accent_group']}] WER {r['wer']:.2f}")
                print(f"  ref:  {r['reference_norm']}")
                print(f"  pred: {r['prediction_norm']}")

    print(f"\nWrote {args.out} ({len(res):,} rows)")
    return res


if __name__ == "__main__":
    main()
