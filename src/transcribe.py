"""
Phase 3: transcribe every selected clip with one or more models, saving as it goes.

    python src/transcribe.py --models tiny base small
    python src/transcribe.py --models large-v3 --out-dir /content/drive/MyDrive/accent-bias-asr/results/raw

For each model this writes <out-dir>/<model>.csv (clip_id, model, prediction, seconds),
one row per clip, flushed to disk immediately. Re-running the same command skips
clips that are already done, so a Colab disconnect only loses the clip in progress.
A <model>.json next to it records the exact settings and library versions.

Scoring (normalization + WER) is a separate step: src/score.py. Keeping raw
predictions means scoring can be changed and re-run without a GPU.

Decoding settings (the same for every Whisper model):
  - language="en": the question is how well English is transcribed, so a wrong
    language guess shouldn't be mixed into the error rate;
  - beam_size=5 with temperature fallback: the settings used in the Whisper
    paper and by Whisper's own command-line tool;
  - a fixed random seed per clip (from its ID), so the fallback sampling is
    reproducible and doesn't depend on the order clips are processed in.
"""

import argparse
import csv
import io
import json
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]
# Stretch goal: a non-Whisper model. Trained only on LibriSpeech audiobooks,
# so it's a useful contrast to Whisper's 680k hours of web audio.
HF_MODELS = {"wav2vec2": "facebook/wav2vec2-large-960h-lv60-self"}

WHISPER_DECODE = {
    "language": "en",
    "task": "transcribe",
    "beam_size": 5,
    "best_of": 5,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "condition_on_previous_text": False,
}


def pick_device():
    import torch
    # Apple's MPS backend is unreliable for Whisper, so off-GPU we use the CPU.
    return "cuda" if torch.cuda.is_available() else "cpu"


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        version_file = PROJECT_ROOT / "VERSION"
        return version_file.read_text().strip() if version_file.exists() else None


def read_transcripts(path: Path, repair=False) -> pd.DataFrame:
    """
    Read a <model>.csv, ignoring a half-written last line (left behind if Colab
    disconnects mid-write). Every complete row ends in a newline, so anything
    after the last newline is incomplete. With repair=True the file is trimmed.
    """
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    if repair and complete != data:
        path.write_bytes(complete)
        print(f"  Removed a half-written last line from {path.name}")
    if not complete:
        return pd.DataFrame(columns=["clip_id", "model", "prediction", "seconds"])
    df = pd.read_csv(io.StringIO(complete.decode("utf-8")), dtype=str, keep_default_na=False)
    return df.drop_duplicates("clip_id", keep="last")


# --------------------------------------------------------------------------- #
# Model wrappers: each returns a function  path -> transcript
# --------------------------------------------------------------------------- #

def load_whisper(name, device):
    import torch
    import whisper

    model = whisper.load_model(name, device=device)
    options = dict(WHISPER_DECODE, fp16=(device == "cuda"))

    def run(path, clip_id):
        torch.manual_seed(zlib.crc32(clip_id.encode()))
        return model.transcribe(str(path), verbose=None, **options)["text"].strip()

    return run, {"library": "openai-whisper", "decode_options": options}


def load_wav2vec2(name, device):
    import torch
    import whisper  # only for its ffmpeg-based load_audio (16 kHz mono float32)
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    hf_id = HF_MODELS[name]
    processor = Wav2Vec2Processor.from_pretrained(hf_id)
    model = Wav2Vec2ForCTC.from_pretrained(hf_id).to(device).eval()

    def run(path, clip_id):
        audio = whisper.load_audio(str(path))
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device)).logits
        return processor.batch_decode(logits.argmax(dim=-1))[0].strip()

    return run, {"library": "transformers", "hf_model_id": hf_id, "decoding": "greedy CTC"}


# --------------------------------------------------------------------------- #

def transcribe_model(name, clips, audio_dir: Path, out_dir: Path, device, limit=None):
    out_csv = out_dir / f"{name}.csv"
    done = set()
    if out_csv.exists():
        done = set(read_transcripts(out_csv, repair=True)["clip_id"])
    todo = [c for c in clips if c not in done]
    if limit:
        todo = todo[:limit]
    print(f"\n=== {name}: {len(done):,} already done, {len(todo):,} to go ===")
    if not todo:
        return

    missing = [c for c in todo if not (audio_dir / c).exists()]
    if missing:
        sys.exit(f"{len(missing)} audio files not found in {audio_dir}, e.g. {missing[:3]}")

    t0 = time.time()
    run, settings = (load_wav2vec2 if name in HF_MODELS else load_whisper)(name, device)
    print(f"Loaded {name} on {device} in {time.time() - t0:.0f}s")

    meta = {
        "model": name,
        "device": device,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "versions": {p: package_version(p) for p in
                     ["openai-whisper", "torch", "transformers", "jiwer", "pandas"]},
        **settings,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2, default=list))

    new_file = not out_csv.exists() or out_csv.stat().st_size == 0
    start = time.time()
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["clip_id", "model", "prediction", "seconds"])
        for i, clip_id in enumerate(todo, start=1):
            t = time.time()
            text = " ".join(run(audio_dir / clip_id, clip_id).split())  # one line per row
            writer.writerow([clip_id, name, text, f"{time.time() - t:.2f}"])
            f.flush()
            if i % 100 == 0 or i == len(todo):
                rate = (time.time() - start) / i
                eta = rate * (len(todo) - i) / 60
                print(f"  {i:,}/{len(todo):,}  {rate:.2f}s/clip  ~{eta:.0f} min left", flush=True)

    # Free GPU memory before the next model
    del run
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True,
                   help=f"Any of: {' '.join(WHISPER_MODELS + list(HF_MODELS))}")
    p.add_argument("--clips", default=str(PROJECT_ROOT / "data" / "clips_selected.csv"))
    p.add_argument("--audio-dir", default=str(PROJECT_ROOT / "data" / "audio_subset"))
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "raw"))
    p.add_argument("--limit", type=int, default=None,
                   help="Only do this many new clips per model (for a quick test)")
    args = p.parse_args(argv)

    unknown = [m for m in args.models if m not in WHISPER_MODELS and m not in HF_MODELS]
    if unknown:
        sys.exit(f"Unknown model(s): {unknown}. Choose from {WHISPER_MODELS + list(HF_MODELS)}")

    clips = pd.read_csv(args.clips, dtype=str)["clip_id"].tolist()
    # Fixed shuffled order (by a hash of the clip ID) so a run that stops early
    # still covers every accent group, and GPU speed changes over a session
    # can't line up with one group.
    clips.sort(key=lambda c: zlib.crc32(c.encode()))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    if device == "cpu":
        print("No GPU found: running on CPU (fine for tiny/base tests, slow for bigger models).")

    for name in args.models:
        transcribe_model(name, clips, Path(args.audio_dir), out_dir, device, args.limit)
    print("\nDone. Next: python src/score.py")


if __name__ == "__main__":
    main()
