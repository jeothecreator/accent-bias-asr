"""
Phase 2: build the clean, balanced evaluation set from Common Voice English.

    python src/prepare_dataset.py --cv-dir data/raw/<corpus folder>/en

Reads validated.tsv (and clip_durations.tsv if present) and writes:
    data/clips_selected.csv          the final sample: one row per clip
    data/accent_label_mapping.csv    every distinct raw accent label -> status/group
    data/DATA_CARD.md                how the dataset was built, with all counts

Pipeline
--------
 1. Load validated clips.
 2. Explore the accents field (counts per label, number of blanks).
 3. Map each raw label to an accent group (rules in accent_groups.py).
 4. Drop speakers whose clips map to more than one group.
 5. Drop clips with very short sentences (and very short audio, if durations exist).
 6. Sample per group, speaker-balanced and sentence-first:
      - progressive cap: a first pass allows 1 clip per speaker, the next 2,
        and so on up to --max-per-speaker, so every group is spread over as
        many speakers as it has before anyone contributes a second clip;
      - within each pass, sentences read by the most groups are taken first,
        one clip per group, so groups share as many sentences as possible;
      - each group stops at --per-group clips.
 7. Save the sample and write the data card.

Everything random goes through --seed, so the same inputs give the same sample.
"""

import argparse
import csv
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accent_groups as ag  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATUS_ORDER = ["grouped", "blank", "uninformative", "excluded", "unmapped", "conflict"]


# --------------------------------------------------------------------------- #
# 1. Load
# --------------------------------------------------------------------------- #

def load_validated(cv_dir: Path) -> pd.DataFrame:
    path = cv_dir / "validated.tsv"
    if not path.exists():
        sys.exit(f"Can't find {path}. Point --cv-dir at the folder that holds validated.tsv "
                 f"(usually .../cv-corpus-<version>/en).")

    header = pd.read_csv(path, sep="\t", nrows=0).columns
    wanted = ["client_id", "path", "sentence_id", "sentence", "accents", "gender", "age",
              "up_votes", "down_votes"]
    missing = {"client_id", "path", "sentence", "accents"} - set(header)
    if missing:
        sys.exit(f"validated.tsv is missing required columns: {sorted(missing)}")

    # QUOTE_NONE: Common Voice sentences contain stray quote characters.
    # keep_default_na=False: blank accents stay "" instead of becoming NaN.
    df = pd.read_csv(path, sep="\t", usecols=[c for c in wanted if c in header], dtype=str,
                     quoting=csv.QUOTE_NONE, keep_default_na=False, on_bad_lines="warn")
    for col in ["sentence_id", "gender", "age"]:
        if col not in df:
            df[col] = ""
    df["accents"] = df["accents"].str.strip()
    return df


def load_durations(cv_dir: Path):
    path = cv_dir / "clip_durations.tsv"
    if not path.exists():
        return None
    d = pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False)
    d.columns = ["path", "duration_ms"]
    d["duration_ms"] = pd.to_numeric(d["duration_ms"], errors="coerce")
    return d


def guess_version(cv_dir: Path) -> str:
    match = re.search(r"(cv-corpus-[\w.\-]+|v?\d+\.\d+)", str(cv_dir.resolve()))
    return match.group(1) if match else "unknown"


# --------------------------------------------------------------------------- #
# 2–3. Explore and map labels
# --------------------------------------------------------------------------- #

def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct raw `accents` value, with clip/speaker counts and its mapping."""
    labels = (df.groupby("accents")
                .agg(clips=("path", "size"), speakers=("client_id", "nunique"))
                .reset_index()
                .rename(columns={"accents": "raw_label"}))
    mapped = labels["raw_label"].map(ag.classify_label)
    labels["status"] = [s for s, _ in mapped]
    labels["group"] = [g or "" for _, g in mapped]
    return labels.sort_values("clips", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# 6. Sampling
# --------------------------------------------------------------------------- #

def sample_clips(pool: pd.DataFrame, per_group: int, max_per_speaker: int,
                 seed: int) -> pd.DataFrame:
    """Speaker-balanced, sentence-first greedy sampling. See module docstring."""
    rng = np.random.default_rng(seed)
    pool = pool.reset_index(drop=True)
    pool["_tiebreak"] = rng.random(len(pool))

    coverage = pool.groupby("sentence_key")["accent_group"].nunique()
    sentences = coverage.to_frame("n_groups")
    sentences["_shuffle"] = rng.random(len(sentences))
    sentence_order = sentences.sort_values(["n_groups", "_shuffle"],
                                           ascending=[False, True]).index.tolist()

    # (sentence, group) -> candidate row positions, in random (tiebreak) order
    pool = pool.sort_values("_tiebreak", ignore_index=True)
    candidates = pool.groupby(["sentence_key", "accent_group"], sort=False).indices
    groups_for_sentence = pool.groupby("sentence_key", sort=False)["accent_group"].unique().to_dict()
    speaker_of = pool["client_id"].to_numpy()

    taken = {g: 0 for g in pool["accent_group"].unique()}
    per_speaker = {}
    done_pairs = set()  # (sentence, group) already used: no sentence twice within a group
    chosen = []

    # Progressive cap: pass 1 allows 1 clip per speaker, pass 2 allows 2, ...
    # so a group only goes back to a speaker once it has run out of new ones.
    for cap in range(1, max_per_speaker + 1):
        open_groups = {g for g, n in taken.items() if n < per_group}
        if not open_groups:
            break
        for s in sentence_order:
            if not open_groups:
                break
            for g in groups_for_sentence[s]:
                if g not in open_groups or (s, g) in done_pairs:
                    continue
                best_row, best_count = None, None
                for row in candidates[(s, g)]:
                    count = per_speaker.get(speaker_of[row], 0)
                    if count >= cap:
                        continue
                    if best_count is None or count < best_count:
                        best_row, best_count = row, count
                        if count == 0:
                            break  # can't do better than a new speaker
                if best_row is not None:
                    chosen.append(best_row)
                    spk = speaker_of[best_row]
                    per_speaker[spk] = per_speaker.get(spk, 0) + 1
                    done_pairs.add((s, g))
                    taken[g] += 1
                    if taken[g] >= per_group:
                        open_groups.discard(g)

    out = pool.iloc[chosen].drop(columns="_tiebreak")
    out["sentence_n_groups_in_pool"] = out["sentence_key"].map(coverage)
    return out


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def run(args) -> dict:
    cv_dir = Path(args.cv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    funnel = []  # (step, clips, speakers)

    def log_step(name, frame):
        funnel.append((name, len(frame), frame["client_id"].nunique()))
        print(f"  {name:<58} {len(frame):>10,} clips  {frame['client_id'].nunique():>8,} speakers")

    print(f"Loading {cv_dir / 'validated.tsv'} ...")
    df = load_validated(cv_dir)
    log_step("Validated clips (validated.tsv)", df)

    # 2. Explore
    labels = map_labels(df)
    labels.to_csv(out_dir / "accent_label_mapping.csv", index=False)
    n_blank = int((df["accents"] == "").sum())
    print(f"\n  {labels['raw_label'].ne('').sum():,} distinct non-blank accent labels; "
          f"{n_blank:,} clips ({n_blank / len(df):.1%}) have no accent label.")

    if args.explore_only:
        print_status_summary(labels)
        print(f"\nWrote {out_dir / 'accent_label_mapping.csv'} (explore only, nothing sampled).")
        return {}

    # 3. Drop blanks, map labels
    df = df[df["accents"] != ""].copy()
    log_step("Has an accent label", df)
    label_to_group = labels.set_index("raw_label")["group"]
    df["accent_group"] = df["accents"].map(label_to_group)
    df = df[df["accent_group"] != ""].copy()
    log_step("Label maps cleanly to one accent group", df)

    # 4. Speakers must map to a single group across all their clips
    groups_per_speaker = df.groupby("client_id")["accent_group"].nunique()
    inconsistent = groups_per_speaker[groups_per_speaker > 1].index
    df = df[~df["client_id"].isin(inconsistent)].copy()
    log_step(f"Speaker's clips all map to one group (dropped {len(inconsistent)} speakers)", df)

    # 5. Sentence length and audio duration
    df["n_words"] = df["sentence"].str.split().str.len()
    df = df[df["n_words"] >= args.min_words].copy()
    log_step(f"Sentence has >= {args.min_words} words", df)

    durations = load_durations(cv_dir)
    if durations is not None:
        df = df.merge(durations, on="path", how="left")
        df = df[df["duration_ms"].isna() | (df["duration_ms"] >= args.min_duration_ms)].copy()
        log_step(f"Audio >= {args.min_duration_ms / 1000:g} s (clip_durations.tsv)", df)
    else:
        df["duration_ms"] = np.nan

    # Sentence identity: Common Voice's sentence_id when available, else the normalized text
    text_key = df["sentence"].str.lower().str.replace(r"[^\w\s']", "", regex=True) \
                             .str.split().str.join(" ")
    df["sentence_key"] = np.where(df["sentence_id"] != "", df["sentence_id"], text_key)

    pool = df
    pool_stats = summarize(pool)

    # 6. Sample
    print(f"\nSampling up to {args.per_group} clips per group "
          f"(max {args.max_per_speaker} per speaker, seed {args.seed}) ...")
    sample = sample_clips(pool, args.per_group, args.max_per_speaker, args.seed)
    funnel.append(("Final sample", len(sample), sample["client_id"].nunique()))

    # 7. Save. Speakers get project-local pseudonyms (spk_0001, ...) so the
    # Common Voice client_id hash is never republished; the pseudonym is kept
    # so Phase 4 can bootstrap by speaker.
    speaker_ids = {c: f"spk_{i:04d}" for i, c in
                   enumerate(sorted(sample["client_id"].unique()), start=1)}
    sample = sample.assign(speaker=sample["client_id"].map(speaker_ids))
    sample = sample.sort_values(["accent_group", "speaker", "path"])
    out = pd.DataFrame({
        "clip_id": sample["path"],
        "accent_group": sample["accent_group"],
        "sentence": sample["sentence"],
        "sentence_id": sample["sentence_key"],
        "speaker": sample["speaker"],
        "accent_label_raw": sample["accents"],
        "gender": sample["gender"],
        "age": sample["age"],
        "n_words": sample["n_words"],
        "duration_s": (sample["duration_ms"] / 1000).round(2),
        "sentence_n_groups_in_pool": sample["sentence_n_groups_in_pool"],
    })
    out.to_csv(out_dir / "clips_selected.csv", index=False)

    sample_stats = summarize(sample)
    card = build_data_card(args, cv_dir, labels, n_blank, funnel, pool_stats, sample_stats,
                           sample, len(inconsistent), durations is not None)
    (out_dir / "DATA_CARD.md").write_text(card)

    print("\nFinal sample:")
    print(sample_stats[["clips", "speakers", "sentences", "max_clips_per_speaker"]]
          .to_string())
    for g, row in sample_stats.iterrows():
        if row["clips"] < args.per_group:
            print(f"  WARNING: {g} has only {row['clips']} clips (target {args.per_group}).")
        if row["speakers"] < args.min_speakers:
            print(f"  WARNING: {g} has only {row['speakers']} speakers "
                  f"(< {args.min_speakers}); results for it will be fragile.")
    print(f"\nWrote {out_dir / 'clips_selected.csv'}, {out_dir / 'accent_label_mapping.csv'}, "
          f"{out_dir / 'DATA_CARD.md'}")
    return {"sample": out, "labels": labels, "stats": sample_stats}


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.groupby("accent_group")
    stats = pd.DataFrame({
        "clips": g.size(),
        "speakers": g["client_id"].nunique(),
        "sentences": g["sentence_key"].nunique(),
        "max_clips_per_speaker": g["client_id"].agg(lambda s: s.value_counts().max()),
        "mean_words": g["n_words"].mean().round(1),
        "median_duration_s": (g["duration_ms"].median() / 1000).round(2),
    })
    order = [k for k in ag.GROUPS if k in stats.index]
    return stats.loc[order]


def print_status_summary(labels: pd.DataFrame):
    by_status = labels.groupby("status")["clips"].sum().reindex(STATUS_ORDER, fill_value=0)
    print("\n  Clips by mapping status:")
    for status, n in by_status.items():
        print(f"    {status:<14} {n:>10,}")
    by_group = labels[labels["status"] == "grouped"].groupby("group")["clips"].sum()
    print("\n  Clips per group (before speaker/sentence filters):")
    for g, n in by_group.sort_values(ascending=False).items():
        print(f"    {g:<16} {n:>10,}")
    top = labels[labels["status"].isin(["unmapped", "conflict"])].head(25)
    if len(top):
        print("\n  Biggest unmapped/conflicting labels (consider adding rules):")
        for _, r in top.iterrows():
            print(f"    {r['clips']:>8,}  [{r['status']}]  {r['raw_label'][:90]}")


# --------------------------------------------------------------------------- #
# Data card
# --------------------------------------------------------------------------- #

def md_table(frame: pd.DataFrame, index=True) -> str:
    frame = frame.reset_index() if index else frame
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in frame.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float) and np.isnan(v):
                v = "–"
            elif isinstance(v, (int, np.integer)):
                v = f"{v:,}"
            cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_data_card(args, cv_dir, labels, n_blank, funnel, pool_stats, sample_stats, sample,
                    n_inconsistent, has_durations) -> str:
    version = args.cv_version or guess_version(cv_dir)
    groups_present = list(sample_stats.index)
    n_groups = len(groups_present)

    per_sentence = sample.groupby("sentence_key")["accent_group"].nunique()
    shared_all = int((per_sentence == n_groups).sum())
    shared_2 = int((per_sentence >= 2).sum())
    clips_on_shared_all = int(sample["sentence_key"].map(per_sentence).eq(n_groups).sum())

    funnel_df = pd.DataFrame(funnel, columns=["Step", "Clips", "Speakers"])
    status_df = (labels.groupby("status")
                       .agg(distinct_labels=("raw_label", "size"), clips=("clips", "sum"))
                       .reindex(STATUS_ORDER).dropna().astype(int))

    def crosstab(col):
        t = pd.crosstab(sample["accent_group"], sample[col].replace("", "(blank)"))
        return t.loc[[g for g in ag.GROUPS if g in t.index]]

    rule_rows = [{"#": i + 1, "Result": res, "Pattern (regex)": f"`{pat}`", "Reason": why}
                 for i, (pat, res, why) in enumerate(ag.RULES)]

    s = []
    s.append("# Data card: Common Voice English accent evaluation set\n")
    s.append(f"_Generated by `src/prepare_dataset.py` on {date.today().isoformat()}. "
             f"Do not edit by hand; change the code or rules and re-run._\n")

    s.append("## Source\n")
    s.append(f"- **Dataset:** Mozilla Common Voice, English (`en`), validated split")
    s.append(f"- **Version:** {version}")
    s.append(f"- **Obtained from:** Mozilla Data Collective")
    s.append(f"- **Licence:** CC0 1.0. Audio is not redistributed here; this repo contains clip IDs "
             f"and metadata only.")
    s.append(f"- **Input files:** `validated.tsv`" + (", `clip_durations.tsv`" if has_durations else ""))
    s.append("")

    s.append("## Settings\n")
    s.append(md_table(pd.DataFrame([
        ("Target clips per group", args.per_group),
        ("Max clips per speaker", args.max_per_speaker),
        ("Min words per sentence", args.min_words),
        ("Min audio duration (s)", args.min_duration_ms / 1000 if has_durations else "n/a"),
        ("Random seed", args.seed),
    ], columns=["Setting", "Value"]), index=False))
    s.append("")

    s.append("## Filtering steps\n")
    s.append(md_table(funnel_df, index=False))
    s.append("")
    s.append(f"- {n_blank:,} validated clips ({n_blank / funnel[0][1]:.1%}) had a blank accent field "
             f"and were dropped.")
    s.append(f"- {n_inconsistent:,} speakers had clips that mapped to more than one group "
             f"(they changed their profile label) and were dropped entirely.")
    s.append(f"- Sentences under {args.min_words} words were dropped: on a 3-word sentence a single "
             f"error is already 33% WER, which makes per-clip scores very noisy.")
    s.append("")

    s.append("## How accent labels were grouped\n")
    s.append("Common Voice accent labels are self-reported free text; a speaker can list several, "
             "separated by `|`. Each piece is lowercased, has periods removed, and is checked "
             "against the rules below in order (first match wins). A clip is kept only if "
             "**every informative piece maps to the same group**. Pieces marked IGNORE "
             "(e.g. \"neutral\") are skipped; any EXCLUDE, unmapped, or conflicting piece removes "
             "the clip.\n")
    s.append("### Groups\n")
    s.append(md_table(pd.DataFrame([(k, v) for k, v in ag.GROUPS.items()],
                                   columns=["Group", "Description"]), index=False))
    s.append("")
    s.append("### Rules (in order)\n")
    s.append(md_table(pd.DataFrame(rule_rows), index=False))
    s.append("")
    s.append("### Result of mapping every distinct label\n")
    s.append(md_table(status_df))
    s.append("")
    s.append("Every raw label, its clip and speaker counts, and the status/group it was given are in "
             "[`accent_label_mapping.csv`](accent_label_mapping.csv). The labels that were assigned "
             "to each group:\n")
    grouped = labels[labels["status"] == "grouped"]
    for g in ag.GROUPS:
        sub = grouped[grouped["group"] == g]
        if sub.empty:
            continue
        s.append(f"<details><summary><b>{g}</b>: {len(sub):,} distinct labels, "
                 f"{sub['clips'].sum():,} clips</summary>\n")
        s.append(md_table(sub[["raw_label", "clips", "speakers"]], index=False))
        s.append("\n</details>\n")

    s.append("## Sampling\n")
    s.append(f"From the eligible pool, up to {args.per_group} clips were drawn per group with a "
             f"speaker-balanced, sentence-first greedy procedure:\n")
    s.append("1. Each sentence is scored by how many groups have at least one eligible clip of it.")
    s.append(f"2. Sampling runs in passes with a rising per-speaker cap: pass 1 allows 1 clip per "
             f"speaker, pass 2 allows 2, up to {args.max_per_speaker}. A group only returns to a "
             f"speaker once it has run out of new speakers.")
    s.append("3. Within each pass, sentences are visited from most-shared to least-shared (ties in "
             "random order, fixed seed); every group that has the sentence and is not yet full "
             "takes one clip of it, from the speaker with the fewest clips selected so far.")
    s.append("4. No sentence appears twice within a group. Each group stops at the target.\n")
    s.append("Speaker spread comes first because clips from the same person are not independent: "
             "400 clips from 400 speakers say far more about a group than 400 clips from 40. "
             "Preferring shared sentences then makes groups read the same sentences where possible, "
             "so WER differences are less likely to come from some groups getting harder "
             "sentences.\n")
    s.append("### Eligible pool (before sampling)\n")
    s.append(md_table(pool_stats))
    s.append("")

    s.append("## Final dataset\n")
    s.append(md_table(sample_stats))
    s.append("")
    s.append(f"- **Total:** {len(sample):,} clips from {sample['client_id'].nunique():,} speakers "
             f"in {n_groups} groups.")
    s.append(f"- **Sentence overlap:** {shared_all:,} sentences appear in all {n_groups} groups "
             f"({clips_on_shared_all:,} clips); {shared_2:,} sentences appear in at least 2 groups.")
    s.append("")
    s.append("### Gender (self-reported) per group\n")
    s.append(md_table(crosstab("gender")))
    s.append("")
    s.append("### Age (self-reported) per group\n")
    s.append(md_table(crosstab("age")))
    s.append("")
    s.append("Gender and age are reported, not balanced. If they differ a lot between groups they "
             "are possible confounds; mention this in the limitations section.\n")

    s.append("## Output file: `clips_selected.csv`\n")
    s.append(md_table(pd.DataFrame([
        ("clip_id", "Common Voice audio filename (in `clips/`)"),
        ("accent_group", "Group assigned by the rules above"),
        ("sentence", "Reference transcript (the prompt the speaker read)"),
        ("sentence_id", "Common Voice sentence_id (or normalized text in older releases)"),
        ("speaker", "Pseudonymous speaker ID local to this project (not the Common Voice client_id)"),
        ("accent_label_raw", "The speaker's original self-reported accent text"),
        ("gender, age", "Self-reported, as given in Common Voice (may be blank)"),
        ("n_words", "Words in the reference sentence"),
        ("duration_s", "Clip length in seconds (blank if clip_durations.tsv was not available)"),
        ("sentence_n_groups_in_pool", "How many groups had this sentence in the eligible pool"),
    ], columns=["Column", "Meaning"]), index=False))
    s.append("")

    s.append("## Ethics and known limitations\n")
    s.append("- Results are reported only at group level. No attempt is made to identify speakers, "
             "and Common Voice client IDs are replaced with project-local pseudonyms.")
    s.append("- Accent labels are self-reported and inconsistent; the grouping rules above are a "
             "judgement call and different rules could change the results.")
    s.append("- Groups pool together real variation (e.g. all of the US, or all European first "
             "languages). A group's WER is an average over that variation.")
    s.append("- Common Voice is read speech recorded on contributors' own devices; recording "
             "quality may differ between groups.")
    s.append("- Contributors are not a random sample of each accent's speakers.")
    return "\n".join(s) + "\n"


# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cv-dir", required=True,
                   help="Folder containing validated.tsv (e.g. data/raw/cv-corpus-23.0/en)")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--cv-version", default=None,
                   help="Version string for the data card (guessed from the path if omitted)")
    p.add_argument("--per-group", type=int, default=400)
    p.add_argument("--max-per-speaker", type=int, default=10)
    p.add_argument("--min-words", type=int, default=4)
    p.add_argument("--min-duration-ms", type=int, default=1000)
    p.add_argument("--min-speakers", type=int, default=30,
                   help="Warn if a group ends up with fewer speakers than this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--explore-only", action="store_true",
                   help="Only count and map accent labels; don't sample")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
