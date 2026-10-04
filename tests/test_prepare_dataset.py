"""
Tests run on a small fake Common Voice folder, so they work before you've
downloaded anything:  .venv/bin/pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import accent_groups as ag  # noqa: E402
import prepare_dataset as prep  # noqa: E402


# --------------------------------------------------------------------------- #
# Label classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw, expected", [
    # Current Common Voice predefined labels
    ("United States English", ("grouped", "us")),
    ("England English", ("grouped", "england")),
    ("Canadian English", ("grouped", "canada")),
    ("Australian English", ("grouped", "australia_nz")),
    ("New Zealand English", ("grouped", "australia_nz")),
    ("India and South Asia (India, Pakistan, Sri Lanka)", ("grouped", "south_asia")),
    ("Southern African (South Africa, Zimbabwe, Namibia)", ("grouped", "southern_africa")),
    ("Filipino", ("grouped", "southeast_asia")),
    ("Singaporean English", ("grouped", "southeast_asia")),
    ("German English,Non native speaker", ("grouped", "l2_european")),
    # Legacy (pre-free-text) labels
    ("us", ("grouped", "us")),
    ("indian", ("grouped", "south_asia")),
    ("newzealand", ("grouped", "australia_nz")),
    ("african", ("excluded", None)),
    # Traps: substring matches that must NOT land in the obvious group
    ("New England", ("grouped", "us")),
    ("African American", ("excluded", None)),
    ("West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad)", ("excluded", None)),
    ("Latin American", ("excluded", None)),
    ("French Canadian", ("grouped", "l2_european")),
    ("Scottish English", ("excluded", None)),
    ("British", ("excluded", None)),
    ("Hong Kong English", ("excluded", None)),
    ("Kenyan English", ("excluded", None)),
    ("U.S.A.", ("grouped", "us")),
    # Multi-label rules
    ("United States English|Midwestern", ("grouped", "us")),
    ("United States English|Neutral", ("grouped", "us")),
    ("United States English|England English", ("conflict", None)),
    ("United States English|something odd", ("unmapped", None)),
    ("Mixed American and British", ("excluded", None)),
    ("Neutral", ("uninformative", None)),
    # Non-native marker only counts alongside a European first language
    ("Non native speaker|German English", ("grouped", "l2_european")),
    ("Non-native|Spanish|Foreign", ("grouped", "l2_european")),
    ("Slavic|polish", ("grouped", "l2_european")),
    ("Non native speaker", ("unmapped", None)),
    ("Non native speaker|United States English", ("conflict", None)),
    # Regional descriptions found in Common Voice 27.0
    ("England English|Liverpool English|Lancashire English", ("grouped", "england")),
    ("United States English|Midwestern|Minnesotan", ("grouped", "us")),
    ("Canadian English|Ontario", ("grouped", "canada")),
    ("British Columbia", ("grouped", "canada")),
    ("United States English|Speech impediment|Rhotacism", ("excluded", None)),
    ("South Atlantic (Falkland Islands, Saint Helena)", ("excluded", None)),
    ("", ("blank", None)),
])
def test_classify_label(raw, expected):
    assert ag.classify_label(raw) == expected


def test_every_rule_result_is_known():
    for _, result, _ in ag.RULES:
        assert result in ag.GROUPS or result in (ag.EXCLUDE, ag.IGNORE, ag.L2_MARKER)


# --------------------------------------------------------------------------- #
# End-to-end on a fake corpus
# --------------------------------------------------------------------------- #

GROUP_LABELS = {
    "us": ["United States English", "United States English|Midwestern"],
    "england": ["England English"],
    "canada": ["Canadian English"],
    "australia_nz": ["Australian English", "New Zealand English"],
    "south_asia": ["India and South Asia (India, Pakistan, Sri Lanka)"],
    "southern_africa": ["Southern African (South Africa, Zimbabwe, Namibia)"],
    "southeast_asia": ["Filipino", "Malaysian English"],
    "l2_european": ["German English,Non native speaker", "Spanish accent"],
}
SPEAKERS_PER_GROUP = {"us": 120, "england": 80, "canada": 40, "australia_nz": 40,
                      "south_asia": 60, "southern_africa": 30, "southeast_asia": 12,
                      "l2_european": 40}
NOISE_LABELS = ["", "", "", "Scottish English", "Neutral", "United States English|England English",
                "Mixed", "something odd"]


@pytest.fixture(scope="module")
def fake_cv(tmp_path_factory):
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("cv-corpus-99.0-2026-09-01") / "en"
    root.mkdir()
    sentences = [f"this is test sentence number {i} for the audit" for i in range(300)]
    sentences += ["too short", "hi"]

    rows, n = [], 0
    def add(client, label, sent):
        nonlocal n
        rows.append({"client_id": client, "path": f"common_voice_en_{n}.mp3",
                     "sentence_id": f"sid{sentences.index(sent)}", "sentence": sent,
                     "sentence_domain": "", "up_votes": "2", "down_votes": "0",
                     "age": rng.choice(["twenties", "thirties", ""]),
                     "gender": rng.choice(["male_masculine", "female_feminine", ""]),
                     "accents": label, "variant": "", "locale": "en", "segment": ""})
        n += 1

    for g, n_spk in SPEAKERS_PER_GROUP.items():
        for s in range(n_spk):
            client = f"{g}_client_{s}"
            label = GROUP_LABELS[g][s % len(GROUP_LABELS[g])]
            # Prolific speakers: some read 60 clips, most read a few
            n_clips = 60 if s < 3 else int(rng.integers(3, 15))
            for sent in rng.choice(sentences, size=n_clips, replace=True):
                add(client, label, sent)
    for i, label in enumerate(NOISE_LABELS * 30):
        add(f"noise_{i}", label, sentences[i % 300])
    # A speaker who changed their profile: US clips and England clips
    for sent in sentences[:5]:
        add("switcher", "United States English", sent)
        add("switcher", "England English", sent)

    df = pd.DataFrame(rows)
    df.to_csv(root / "validated.tsv", sep="\t", index=False)
    durations = pd.DataFrame({"clip": df["path"], "duration[ms]": 4000})
    durations.loc[0, "duration[ms]"] = 300  # one sub-second clip
    durations.to_csv(root / "clip_durations.tsv", sep="\t", index=False)
    return root


@pytest.fixture(scope="module")
def result(fake_cv, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    args = prep.parse_args(["--cv-dir", str(fake_cv), "--out-dir", str(out),
                            "--per-group", "100", "--max-per-speaker", "10"])
    res = prep.run(args)
    res["out"] = out
    return res


def test_outputs_written(result):
    for name in ["clips_selected.csv", "accent_label_mapping.csv", "DATA_CARD.md"]:
        assert (result["out"] / name).stat().st_size > 0


def test_speaker_cap(result):
    assert result["sample"].groupby("speaker").size().max() <= 10


def test_group_targets(result):
    counts = result["sample"]["accent_group"].value_counts()
    for g in SPEAKERS_PER_GROUP:
        if g == "southeast_asia":
            # 12 speakers x 10 cap = 120 possible, minus short sentences etc.
            assert counts[g] <= 100
        else:
            assert counts[g] == 100, g


def test_no_bad_rows(result):
    s = result["sample"]
    assert (s["accent_label_raw"] != "").all()
    assert (s["n_words"] >= 4).all()
    assert not s.duplicated(["accent_group", "sentence_id"]).any()
    assert set(s["accent_group"]) <= set(ag.GROUPS)
    assert "switcher" not in set(s["speaker"])  # pseudonymized and dropped anyway
    assert s["speaker"].str.match(r"^spk_\d{4}$").all()
    assert (s["duration_s"] >= 1).all()


def test_no_client_ids_leak(result):
    text = (result["out"] / "clips_selected.csv").read_text()
    assert "_client_" not in text


def test_sentences_shared_across_groups(result):
    s = result["sample"]
    per_sentence = s.groupby("sentence_id")["accent_group"].nunique()
    # With 300 shared sentences available, most picks should be shared ones
    assert (s["sentence_id"].map(per_sentence) >= 2).mean() > 0.8


def test_spreads_over_speakers(result):
    # Prolific speakers (60 clips) must not crowd out others
    s = result["sample"]
    us = s[s["accent_group"] == "us"]
    assert us["speaker"].nunique() >= 50


def test_deterministic(fake_cv, tmp_path):
    a = prep.run(prep.parse_args(["--cv-dir", str(fake_cv), "--out-dir", str(tmp_path / "a"),
                                  "--per-group", "50"]))
    b = prep.run(prep.parse_args(["--cv-dir", str(fake_cv), "--out-dir", str(tmp_path / "b"),
                                  "--per-group", "50"]))
    pd.testing.assert_frame_equal(a["sample"].reset_index(drop=True),
                                  b["sample"].reset_index(drop=True))
