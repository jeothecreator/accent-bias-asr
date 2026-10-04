import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import analyze  # noqa: E402


@pytest.fixture
def fake_results():
    rng = np.random.default_rng(0)
    rows = []
    # Inner groups ~5% WER, other groups ~15%, two models; 30 speakers x 2 clips each
    for g, base in [("us", .05), ("england", .05), ("canada", .05), ("australia_nz", .05),
                    ("south_asia", .15), ("southern_africa", .15), ("southeast_asia", .15),
                    ("l2_european", .15)]:
        for s in range(30):
            for c in range(2):
                words = int(rng.integers(5, 15))
                for m, scale in [("tiny", 2.0), ("large-v3", 1.0)]:
                    err = rng.binomial(words, min(1, base * scale))
                    rows.append({"clip_id": f"{g}_{s}_{c}.mp3", "accent_group": g, "speaker": f"{g}{s}",
                                 "model": m, "reference": "x", "prediction": "y", "wer": err / words,
                                 "substitutions": err, "deletions": 0, "insertions": 0,
                                 "ref_words": words})
    return pd.DataFrame(rows)


def test_point_estimate_is_pooled_wer(fake_results):
    r = fake_results.assign(errors=fake_results["substitutions"])
    groups, models = list(analyze.CIRCLE), ["tiny", "large-v3"]
    point, boot, _, _ = analyze.bootstrap(r, groups, models, 200, 0)
    sub = r[(r.accent_group == "us") & (r.model == "tiny")]
    assert point["us"][0] == pytest.approx(sub.errors.sum() / sub.ref_words.sum())
    lo, hi = analyze.ci(boot["us"])
    assert lo[0] <= point["us"][0] <= hi[0]


def test_end_to_end_outputs(fake_results, tmp_path):
    f = tmp_path / "results.csv"
    fake_results.to_csv(f, index=False)
    analyze.main(["--results", str(f), "--out-dir", str(tmp_path), "--n-boot", "100", "--n-worst", "10"])
    gap = pd.read_csv(tmp_path / "gap_by_model.csv").set_index("model")
    assert gap.loc["tiny", "tier_ratio"] == pytest.approx(3, rel=0.35)   # ~15% / ~5%
    assert len(pd.read_csv(tmp_path / "worst_clips_to_tag.csv")) == 10
    for name in ["fig1_wer_by_group_large-v3.png", "fig2_gap_vs_model_size.png",
                 "fig3_wer_vs_model_size_by_group.png", "fig1b_wer_by_group_all_models.png"]:
        assert (tmp_path / "figures" / name).stat().st_size > 10_000
    assert "Draft key findings" in (tmp_path / "PHASE4_SUMMARY.md").read_text()
