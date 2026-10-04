import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
pytest.importorskip("whisper")
import score  # noqa: E402
from transcribe import read_transcripts  # noqa: E402


def test_normalization_differences_are_not_errors():
    r = score.score_pair("Mr. Smith's colour is $20.", "mister smith's color is $20")
    assert r["wer"] == 0


def test_counts_each_error_type():
    # cat->bat (substitution), "big" missing (deletion), "today" extra (insertion)
    r = score.score_pair("the big cat sat on a mat", "the bat sat on a mat today")
    assert (r["substitutions"], r["deletions"], r["insertions"]) == (1, 1, 1)
    assert r["ref_words"] == 7 and r["wer"] == pytest.approx(3 / 7)


def test_empty_prediction_is_all_deletions():
    r = score.score_pair("the quick brown fox", "")
    assert r["deletions"] == 4 and r["wer"] == 1


def test_half_written_line_is_ignored_and_repaired(tmp_path):
    f = tmp_path / "tiny.csv"
    f.write_text('clip_id,model,prediction,seconds\r\na.mp3,tiny,hello there,0.30\r\nb.mp3,tiny,"half wri')
    assert list(read_transcripts(f)["clip_id"]) == ["a.mp3"]
    read_transcripts(f, repair=True)
    assert f.read_bytes().endswith(b"0.30\r\n")
