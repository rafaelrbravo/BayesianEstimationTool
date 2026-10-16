"""Offline regression tests for exported propofol result summaries."""
import csv
import numpy as np
import pytest

from summarize_scores import read_scores, summarize


def write_example(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["patient_id", "baseline_unconditioned", "covariates_unconditioned",
                         "baseline_conditioned", "covariates_conditioned"])
        writer.writerows(rows)


def test_read_and_summarize(tmp_path, capsys):
    path = tmp_path / "scores.csv"
    write_example(path, [[10, 1., 0.5, 2., 1.], [20, 2., 1., 4., 2.]])
    ids, scores = read_scores(path)
    assert ids == [10, 20]
    np.testing.assert_allclose(scores["covariates_conditioned"], [1., 2.])
    summarize(ids, scores)
    output = capsys.readouterr().out
    assert "improved patients=2/2" in output
    assert "covariates better" in output


def test_duplicate_patient_ids_rejected(tmp_path):
    path = tmp_path / "scores.csv"
    write_example(path, [[10, 1., 1., 1., 1.], [10, 1., 1., 1., 1.]])
    with pytest.raises(ValueError, match="Duplicate"):
        read_scores(path)


def test_nan_rejected(tmp_path):
    path = tmp_path / "scores.csv"
    write_example(path, [[10, 1., "nan", 1., 1.]])
    with pytest.raises(ValueError, match="Invalid"):
        read_scores(path)


@pytest.mark.parametrize("rows", [
    [],
    [[10, -1., 1., 1., 1.]],
    [[10, "inf", 1., 1., 1.]],
    [[10, "bad", 1., 1., 1.]],
])
def test_invalid_score_rows_rejected(tmp_path, rows):
    path = tmp_path / "scores.csv"
    write_example(path, rows)
    with pytest.raises(ValueError):
        read_scores(path)


def test_changed_csv_schema_rejected(tmp_path):
    path = tmp_path / "scores.csv"
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["id", "baseline_unconditioned", "covariates_unconditioned",
                         "baseline_conditioned", "covariates_conditioned"])
        writer.writerow([10, 1., 1., 1., 1.])
    with pytest.raises(ValueError, match="columns"):
        read_scores(path)
