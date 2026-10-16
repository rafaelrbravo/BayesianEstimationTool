"""Regression tests for patient-aligned propofol score reporting."""
import csv
import numpy as np
import pytest

from summarize_scores import COLUMNS, read_scores, summarize


def test_read_scores_preserves_patient_alignment(tmp_path):
    path = tmp_path / "scores.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["patient_id", *COLUMNS])
        writer.writerow([42, 2.0, 1.0, 3.0, 2.0])
        writer.writerow([7, 4.0, 5.0, 6.0, 7.0])
    ids, scores = read_scores(path)
    assert ids == [42, 7]
    np.testing.assert_array_equal(scores["covariates_unconditioned"] - scores["baseline_unconditioned"], [-1., 1.])
    summarize(ids, scores)


def test_paired_bootstrap_identical_errors_has_zero_interval():
    from paired_stats import paired_bootstrap_interval

    low, high = paired_bootstrap_interval(np.zeros(8), n_boot=100)
    assert low == high == 0.


def test_paired_bootstrap_constant_improvement():
    from paired_stats import paired_bootstrap_interval

    low, high = paired_bootstrap_interval(-np.ones(8), n_boot=100)
    assert low == high == -1.


@pytest.mark.parametrize("rows", [
    [[1, 1., 1., 1., 1.], [1, 2., 2., 2., 2.]],
    [[1, float("nan"), 1., 1., 1.]],
    [[1, -1., 1., 1., 1.]],
    [],
])
def test_read_scores_rejects_invalid_rows(tmp_path, rows):
    path = tmp_path / "invalid.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["patient_id", *COLUMNS])
        writer.writerows(rows)
    with pytest.raises(ValueError):
        read_scores(path)


def test_read_scores_rejects_wrong_column_order(tmp_path):
    path = tmp_path / "wrong.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["patient_id", *reversed(COLUMNS)])
        writer.writerow([1, 1., 2., 3., 4.])
    with pytest.raises(ValueError):
        read_scores(path)


def test_summary_cli_enforces_expected_patient_count(tmp_path, monkeypatch, capsys):
    import sys
    from summarize_scores import main

    path = tmp_path / "scores.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["patient_id", *COLUMNS])
        writer.writerow([42, 2., 1., 3., 2.])
        writer.writerow([7, 4., 5., 6., 7.])
    monkeypatch.setattr(sys, "argv", ["summarize_scores.py", str(path), "--expected-patients", "2"])
    main()
    assert "2 patients" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["summarize_scores.py", str(path), "--expected-patients", "3"])
    with pytest.raises(ValueError, match="Expected 3 held-out patients, found 2"):
        main()
    monkeypatch.setattr(sys, "argv", ["summarize_scores.py", str(path), "--expected-patients", "0"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
