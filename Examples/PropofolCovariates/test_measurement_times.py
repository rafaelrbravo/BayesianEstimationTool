"""Regression tests for rounded observation/event time validation."""
import pandas as pd
import pytest

from inspect_data import validate_measurement_times


def tables(observation_minutes, event_seconds):
    observations = pd.DataFrame({
        "ID": [1] * len(observation_minutes),
        "TIME": observation_minutes,
        "CP": [1.] * len(observation_minutes),
    })
    events = pd.DataFrame({
        "ID": [1] * len(event_seconds),
        "TIME": event_seconds,
        "ISMEASUREMENT": [1] * len(event_seconds),
    })
    return {"observations": observations, "events": events}


def test_measurement_times_allow_nearest_minute_rounding():
    # The second observation is 1 minute, while its event is at 90 seconds.
    validate_measurement_times(tables([0., 1.], [0., 90.]), [1])


def test_measurement_times_reject_disagreement_beyond_rounding_tolerance():
    with pytest.raises(ValueError, match="times disagree"):
        validate_measurement_times(tables([0., 1.], [0., 91.]), [1])


def test_measurement_times_allow_explicit_tighter_tolerance():
    with pytest.raises(ValueError, match="times disagree"):
        validate_measurement_times(tables([0., 1.], [0., 90.]), [1], tolerance_seconds=1.)
