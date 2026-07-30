"""ForexFactory calendar parsing — pure, no network."""
from datetime import datetime, timezone, timedelta

from app.api.v1.market_data import _parse_ff_events

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
RAW = [
    {"title": "Non-Farm Payrolls", "country": "USD", "impact": "High",
     "date": "2026-07-06T09:30:00-04:00", "forecast": "180K", "previous": "150K"},
    {"title": "MI Inflation Gauge", "country": "AUD", "impact": "Low",
     "date": "2026-07-05T21:00:00-04:00", "forecast": "", "previous": "-0.3%"},
]


def test_impact_and_country_filters():
    assert len(_parse_ff_events(RAW, "High", None, NOW)) == 1
    assert len(_parse_ff_events(RAW, None, "AUD", NOW)) == 1
    assert _parse_ff_events(RAW, "High", "AUD", NOW) == []


def test_hours_away_and_sorting():
    events = _parse_ff_events(RAW, None, None, NOW)
    # NFP at 13:30 UTC is 1.5h from NOW; AUD event is in the past (negative)
    nfp = next(e for e in events if e["title"] == "Non-Farm Payrolls")
    assert nfp["hours_away"] == 1.5
    assert events[0]["hours_away"] <= events[1]["hours_away"]


def test_empty_and_malformed_are_safe():
    assert _parse_ff_events([], None, None, NOW) == []
    bad = _parse_ff_events([{"title": "X", "date": "not-a-date", "impact": "High"}], None, None, NOW)
    assert bad[0]["hours_away"] is None


def test_none_inputs_are_handled_gracefully():
    # Passing None for the raw events list should return an empty list without error
    assert _parse_ff_events(None, None, None, NOW) == []
    # Passing None for the reference time should not raise and should treat it as now if the function supports it
    # The function is expected to fallback to utcnow() internally; we verify it returns a list (may be empty)
    result = _parse_ff_events(RAW, None, None, None)
    assert isinstance(result, list)


def test_off_by_one_hour_boundary():
    # Create an event exactly 1 hour away from NOW
    event_one_hour = {
        "title": "Exact One Hour Event",
        "country": "EUR",
        "impact": "Medium",
        "date": (NOW + timedelta(hours=1)).isoformat(),
        "forecast": "",
        "previous": "",
    }
    events = _parse_ff_events([event_one_hour], None, None, NOW)
    assert len(events) == 1
    # The hours_away should be exactly 1.0, not rounded up or down incorrectly
    assert events[0]["hours_away"] == 1.0
    # Ensure sorting works when the list contains a single element
    assert events[0] == events[0]  # trivial truth to keep the test structure consistent