"""Tests for the desk registry / org structure."""
from app.tasks.org.desk_registry import DESKS, get_all_employees, get_desk

def test_all_desks_have_employees():
    for name, desk in DESKS.items():
        assert len(desk.employees) > 0, f"Desk {name} has no employees"

def test_all_desks_have_slack_channel():
    for name, desk in DESKS.items():
        assert desk.slack_channel.startswith("#"), f"Desk {name} missing Slack channel"

def test_get_all_employees_returns_all():
    all_emps = get_all_employees()
    assert len(all_emps) >= 10  # at least 10 employees

def test_get_desk_unknown_returns_none():
    assert get_desk("nonexistent") is None

def test_get_desk_strategy():
    d = get_desk("strategy")
    assert d is not None
    assert d.name == "Strategy Desk"
