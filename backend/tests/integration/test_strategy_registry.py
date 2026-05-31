"""
Integration tests: strategy registry imports and initialization.
Catches import errors, missing abstract methods, and broken __init__.py wiring.
"""
from __future__ import annotations

import pytest
from app.strategies import STRATEGY_REGISTRY, get_strategy


class TestStrategyRegistry:
    def test_registry_not_empty(self):
        assert len(STRATEGY_REGISTRY) >= 50, (
            f"Expected 50+ strategies, got {len(STRATEGY_REGISTRY)}"
        )

    def test_get_strategy_returns_instance(self):
        name = next(iter(STRATEGY_REGISTRY))
        instance = get_strategy(name)
        assert instance is not None

    def test_all_strategy_classes_are_importable(self):
        """Every value in the registry must be a class, not None."""
        for name, cls in STRATEGY_REGISTRY.items():
            assert cls is not None, f"Strategy '{name}' registered as None"
            assert callable(cls), f"Strategy '{name}' is not callable"

    def test_required_strategy_groups_present(self):
        """Spot-check one strategy from each major desk."""
        required = [
            # Equities
            "momentum", "mean_reversion", "pairs_trading", "rsi_macd",
            # Crypto
            "triangular_arb", "funding_rate_arb", "crypto_adaptive_trend",
            # Polymarket
            "poly_binary_arb", "poly_calibration_arb",
            # Options
            "covered_call", "iron_condor",
            # Stat arb
            "pca_stat_arb", "kalman_pairs",
        ]
        missing = [s for s in required if s not in STRATEGY_REGISTRY]
        assert not missing, f"Missing strategies: {missing}"

    def test_strategy_classes_have_required_attributes(self):
        """Each strategy class must declare name and market_type."""
        for strat_name, cls in STRATEGY_REGISTRY.items():
            instance = cls()
            assert hasattr(instance, "name"), (
                f"Strategy '{strat_name}' missing 'name' attribute"
            )
            assert hasattr(instance, "market_type"), (
                f"Strategy '{strat_name}' missing 'market_type' attribute"
            )
            assert instance.market_type in ("equity", "crypto", "polymarket", "options", "fx"), (
                f"Strategy '{strat_name}' has invalid market_type: {instance.market_type!r}"
            )

    def test_strategy_classes_have_analyze_method(self):
        """Each strategy must implement analyze()."""
        import inspect
        for strat_name, cls in STRATEGY_REGISTRY.items():
            assert hasattr(cls, "analyze"), (
                f"Strategy '{strat_name}' missing 'analyze' method"
            )

    def test_no_duplicate_names(self):
        """Strategy instance names must match their registry key."""
        for key, cls in STRATEGY_REGISTRY.items():
            instance = cls()
            assert instance.name == key, (
                f"Registry key '{key}' but strategy.name is '{instance.name}'"
            )
