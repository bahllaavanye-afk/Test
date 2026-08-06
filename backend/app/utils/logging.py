import structlog
import logging
from app.config import settings
from typing import List, Callable


def _get_logging_level() -> int:
    """Determine the logging level based on the debug flag."""
    return logging.DEBUG if settings.debug else logging.INFO


def _get_processors() -> List[Callable]:
    """Build the list of structlog processors."""
    base_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer = (
        structlog.dev.ConsoleRenderer()
        if settings.debug
        else structlog.processors.JSONRenderer()
    )
    return base_processors + [renderer]


def _get_wrapper_class(level: int):
    """Create a filtering bound logger class for the given level."""
    return structlog.make_filtering_bound_logger(level)


def configure_logging() -> None:
    """Configure the standard logging and structlog settings."""
    level = _get_logging_level()
    logging.basicConfig(level=level, format="%(message)s")

    structlog.configure(
        processors=_get_processors(),
        wrapper_class=_get_wrapper_class(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger()


# ==========================
# Unit tests for edge cases
# ==========================
if __name__ == "__main__":
    import unittest
    from unittest import mock

    class TestConfigureLogging(unittest.TestCase):
        def setUp(self):
            # Ensure a clean logging configuration before each test
            logging.root.handlers.clear()

        def tearDown(self):
            # Clean up after test to avoid side‑effects for subsequent runs
            logging.root.handlers.clear()

        def test_debug_true_uses_console_renderer(self):
            """When settings.debug is truthy, DEBUG level and ConsoleRenderer should be used."""
            with mock.patch.object(settings, "debug", True):
                configure_logging()
                self.assertEqual(logging.getLogger().level, logging.DEBUG)
                # The last processor should be ConsoleRenderer
                processors = structlog.get_config()["processors"]
                self.assertIsInstance(processors[-1], structlog.dev.ConsoleRenderer)

        def test_debug_false_uses_json_renderer(self):
            """When settings.debug is falsy, INFO level and JSONRenderer should be used."""
            with mock.patch.object(settings, "debug", False):
                configure_logging()
                self.assertEqual(logging.getLogger().level, logging.INFO)
                processors = structlog.get_config()["processors"]
                self.assertIsInstance(processors[-1], structlog.processors.JSONRenderer)

        def test_multiple_calls_idempotent(self):
            """Calling configure_logging multiple times should not add duplicate handlers."""
            with mock.patch.object(settings, "debug", True):
                configure_logging()
                first_handler_count = len(logging.root.handlers)
                configure_logging()
                second_handler_count = len(logging.root.handlers)
                self.assertEqual(first_handler_count, second_handler_count)
                self.assertGreater(first_handler_count, 0)

        def test_non_boolean_debug_treated_as_truthy(self):
            """Non‑boolean truthy values for settings.debug should still select DEBUG level."""
            with mock.patch.object(settings, "debug", "yes"):
                configure_logging()
                self.assertEqual(logging.getLogger().level, logging.DEBUG)

    unittest.main(argv=["first-arg-is-ignored"], exit=False)