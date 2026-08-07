import structlog
import logging
from app.config import settings

# Module‑level flag to ensure logging is configured only once.
_logging_configured = False


def configure_logging() -> None:
    """Configure structlog and the standard logging library.

    This function is idempotent – repeated calls will have no effect after the
    first successful configuration, avoiding duplicate handlers and unnecessary
    re‑initialisation overhead.
    """
    global _logging_configured
    if _logging_configured:
        return

    # Determine log level based on the ``debug`` flag.  Non‑boolean truthy values are
    # treated as ``True`` to preserve the original behaviour.
    level = logging.DEBUG if bool(settings.debug) else logging.INFO

    # ``basicConfig`` is only called once because of the early‑exit guard.
    logging.basicConfig(level=level, format="%(message)s")

    # Build the processor chain.  ``StackInfoRenderer`` is expensive and only
    # useful when debugging, so we include it conditionally.
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if bool(settings.debug):
        processors.append(structlog.processors.StackInfoRenderer())
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    _logging_configured = True


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
            # Reset module‑level flag for each test
            global _logging_configured
            _logging_configured = False

        def tearDown(self):
            # Clean up after test to avoid side‑effects for subsequent runs
            logging.root.handlers.clear()
            global _logging_configured
            _logging_configured = False

        def test_debug_true_uses_console_renderer(self):
            """When settings.debug is truthy, DEBUG level and ConsoleRenderer should be used."""
            with mock.patch.object(settings, "debug", True):
                configure_logging()
                self.assertEqual(logging.getLogger().level, logging.DEBUG)
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