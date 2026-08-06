import structlog
import logging
from app.config import settings


def configure_logging() -> None:
    level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
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

        def test_debug_none_uses_info_level_and_json_renderer(self):
            """When settings.debug is None, INFO level and JSONRenderer should be used."""
            with mock.patch.object(settings, "debug", None):
                configure_logging()
                self.assertEqual(logging.getLogger().level, logging.INFO)
                processors = structlog.get_config()["processors"]
                self.assertIsInstance(processors[-1], structlog.processors.JSONRenderer)

        def test_logger_factory_is_print_logger_factory(self):
            """The logger_factory should be an instance of PrintLoggerFactory."""
            with mock.patch.object(settings, "debug", True):
                configure_logging()
                logger_factory = structlog.get_config()["logger_factory"]
                self.assertIsInstance(logger_factory, structlog.PrintLoggerFactory)

        def test_logging_format_is_message_only(self):
            """The logging formatter should be set to only output the message."""
            with mock.patch.object(settings, "debug", False):
                configure_logging()
                # There should be at least one handler configured by basicConfig
                self.assertGreaterEqual(len(logging.root.handlers), 1)
                formatter = logging.root.handlers[0].formatter
                self.assertEqual(formatter._fmt, "%(message)s")

    unittest.main(argv=["first-arg-is-ignored"], exit=False)