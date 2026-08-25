import contextvars
import logging

from pythonjsonlogger import jsonlogger

# Put the HTTP Request ID in here.
request_id_context = contextvars.ContextVar("request_id", default="SYSTEM")


class RequestIdFilter(logging.Filter):
    """Injects the current request_id into every log record."""

    def filter(self, record):
        record.request_id = request_id_context.get()
        return True


def get_logger(name: str) -> logging.Logger:
    """Returns a configured JSON logger."""
    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers if the logger is requested multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()

        # Define what fields should appear in the JSON output
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
            rename_fields={"levelname": "level"},
        )

        handler.setFormatter(formatter)
        handler.addFilter(RequestIdFilter())

        logger.addHandler(handler)
        logger.propagate = False

    return logger
