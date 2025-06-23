import logging


class QCatchLogger(logging.Logger):
    """Custom logger class for QCatch that collects warnings for later retrieval."""

    _collected_warnings: list[str] = []

    def record_warning(self, msg: str):
        """Record a warning message and store it in the internal list."""
        self.warning(msg)
        self._collected_warnings.append(msg)

    def get_record_log(self) -> list[str]:
        """Retrieve the list of recorded warning messages."""
        return list(self._collected_warnings)

    def clear_log(self):
        """Clear the list of recorded warning messages."""
        self._collected_warnings.clear()


logging.setLoggerClass(QCatchLogger)


def setup_logger(name: str, verbose: bool) -> QCatchLogger:
    """
    Configure and return a package-wide logger.

    Parameters
    ----------
    name
        The name of the logger.
    verbose
        If True, sets the logging level to DEBUG; otherwise, sets it to INFO.

    Returns
    -------
    QCatchLogger
        The configured logger instance.
    """
    # Forcefully create a new QCatchLogger and assign it in the loggerDict
    new_logger = QCatchLogger(name)
    logging.Logger.manager.loggerDict[name] = new_logger

    # Remove all existing handlers from the root logger
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)

    # Suppress noisy third-party libraries
    logging.getLogger("numba").setLevel(logging.WARNING)

    # Set logging level based on the verbose flag
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s - %(levelname)s :\n %(message)s"
    )

    return new_logger


def generate_warning_html(warning_list: list[str]) -> str:
    """
    Generate an HTML snippet for displaying a dynamic warning message.

    Parameters
    ----------
    warning_list
        A list of warning messages to include in the HTML.

    Returns
    -------
    warning_html
        An HTML string containing the warnings. Returns an empty string if no warnings are provided.
    """
    if not warning_list or len(warning_list) == 0:
        return ""

    # Build a bullet-style list of warnings
    inner = "".join(f"- {msg}<br>" for msg in warning_list)
    warning_html = f"""
    <div class="alert alert-warning" role="alert">
        <strong>⚠️ Low Data Quality Detected</strong>
        <button class="btn btn-sm btn-link p-0" type="button" data-bs-toggle="collapse" data-bs-target="#warningDetails" aria-expanded="false" aria-controls="warningDetails">
            (Show details)
        </button>
        <div class="collapse mt-2" id="warningDetails">
            <div class="alert alert-warning">
                {inner}
            </div>
        </div>
    </div>
    """
    return warning_html


__all__ = ["setup_logger"]
