# qcatch/logger.py
import logging
from typing import List

class QCatchLogger(logging.Logger):
    _collected_warnings: List[str] = []

    def record_warning(self, msg: str):
        self.warning(msg)
        self._collected_warnings.append(msg)

    def get_record_log(self) -> List[str]:
        return list(self._collected_warnings)

    def clear_log(self):
        self._collected_warnings.clear()

logging.setLoggerClass(QCatchLogger)

def setup_logger(name, verbose) -> QCatchLogger:
    """
    Configure and return a package-wide logger.
    Call this once (e.g. at program start).
    """
    logger = logging.getLogger(name)
    
    # Remove all existing handlers from the root logger
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    # Suppress noisy third-party libraries
    logging.getLogger('numba').setLevel(logging.WARNING)

    # Set logging level based on the verbose flag
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s :\n %(message)s"
    )


    return logger

def generate_warning_html(warning_list: List[str]) -> str:
    """
    Return HTML snippet for displaying a dynamic warning message.
    This can be injected into an element with a specific ID.
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