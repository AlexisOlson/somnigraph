"""Backward-compat shim — production scripts import from memory.sync."""
from memory.events import _row_get, _log_event  # noqa: F401
