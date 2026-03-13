"""Privacy stripping — redact secrets before storage."""

import re

_PRIVACY_PATTERNS = [
    (re.compile(r'(?:sk|pk|api)[_-][a-zA-Z0-9_-]{20,}'), '[REDACTED_API_KEY]'),
    (re.compile(r'eyJ[a-zA-Z0-9_-]{20,}\.eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]+'), '[REDACTED_JWT]'),
    (re.compile(r'(?:postgresql?|mysql|mssql|snowflake)://\S+'), '[REDACTED_CONN_STRING]'),
    (re.compile(r'(?:password|passwd|pwd)\s*[=:]\s*\S+', re.IGNORECASE), '[REDACTED_PASSWORD]'),
    (re.compile(r'(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}'), '[REDACTED_GITHUB_TOKEN]'),
    (re.compile(r'xox[bpsar]-[A-Za-z0-9-]{10,}'), '[REDACTED_SLACK_TOKEN]'),
]


def _strip_sensitive(text: str) -> str:
    """Remove API keys, tokens, and passwords from text before storage."""
    for pattern, replacement in _PRIVACY_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
