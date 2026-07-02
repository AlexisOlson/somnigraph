"""Privacy stripping — redact secrets before storage."""

import re

# Order matters: multi-line PEM blocks are redacted first so nothing inside them
# leaks past a later single-line pattern. All replacements use the [REDACTED_*]
# marker convention so redaction is visible (and greppable) in stored content.
_PRIVACY_PATTERNS = [
    # PEM private-key blocks (RSA/EC/OPENSSH/generic). DOTALL so the body spans lines.
    (re.compile(r'-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----', re.DOTALL), '[REDACTED_PRIVATE_KEY]'),
    (re.compile(r'(?:sk|pk|api)[_-][a-zA-Z0-9_-]{20,}'), '[REDACTED_API_KEY]'),
    (re.compile(r'eyJ[a-zA-Z0-9_-]{20,}\.eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]+'), '[REDACTED_JWT]'),
    # Bearer tokens in Authorization headers / pasted curl commands.
    (re.compile(r'\bbearer\s+[A-Za-z0-9._~+/=-]{20,}', re.IGNORECASE), '[REDACTED_BEARER_TOKEN]'),
    (re.compile(r'(?:postgresql?|mysql|mssql|snowflake|mongodb(?:\+srv)?|redis|amqp)://\S+'), '[REDACTED_CONN_STRING]'),
    (re.compile(r'(?:password|passwd|pwd)\s*[=:]\s*\S+', re.IGNORECASE), '[REDACTED_PASSWORD]'),
    (re.compile(r'(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}'), '[REDACTED_GITHUB_TOKEN]'),
    (re.compile(r'xox[bpsar]-[A-Za-z0-9-]{10,}'), '[REDACTED_SLACK_TOKEN]'),
    # AWS access key IDs and Google API keys — fixed, high-signal prefixes.
    (re.compile(r'\bAKIA[0-9A-Z]{16}\b'), '[REDACTED_AWS_KEY]'),
    (re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b'), '[REDACTED_GOOGLE_API_KEY]'),
    # Credit-card numbers, anchored to major-network prefixes (Visa/MC/Amex/Discover)
    # to avoid clobbering ordinary long digit strings.
    (re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'), '[REDACTED_CREDIT_CARD]'),
]


def _strip_sensitive(text: str) -> str:
    """Remove API keys, tokens, and passwords from text before storage."""
    for pattern, replacement in _PRIVACY_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
