"""Vector serialization and math operations for sqlite-vec."""

import struct

__all__ = ["serialize_f32", "deserialize_f32"]


def serialize_f32(vec: list[float]) -> bytes:
    """Serialize a list of floats into bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_f32(blob: bytes) -> list[float]:
    """Deserialize bytes back to float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    """Element-wise subtraction a - b."""
    return [x - y for x, y in zip(a, b)]


def _vec_scale(v: list[float], s: float) -> list[float]:
    """Scalar multiplication."""
    return [x * s for x in v]


def _norm(v: list[float]) -> float:
    """L2 norm."""
    return sum(x * x for x in v) ** 0.5


def _novelty_score(query: list[float], seed: list[float], neighbor: list[float]) -> float:
    """Compute query-conditioned novelty of neighbor relative to seed.

    Returns cos(query, residual) where residual = neighbor - proj_seed(neighbor).
    Measures how much query-relevant NEW information the neighbor adds beyond the seed.
    Clamped to [0, 1]. Returns 0 if residual is near-zero (neighbor ≈ seed).
    """
    ns_dot = _dot(neighbor, seed)
    proj = _vec_scale(seed, ns_dot)
    residual = _vec_sub(neighbor, proj)
    r_norm = _norm(residual)
    if r_norm < 1e-8:
        return 0.0
    qr_dot = _dot(query, residual)
    return max(0.0, min(1.0, qr_dot / r_norm))
