r"""Stage outputs written directly as ``.jsonl.gz``, one gzip member per record.

Rule of ``CLAUDE.md`` ("Long runs"): stage outputs are written packed, every finished unit at
once, and a run is resumable. Every record is appended as its own gzip member (a file of several
members is a valid gzip file; ``gzip.open`` reads them as one stream), flushed and synced, so a
finished unit survives the end of the process. A process killed in the middle of a write can
leave an incomplete last member; :func:`complete_length` finds the end of the last complete member
and :func:`repair_tail` cuts the incomplete rest (a part of a record that was never finished)
before a resume appends to the file.
"""

from __future__ import annotations

import gzip
import json
import os
import zlib
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["append_record", "complete_length", "read_lines", "read_records", "repair_tail"]


def _default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append ``record`` as one line and one gzip member; flushed and synced."""
    if path.suffix != ".gz":
        raise ValueError("packed records are written to a .gz file")
    line = (json.dumps(record, default=_default) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(gzip.compress(line, compresslevel=6, mtime=0))
        handle.flush()
        os.fsync(handle.fileno())


def complete_length(path: Path) -> int:
    """Number of leading bytes of ``path`` that form complete gzip members."""
    data = memoryview(path.read_bytes())
    position = 0
    while position < len(data):
        member = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            member.decompress(data[position:])
        except zlib.error:
            break
        if not member.eof:
            break
        position = len(data) - len(member.unused_data)
    return position


def repair_tail(path: Path) -> int:
    """Cut an incomplete last member off ``path``; returns the number of bytes removed."""
    if not path.exists():
        return 0
    size = path.stat().st_size
    keep = complete_length(path)
    if keep < size:
        with path.open("r+b") as handle:
            handle.truncate(keep)
            handle.flush()
            os.fsync(handle.fileno())
    return size - keep


def read_lines(path: Path) -> list[str]:
    """Lines of a packed (``.gz``) or plain JSONL file; a missing file has none."""
    if not path.exists():
        return []
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read().splitlines()
    return path.read_text(encoding="utf-8").splitlines()


def read_records(path: Path) -> list[dict[str, Any]]:
    """Records of a packed or plain JSONL file."""
    return [json.loads(line) for line in read_lines(path) if line.strip()]
