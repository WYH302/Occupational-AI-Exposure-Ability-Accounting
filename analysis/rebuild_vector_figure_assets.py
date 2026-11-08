"""Restore preserved vector plots and normalize intended black strokes.

The active Figure 2 and Figure 3 PDFs had previously been rebuilt from PNG
captures.  This utility starts from the preserved vector crops extracted from
the earlier vector manuscript, changes only the declared near-black stroke
operators to DeviceGray zero, and leaves the PDF byte layout/xref offsets
intact.  No plotted coordinates, labels, or numerical values are regenerated.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "paper" / "reconstructed_latest" / "figures"
SOURCE_DIR = FIGURE_DIR / "vector_sources"


@dataclass(frozen=True)
class VectorPatch:
    source_name: str
    output_name: str
    object_number: int
    old_operator: bytes
    new_operator: bytes
    expected_occurrences: int


PATCHES = (
    VectorPatch(
        source_name="distance_vector.pdf",
        output_name="distance_black.pdf",
        object_number=22,
        old_operator=b"0.3019607843 G",
        new_operator=b"0.0000000000 G",
        expected_occurrences=2,
    ),
    VectorPatch(
        source_name="profiles_vector.pdf",
        output_name="profiles_black.pdf",
        object_number=28,
        old_operator=b"0.1333333333 G",
        new_operator=b"0.0000000000 G",
        expected_occurrences=1,
    ),
)


def _stream_span(payload: bytes, object_number: int) -> tuple[int, int]:
    object_match = re.search(
        rb"(?m)^" + str(object_number).encode("ascii") + rb" 0 obj\b(.*?)endobj",
        payload,
        re.DOTALL,
    )
    if object_match is None:
        raise ValueError(f"PDF object {object_number} was not found")
    stream_match = re.search(
        rb"stream\r?\n(.*?)\r?\nendstream",
        object_match.group(1),
        re.DOTALL,
    )
    if stream_match is None:
        raise ValueError(f"PDF object {object_number} has no stream")
    start = object_match.start(1) + stream_match.start(1)
    end = object_match.start(1) + stream_match.end(1)
    return start, end


def normalize_black_strokes(payload: bytes, patch: VectorPatch) -> bytes:
    """Return a same-length PDF with selected near-black strokes set to black."""
    if len(patch.old_operator) != len(patch.new_operator):
        raise ValueError("color replacements must preserve uncompressed length")
    start, end = _stream_span(payload, patch.object_number)
    compressed = payload[start:end]
    decoded = zlib.decompress(compressed)
    occurrences = decoded.count(patch.old_operator)
    if occurrences != patch.expected_occurrences:
        raise ValueError(
            f"expected {patch.expected_occurrences} color operators in "
            f"{patch.source_name}, found {occurrences}"
        )
    normalized = decoded.replace(patch.old_operator, patch.new_operator)

    replacement_stream: bytes | None = None
    for padding in range(129):
        candidate = zlib.compress(normalized + (b" " * padding), level=6)
        if len(candidate) == len(compressed):
            replacement_stream = candidate
            break
    if replacement_stream is None:
        raise ValueError(
            f"could not preserve the compressed stream length for {patch.source_name}"
        )

    output = payload[:start] + replacement_stream + payload[end:]
    if len(output) != len(payload):
        raise AssertionError("same-size patch invariant failed")
    return output


def build_assets() -> list[Path]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for patch in PATCHES:
        source = SOURCE_DIR / patch.source_name
        destination = SOURCE_DIR / patch.output_name
        payload = source.read_bytes()
        if b"/Subtype /Image" in payload:
            raise ValueError(f"{source} is not a vector-only source PDF")
        if b"/Font" not in payload:
            raise ValueError(f"{source} does not expose embedded vector text fonts")
        normalized = normalize_black_strokes(payload, patch)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(normalized)
        temporary.replace(destination)
        outputs.append(destination)
    return outputs


if __name__ == "__main__":
    for generated in build_assets():
        print(generated.relative_to(ROOT).as_posix())
