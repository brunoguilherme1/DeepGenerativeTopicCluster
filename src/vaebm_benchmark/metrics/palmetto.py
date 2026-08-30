"""Real Palmetto + Wikipedia C_V - the ACTUAL metric both FASTopic and
GloCOM report in their papers (see each protocol's module docstring).

This must never be conflated with the gensim-CoherenceModel-against-the-
training-corpus fallback in metrics/topic_quality.py::coherence() - they
are reported under distinct metric names everywhere in this project
(`cv_palmetto_wikipedia` vs. `cv_local_corpus`) specifically so a
comparison table can never present one as if it were the other.

Usage: place `tools/palmetto/palmetto.jar` and extract the official
`Wikipedia_bd` index under `tools/palmetto/wiki_data/wikipedia_bd/` (same
layout GloCOM's own repo documents). When either is missing,
`palmetto_cv()` raises `PalmettoUnavailable` - callers (protocols/base.py
`evaluate()`) catch this and record the metric as unavailable rather
than silently falling back to a different computation under the same
name.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

DEFAULT_JAR = Path("tools/palmetto/palmetto.jar")
DEFAULT_WIKI_INDEX = Path("tools/palmetto/wiki_data/wikipedia_bd")


class PalmettoUnavailable(Exception):
    """Raised when the jar and/or Wikipedia index are not present, or the
    java subprocess itself fails - never caught silently into a
    substitute number."""


def palmetto_available(jar: Path = DEFAULT_JAR, wiki_index: Path = DEFAULT_WIKI_INDEX) -> bool:
    return Path(jar).exists() and Path(wiki_index).is_dir()


def palmetto_cv(
    topics: list[list[str]],
    jar: Path = DEFAULT_JAR,
    wiki_index: Path = DEFAULT_WIKI_INDEX,
    top_n: int = 10,
) -> float:
    jar = Path(jar)
    wiki_index = Path(wiki_index)
    if not palmetto_available(jar, wiki_index):
        raise PalmettoUnavailable(
            f"Palmetto jar ({jar}) and/or Wikipedia index ({wiki_index}) not found. "
            "cv_palmetto_wikipedia is unavailable - not substituted with another metric."
        )
    truncated = [words[:top_n] for words in topics]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(" ".join(topic) for topic in truncated))
        topics_path = handle.name
    try:
        result = subprocess.run(
            ["java", "-jar", str(jar), str(wiki_index), "C_V", topics_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise PalmettoUnavailable(f"Palmetto invocation failed: {exc}") from exc
    finally:
        Path(topics_path).unlink(missing_ok=True)

    values = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                values.append(float(parts[1]))
            except ValueError:
                continue
    if not values:
        raise PalmettoUnavailable("Palmetto ran but returned no parseable per-topic C_V values.")
    return sum(values) / len(values)
