"""Protocol registry - the one place a protocol name (e.g. "fastopic",
"glocom") resolves to its BaselineProtocol class. Deliberately tiny (a
plain dict, not DTEA's decorator-based registry pattern): this project's
own size guideline is "adding one baseline = one adapter + one protocol +
one config," and a two-entry dict already satisfies that without any
registration machinery to maintain."""

from __future__ import annotations

from vaebm_benchmark.protocols.base import BaselineProtocol


def get_protocol(name: str, smoke_test: bool = True) -> BaselineProtocol:
    if name == "fastopic":
        from vaebm_benchmark.protocols.fastopic_protocol import FASTopicProtocol

        return FASTopicProtocol(smoke_test=smoke_test)
    if name == "glocom":
        from vaebm_benchmark.protocols.glocom_protocol import GloCOMProtocol

        return GloCOMProtocol(smoke_test=smoke_test)
    raise KeyError(f"Unknown protocol '{name}'. Available: fastopic, glocom")


def list_protocols() -> list[str]:
    return ["fastopic", "glocom"]
