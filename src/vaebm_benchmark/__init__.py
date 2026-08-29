"""vaebm_benchmark: faithful, per-baseline reproduction of published topic
model protocols (FASTopic, GloCOM), each compared against VAE-BM under the
exact conditions of the paper being reproduced.

See the top-level README for how this differs from DTEA
(document-topic-evaluatio-arena): DTEA standardizes one benchmark across
many models; this package instead reproduces N separate, non-standardized
protocols, one per baseline paper.
"""

from __future__ import annotations

try:
    # On networks behind a TLS-inspecting corporate proxy (e.g. Netskope),
    # certifi's bundled CA list does not include the proxy's own root CA,
    # so httpx/huggingface_hub (used by sentence-transformers to download
    # doc_embed_model weights) fail certificate verification even though
    # the OS trust store (which DOES have that root CA installed) would
    # accept the same connection. `truststore` makes Python's ssl module
    # use the OS trust store instead of certifi's - a network-environment
    # fix, unrelated to any protocol/model logic. Safe to import even
    # off such a network (falls through to the normal default trust
    # store); wrapped in try/except so its absence (e.g. an environment
    # that didn't install optional deps yet) never breaks a plain import
    # of this package.
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass
