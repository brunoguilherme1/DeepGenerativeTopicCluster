"""vaebm_benchmark: faithful, per-baseline reproduction of published topic
model protocols (FASTopic, GloCOM), each compared against VAE-BM under the
exact conditions of the paper being reproduced.

See the top-level README for how this differs from DTEA
(document-topic-evaluatio-arena): DTEA standardizes one benchmark across
many models; this package instead reproduces N separate, non-standardized
protocols, one per baseline paper.
"""

from __future__ import annotations
