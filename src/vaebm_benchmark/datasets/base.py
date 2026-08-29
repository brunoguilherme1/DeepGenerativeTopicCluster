"""Core dataset abstractions: an immutable raw/ copy per dataset, a SHA256
manifest recorded once at download time, and a canonical in-memory
LoadedDataset every protocol consumes. Mirrors DTEA's
`dtea.datasets.base` pattern in spirit (not by import - this is a
standalone repo); see this project's README for why the two repos stay
independent rather than sharing a dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from vaebm_benchmark.utils.paths import RAW_DIR
from vaebm_benchmark.utils.provenance import verify_manifest, write_manifest


@dataclass
class LoadedDataset:
    dataset_id: str
    documents: list[str]
    labels: Optional[list[int]]
    label_names: Optional[dict[int, str]]

    def __post_init__(self) -> None:
        if self.labels is not None and len(self.labels) != len(self.documents):
            raise ValueError(
                f"{self.dataset_id}: {len(self.documents)} documents but {len(self.labels)} labels"
            )


class BenchmarkDataset:
    """Subclasses implement `_download_raw()` (fetch to `raw_dir()`) and
    `_load_raw()` (parse the immutable raw copy). Checksums/manifests are
    handled here so definitions stay small and declarative - exactly the
    "record provenance and checksums" requirement this project's own README
    calls out as a hard rule, not an afterthought."""

    dataset_id: str = ""

    def __init__(self) -> None:
        if not self.dataset_id:
            raise ValueError(f"{type(self).__name__} must set dataset_id")

    def raw_dir(self):
        d = RAW_DIR / self.dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def download(self, force: bool = False) -> None:
        manifest_path = self.raw_dir() / "MANIFEST.yaml"
        if manifest_path.exists() and not force:
            return
        self._download_raw()
        write_manifest(self.raw_dir())

    def verify(self) -> tuple[bool, list[str]]:
        return verify_manifest(self.raw_dir())

    def load(self) -> LoadedDataset:
        self.download()
        return self._load_raw()

    def _download_raw(self) -> None:
        raise NotImplementedError

    def _load_raw(self) -> LoadedDataset:
        raise NotImplementedError
