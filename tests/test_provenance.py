from vaebm_benchmark.utils.provenance import sha256_of, verify_manifest, write_manifest


def test_write_and_verify_manifest_roundtrip(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")

    write_manifest(tmp_path)
    ok, problems = verify_manifest(tmp_path)
    assert ok
    assert problems == []


def test_verify_manifest_detects_tampering(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    write_manifest(tmp_path)

    (tmp_path / "a.txt").write_text("tampered")
    ok, problems = verify_manifest(tmp_path)
    assert not ok
    assert any("checksum mismatch" in p for p in problems)


def test_verify_manifest_detects_missing_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    write_manifest(tmp_path)

    (tmp_path / "a.txt").unlink()
    ok, problems = verify_manifest(tmp_path)
    assert not ok
    assert any("missing file" in p for p in problems)


def test_sha256_is_deterministic(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("same content")
    assert sha256_of(f) == sha256_of(f)
