import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
spec = importlib.util.spec_from_file_location("run_experiment_script", SCRIPT_PATH)
run_experiment_script = importlib.util.module_from_spec(spec)
sys.modules["run_experiment_script"] = run_experiment_script
spec.loader.exec_module(run_experiment_script)

_parse_ks = run_experiment_script._parse_ks


def test_parse_ks_space_separated():
    assert _parse_ks(["50", "100"]) == [50, 100]


def test_parse_ks_comma_separated():
    assert _parse_ks(["50,100"]) == [50, 100]


def test_parse_ks_mixed():
    assert _parse_ks(["50,75", "100"]) == [50, 75, 100]


def test_parse_ks_single():
    assert _parse_ks(["50"]) == [50]
