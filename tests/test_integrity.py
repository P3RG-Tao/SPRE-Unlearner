import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from spre_unlearner import ReplacementConfig

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_reference_hashes():
    provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
    assert (
        provenance["formal_manifest_sha256"]
        == "0b3ee146dbd08c5e5de037afc4fdfb979601c5e7cdd68bc5830c4883cf3d113a"
    )
    for name, record in provenance["files"].items():
        assert (
            hashlib.sha256((ROOT / "tests/reference" / name).read_bytes()).hexdigest()
            == record["extracted_sha256"]
        )


def test_serialized_formal_defaults_match_code():
    assert json.loads((ROOT / "configs/qwen_formal.json").read_text()) == asdict(
        ReplacementConfig()
    )
