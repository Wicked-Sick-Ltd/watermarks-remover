"""Optional backends must use the patched Pillow release."""

from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "service" / "scripts"


@pytest.mark.parametrize(
    "filename",
    [
        "requirements-synthid-scorer.txt",
        "requirements-ctrlregen.txt",
        "requirements-markllm.txt",
    ],
)
def test_pillow_security_pin(filename):
    lines = (SCRIPTS / filename).read_text(encoding="utf-8").splitlines()
    pins = [line.strip() for line in lines if line.lower().startswith("pillow==")]
    assert pins == ["Pillow==13.0.0"]
