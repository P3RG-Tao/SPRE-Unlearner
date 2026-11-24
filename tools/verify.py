"""CPU验证与可公开校验记录；不加载模型、不调用API。"""

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

import torch

ROOT = Path(__file__).resolve().parents[1]
environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="")
checks = []
for arguments in (
    [sys.executable, "-m", "pytest", "-q"],
    [sys.executable, "-m", "spre_unlearner", "smoke"],
    [
        "ruff",
        "check",
        "src",
        "tests/test_algorithm.py",
        "tests/test_qwen.py",
        "tests/test_integrity.py",
        "tools",
        "examples",
    ],
):
    result = subprocess.run(arguments, cwd=ROOT, env=environment, capture_output=True, text=True)
    display_command = ["python" if a == sys.executable else a for a in arguments]
    checks.append(
        dict(
            command=display_command,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    )
    if result.returncode:
        print(result.stdout + result.stderr)
        raise SystemExit(result.returncode)
report = dict(
    method="SPRE-Unlearner",
    generated_at_utc=datetime.now(timezone.utc).isoformat(),
    python=platform.python_version(),
    torch=torch.__version__,
    device="cpu",
    real_model_tested=False,
    gpu_used=False,
    checks=checks,
)
(ROOT / "TEST_RESULTS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
paths = sorted(
    p
    for p in ROOT.rglob("*")
    if p.is_file()
    and not any(
        part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "build", "dist"}
        or part.endswith(".egg-info")
        for part in p.relative_to(ROOT).parts
    )
    and p.name != "SHA256SUMS"
)
(ROOT / "SHA256SUMS").write_text(
    "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n" for p in paths)
)
print(json.dumps(dict(passed=True, files=len(paths), tests=checks[0]["stdout"].strip())))
