import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests/fixtures"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))


def requires_env(*keys):
    missing = [k for k in keys if not os.environ.get(k)]
    return pytest.mark.skipif(bool(missing), reason=f"needs {', '.join(missing)}")


def requires_path(*paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"needs {', '.join(missing)} (run scripts/setup.sh)")
