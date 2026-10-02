"""Local test bootstrap for cpu/ integration tests.

The DSH workspace sandbox denies creating/writing stdlib temp directories for
this project, so `tempfile.TemporaryDirectory` fails with PermissionError before
any test runs. Redirect test-scoped temporary directories into the workspace
(honouring TMPDIR when set). Test logic is untouched.
"""

import itertools
import os
import shutil
import tempfile
from pathlib import Path

_original = tempfile.TemporaryDirectory
_counter = itertools.count()

_base = Path(os.environ.get("DSH_TEST_TMPDIR") or Path(__file__).resolve().parent.parent / "tmp" / "cpu-tests")
_base.mkdir(parents=True, exist_ok=True)


class _WorkspaceTemporaryDirectory(_original):
    def __init__(self, suffix=None, prefix=None, dir=None, ignore_cleanup_errors=True, *, delete=True):
        target = Path(dir) if dir else _base
        target.mkdir(parents=True, exist_ok=True)
        name = target / f"{prefix or 'tmp'}{next(_counter)}{suffix or ''}"
        name.mkdir(parents=True, exist_ok=True)
        self.name = str(name)
        self._delete = delete
        self._ignore_cleanup_errors = True

    def cleanup(self):
        if self._delete:
            shutil.rmtree(self.name, ignore_errors=True)

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
        return False


tempfile.TemporaryDirectory = _WorkspaceTemporaryDirectory
