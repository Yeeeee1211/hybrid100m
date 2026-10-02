"""Local test bootstrap.

The DSH file sandbox denies chmod on temporary directories, so the stdlib
``tempfile`` cleanup path raises PermissionError after every
``TemporaryDirectory`` block and turns passing tests into errors. Overriding
``TemporaryDirectory`` with an ignore-cleanup-errors variant keeps the tests
meaningful without touching test logic.
"""

import shutil
import tempfile

_original = tempfile.TemporaryDirectory


def _tolerate_cleanup(self):
    try:
        self._rmtree(self.name, ignore_errors=True)
    finally:
        shutil.rmtree(self.name, ignore_errors=True)


class _TemporaryDirectory(_original):
    cleanup = _tolerate_cleanup


tempfile.TemporaryDirectory = _TemporaryDirectory
