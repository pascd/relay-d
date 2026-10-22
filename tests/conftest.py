"""Make `relay_d` and the sibling test modules importable without installing the package."""

import os
import sys

# Qt tests run headless
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_TESTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS)
for path in (_ROOT, _TESTS):
    if path not in sys.path:
        sys.path.insert(0, path)

# Tests read templates such as config/*.yaml with repo-relative paths.
os.chdir(_ROOT)

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole run (Qt singletons such as the settings manager are
    deleted if the application object is destroyed between tests)."""
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
