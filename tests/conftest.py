"""Test bootstrap: make the project root importable and silence the harmless
openpyxl 'Data Validation extension' UserWarning emitted for the real Amazon
sample workbooks."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Data Validation extension is not supported:UserWarning",
    )
