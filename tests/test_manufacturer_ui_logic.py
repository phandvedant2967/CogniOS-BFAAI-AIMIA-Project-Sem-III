"""Runs the Node tests for the manufacturer dashboard renderers (tests/js/manufacturer.test.js)."""
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_manufacturer_js_logic_and_renderers():
    proc = subprocess.run(["node", "--test", os.path.join(HERE, "js", "manufacturer.test.js")],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
