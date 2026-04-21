"""
GoogleTest runner -- executes the rdctst binary in embedded batch mode.
"""

import subprocess
import os
import pytest


class TestGoogleTestSuite:
    """Run the rdctst GoogleTest binary and report results."""

    def test_rdctst_embedded_batch(self, rdctst_path, has_gpu):
        if not has_gpu:
            pytest.skip("No AMD GPU -- cannot run rdctst")

        env = os.environ.copy()
        env["RDC_DISABLE_ROCP"] = "yes"

        result = subprocess.run(
            [rdctst_path, "-b"], capture_output=True, text=True, timeout=300, env=env
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        assert result.returncode == 0, (
            f"rdctst batch mode failed (rc={result.returncode}):\n{result.stderr[-2000:]}"
        )

    def test_rdctst_list_tests(self, rdctst_path):
        result = subprocess.run(
            [rdctst_path, "--gtest_list_tests"], capture_output=True, text=True, timeout=30
        )
        combined = result.stdout + result.stderr
        assert len(combined) > 0, "rdctst --gtest_list_tests produced no output"
