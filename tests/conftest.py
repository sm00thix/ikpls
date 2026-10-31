"""
Test-session checks shared by all test modules.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import sys
import sysconfig

import pytest


def pytest_sessionfinish(session, exitstatus):
    """
    On a free-threaded interpreter, fail the run if the GIL was re-enabled during it.
    Importing an extension module that does not declare free-threading support
    re-enables the GIL with only a RuntimeWarning, after which the tests would pass
    without exercising free threading. This covers every module the tests import, not
    only the top-level ones checked before the run in CI. Not applied when the GIL was
    requested explicitly (PYTHON_GIL=1 or -X gil=1).
    """
    if not sysconfig.get_config_var("Py_GIL_DISABLED") or sys.flags.gil == 1:
        return
    if sys._is_gil_enabled():
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write("\n")
            reporter.write_sep(
                "=",
                "the GIL was re-enabled during the test session by an imported "
                "extension module without free-threading support",
                red=True,
            )
        if session.exitstatus == pytest.ExitCode.OK:  # keep e.g. "interrupted"
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
