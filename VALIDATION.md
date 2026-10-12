# Release validation

Version 1.2.1 was checked in a Windows CPython 3.12.15 environment.
The installed-wheel checks ran outside the source directory.

| Check | Result |
| --- | --- |
| Complete offline application tests | 114 passed; one third-party TestClient deprecation warning |
| Portable scientific source, bundle and fixture checks | Passed |
| Default scientific configuration parsing | Passed; GROMACS 2024.4 and original rules retained |
| Dependency consistency and public-tree security scan | Passed |
| Static guide and retained interface | 3 Python tests and 6 Node tests passed |
| Desktop and mobile guide | Passed at 1440 px and 390 px; copy controls and task templates verified |
| Codex registration example | Accepted by the installed CLI in an isolated configuration directory |
| Recipient project configuration | Eight delegated subprocess commands checked offline |
| Replay-only wheel installation | Health and replay page passed; production APIs unavailable in replay mode |
| Full scientific wheel installation | Original valid input passed and invalid input rejected; audit and MCP modules imported |
| Scientific deployment ZIP | 153 files; source and fixture hashes verified |

GitHub Actions runs the final source suite on Windows and Linux with Python 3.12.
The workflow results are available under the repository's Actions tab.

Three deeply nested acceptance-fixture copies are omitted from the wheel's
readable-source tree to accommodate Windows path limits. Their original bytes
remain in the readable Git repository and the same-version deployment ZIP.
Installed-wheel bundle generation restores only these three fixtures after
checking their recorded hashes. Other missing recorded source files cause an
error.

The checks used retained records and local tools. They made no model calls and
started no production molecular-dynamics jobs. EMC, GROMACS and remote-site
installation are checked by the user's configured deployment preflight.
