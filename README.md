# PE-LCB100 Polymer-Agent

Installable original workflow and interactive case replay for *Validating LLM
Agents for Reliable Polymer Simulations*.

[中文说明](README_CN.md) · [Source](https://github.com/liuze101010/PE-LCB100-demo) ·
[Demo after deployment](https://liuze101010.github.io/PE-LCB100-demo/)

The workflow connects a natural-language goal to design, construction and
validation 1, MD and analysis, validation 2, audit and registered repairs,
revalidation, and archiving. Scientific decisions use the original executable
rules. Actual execution uses the user's own model credentials and compute site.

The public website replays retained paper records. It contains 100 terminal
cases: 92 PASS and 8 REVIEW. The retained audit ledger separately contains 97
cases and 291 model votes, labelled with its own scope.

## Local replay

Use Python 3.12 on Windows or Linux:

```bash
git clone https://github.com/liuze101010/PE-LCB100-demo.git
cd PE-LCB100-demo
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install .
python -m polymer_agent_demo.scripts.doctor --stage demo --json
python -m uvicorn polymer_agent_demo.app:app --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/auto`. Windows also provides `start_demo.cmd`;
Linux provides `bash run_demo.sh`. Replay needs no model key, database, GPU or
cluster account. Target text searches existing records.

## Original scientific tools

```bash
python -m pip install ".[science]"
python science_core/examples/minimal_validation.py
python -m polymer_agent_demo.scripts.doctor --stage local-validation --json
```

The example calls the original RDKit validator and reports one `pass` and one
`reject`. Readable original source, registered tools, schemas, plugins and the
four-stage MDP files are under [science_core](science_core/). Generate the remote
deployment ZIP from this same source with
`python scripts/build_science_core_bundle.py`.

The stdio MCP entry is
`science_core/generated_code/md_analysis/mcp/pelcb_mcp_server.py`.
Fill a local copy of the example project configuration, set `PELCB_PROJECT_CONFIG`,
then start this entry using Python. Tool names and contracts retain the original
framework. See the [MCP guide](science_core/generated_code/md_analysis/mcp/README.md).

## Run your own project

```bash
python -m pip install ".[platform,science]"
```

Windows provides `start_platform.cmd`. For manual startup, copy `.env.example`
to `.env`, fill local settings and set `POLYMER_AGENT_APP_MODE=platform` before
the Uvicorn command above. Use `doctor --stage design`, `--stage audit` or
`--stage md` to check each stage without making model or cluster calls.

Design uses the configured planning key; audit uses the registered three-model
panel and its keys. MD uses your EMC/GROMACS installation and authorized compute
connection. The original preflight checks software, permissions and live
resources. Physical parameters and rule thresholds remain versioned; run
completion and scientific acceptance are recorded separately.

Follow the [deployment guide](deploy/README_DEPLOY.md) for the workbench and
`docker compose`. A standalone replay image uses:

```bash
docker build -t polymer-agent-demo .
docker run --rm -p 127.0.0.1:8765:8765 polymer-agent-demo
```

EMC, GROMACS and force-field assets are obtained under their respective terms.
They are not redistributed in this source release.

## Develop and verify

```bash
python -m pip install -e ".[platform,science,dev]"
python -m pip check
python -m pytest -q
python scripts/security_scan.py
python scripts/build_pages.py
python -m build --wheel --no-isolation
```

`site/` is the generated website. GitHub Actions builds and deploys it using
Pages. Assets use relative paths under `/PE-LCB100-demo/`. CI covers Windows and
Linux on Python 3.12. See [VALIDATION.md](VALIDATION.md) for actual release checks
and [CHANGELOG.md](CHANGELOG.md) for changes.

The two `requirements-lock-py312-*.txt` files record Windows and Linux Python 3.12
runtime dependency resolutions. Install the matching file before installing this
package when reproducing those dependency versions.

This release focuses on the PE-LCB paper. Polyimide work and private production
data remain in their own working directories.

## License and citation

Original workflow code is MIT-licensed. See [LICENSE](LICENSE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [CITATION.cff](CITATION.cff).
Local credentials and runtime files stay outside version control.
