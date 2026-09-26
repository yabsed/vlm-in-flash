"""Provenance and artifacts only; experiment logic stays in the notebooks."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import Markdown, display


def begin(root, experiment, seed):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run = root / "runs" / f"{experiment}-{stamp}-{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True)
    sources = {}
    for path in sorted(root.glob("*.py")) + [root / f"{experiment}.ipynb", root / f"{experiment}.md"]:
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text())
            content = json.dumps([cell["source"] for cell in notebook["cells"]], ensure_ascii=False).encode()
        else:
            content = path.read_bytes()
        sources[path.name] = hashlib.sha256(content).hexdigest()
    meta = dict(experiment=experiment, run_id=run.name, seed=seed,
                started_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                python=sys.version, sources_sha256=sources,
                versions={name: importlib.metadata.version(name) for name in
                          ["numpy", "pandas", "matplotlib", "scipy", "torch", "transformers", "nbclient"]},
                git=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                provenance="All numerical results are computed in this execution; no result inputs.")
    (run / "metadata.json").write_text(json.dumps(meta, indent=2))
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .2, "font.size": 10})
    display(Markdown(f"**이번 실행:** `{run.name}` · seed `{seed}` · 새 계산/측정"))
    return run


def figure(run, name):
    plt.tight_layout()
    plt.savefig(run / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.savefig(run / f"{name}.svg", bbox_inches="tight")
    plt.show()
    plt.close()


def finish(run, **findings):
    findings["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (run / "findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2, default=str))
    display(Markdown("**실행에서 확인한 값**\n\n" + "\n\n".join(f"- **{k}**: {v}" for k, v in findings.items())))
