"""Save each notebook's latest tables and figures in a fixed directory."""
import json
import shutil

import matplotlib.pyplot as plt
from IPython.display import Markdown, display


def begin(root, experiment, seed):
    run = root / "runs" / experiment
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    (run / "settings.json").write_text(json.dumps(dict(seed=seed), indent=2))
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .2, "font.size": 10})
    display(Markdown(f"**결과:** `runs/{experiment}/` · seed `{seed}`"))
    return run


def figure(run, name):
    plt.tight_layout()
    plt.savefig(run / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.savefig(run / f"{name}.svg", bbox_inches="tight")
    plt.show()
    plt.close()


def finish(run, **findings):
    (run / "findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2, default=str))
    display(Markdown("**실행에서 확인한 값**\n\n" + "\n\n".join(f"- **{k}**: {v}" for k, v in findings.items())))
