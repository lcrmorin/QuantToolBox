"""Run all standalone examples and guided-page Python blocks in fresh processes.

Run from the repository root with the package and its docs/viz dependencies
installed: ``python scripts/check_examples.py``. No figures are displayed.
Add ``--render-plots --pattern "*.md"`` to regenerate the guided-page images
from the same Python blocks readers run, in their documented order.
Markdown blocks share state within a page, in reading order; pages and
standalone scripts never share state. The JSON report retains stdout,
stderr, return codes, and durations so warnings can be reviewed explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_example(path, timeout, render_plots=False):
    start = time.monotonic()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qtb-examples-mpl"))
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    if path.suffix == ".md":
        blocks = re.findall(r"```python\s*\n(.*?)```", path.read_text(), re.S)
        code = "\n\n".join(textwrap.dedent(block) for block in blocks)
        if render_plots:
            images = re.findall(
                r"!\[[^\]]*\]\((\.\./assets/examples/[^)]+\.png)\)", path.read_text()
            )
            if len(images) != code.count("plt.show()"):
                raise ValueError(f"{path}: each plot needs one documented image")
            for relative_image in images:
                target = (path.parent / relative_image).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                code = code.replace(
                    "plt.show()",
                    f'plt.gcf().savefig({str(target)!r}, dpi=160); plt.close("all")',
                    1,
                )
        else:
            # Render figures on Agg, then close them instead of opening GUI windows.
            code = code.replace("plt.show()", 'plt.gcf().canvas.draw(); plt.close("all")')
        command = [sys.executable, "-c", code]
    else:
        command = [sys.executable, str(path)]
    try:
        run = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout
        )
        result = dict(
            status="passed" if run.returncode == 0 else "failed",
            returncode=run.returncode,
            stdout=run.stdout,
            stderr=run.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        result = dict(
            status="timeout",
            returncode=None,
            stdout=(exc.stdout or b"").decode(errors="replace"),
            stderr=(exc.stderr or b"").decode(errors="replace"),
        )
    return {
        "example": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seconds": round(time.monotonic() - start, 2),
        **result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-plots", action="store_true", help="Regenerate guided-page PNGs from their code"
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--pattern", default="*", help="Match filenames, e.g. 'svm*'")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/examples-execution.json")
    args = parser.parse_args()
    paths = sorted(
        p
        for p in (ROOT / "docs/examples").rglob(args.pattern)
        if p.suffix == ".py" or (p.suffix == ".md" and "```python" in p.read_text())
    )
    if not paths or args.jobs < 1 or args.timeout < 1:
        parser.error("Select at least one example and positive jobs/timeout")
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "dependencies": {
            name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib")
        },
        "results": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_example, p, args.timeout, args.render_plots) for p in paths]
        for future in as_completed(futures):
            result = future.result()
            report["results"].append(result)
            report["results"].sort(key=lambda r: r["example"])
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"{len(report['results'])}/{len(paths)} {result['status']} {result['example']} "
                f"({result['seconds']}s)",
                flush=True,
            )
    report["summary"] = {
        "total": len(paths),
        "passed": sum(r["status"] == "passed" for r in report["results"]),
        "with_stderr": sum(bool(r["stderr"]) for r in report["results"]),
    }
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return int(any(r["status"] != "passed" for r in report["results"]))


if __name__ == "__main__":
    sys.exit(main())
