"""Fail when a function has a CRAP score at or above the limit.

CRAP(f) = cc(f)^2 * (1 - cov(f))^3 + cc(f), with cc the cyclomatic complexity (radon)
and cov the line coverage of the function (coverage.py JSON report, `functions` section).
Usage: uv run python scripts/crap.py coverage.json [--limit 6]
"""

import argparse
import json
import sys
from pathlib import Path

from radon.complexity import cc_visit


def crap(complexity: int, coverage: float) -> float:
    return complexity**2 * (1 - coverage) ** 3 + complexity


def scores(report: dict) -> list[tuple[float, str, int, float]]:
    rows = []
    for filename, data in report["files"].items():
        functions = data.get("functions", {})
        for block in cc_visit(Path(filename).read_text()):
            name = (
                f"{block.classname}.{block.name}"
                if getattr(block, "classname", None)
                else block.name
            )
            summary = functions.get(name, {}).get("summary")
            if summary is None:
                continue
            cov = summary["percent_covered"] / 100
            rows.append(
                (
                    crap(block.complexity, cov),
                    f"{filename}:{block.lineno} {name}",
                    block.complexity,
                    cov,
                )
            )
    return sorted(rows, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--limit", type=float, default=6.0)
    args = parser.parse_args()
    rows = scores(json.loads(args.report.read_text()))
    bad = [r for r in rows if r[0] >= args.limit]
    for score, where, cc, cov in rows[:5]:
        print(f"CRAP {score:5.2f}  cc={cc}  cov={cov:.0%}  {where}")
    print(f"{len(rows)} functions, {len(bad)} with CRAP >= {args.limit}")
    for score, where, cc, cov in bad:
        print(f"FAIL CRAP {score:.2f} cc={cc} cov={cov:.0%} {where}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
