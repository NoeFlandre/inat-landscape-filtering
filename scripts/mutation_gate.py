"""Fail when a mutant survives. Run after `mutmut run`."""

import subprocess
import sys

result = subprocess.run(["mutmut", "results"], capture_output=True, text=True, check=False)
survivors = [line.strip() for line in result.stdout.splitlines() if ": " in line]
for line in survivors:
    print(f"SURVIVOR {line}", file=sys.stderr)
print(f"{len(survivors)} surviving mutants")
sys.exit(1 if survivors else 0)
