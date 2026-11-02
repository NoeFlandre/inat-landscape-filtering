# Quality gates

Run all gates with `make qa`. The CI runs the same gates. A failed gate blocks the merge.

| Gate | Tool | Rule |
|---|---|---|
| Format and lint | ruff | Rules E, W, F, I, B, UP, SIM, C90, N, PT, PL, RUF, S, ARG, PTH, TID. Maximum cyclomatic complexity: 5. |
| Types | ty | No errors. |
| Tests | pytest, Hypothesis | Unit, property, integration and acceptance tests. Branch coverage of 90% or more. |
| Architecture | import-linter | Layers: `cli` → `pipeline` → `adapters` → `domain`. The domain imports no I/O or ML library. |
| CRAP | `scripts/crap.py` (radon and coverage) | CRAP score less than 6 for each function. |
| Mutation | mutmut | No surviving mutant in `domain`. |
| Smoke | `scripts/smoke.py` | The installed CLI runs on a fixture dump. |
| Docs | MkDocs | `mkdocs build --strict` is successful. |

## Lint rules

- `C90` keeps functions small. The limit (5) keeps the CRAP score below 6 when the coverage is full.
- `S` finds security problems, for example unsafe URL schemes.
- `TID` forbids relative imports. Imports show the layer of each module.
- In tests, `S101` (assert), `PLR2004` (magic values), `S311` (seeded random) and `ARG` (fakes match real signatures) are allowed.

## Equivalent mutants

Three lines have `# pragma: no mutate`. A mutant of these lines cannot change the result. Each line has a comment that tells why.
