# Deterministic QA gauntlet. `make qa` runs every gate in order and stops at the first failure.
.PHONY: qa lint types test arch crap mutation smoke docs

qa: lint types test arch crap mutation smoke

lint:
	uv run ruff format --check .
	uv run ruff check .

types:
	uv run ty check

# Unit, property (Hypothesis), integration and acceptance tests, with branch coverage.
test:
	uv run pytest --cov --cov-report=term --cov-report=json

arch:
	uv run lint-imports

crap: test
	uv run python scripts/crap.py coverage.json --limit 6

mutation:
	rm -rf mutants
	uv run mutmut run --max-children 4
	uv run python scripts/mutation_gate.py

# Real CLI on the fake dump fixture: checks the installed entry point end to end.
smoke:
	uv run inat-landscape --help > /dev/null
	uv run python scripts/smoke.py

docs:
	uv run mkdocs build --strict
