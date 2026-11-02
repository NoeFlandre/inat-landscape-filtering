"""The Hugging Face dataset card, generated from the run results."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Subset:
    """One threshold config."""

    config: str
    threshold: float
    rows: int


@dataclass(frozen=True)
class CardInfo:
    """Everything the card states about one published run."""

    snapshot: str
    model_id: str
    landscape_prompts: tuple[str, ...]
    closeup_prompts: tuple[str, ...]
    total: int
    subsets: tuple[Subset, ...]
    licenses: Mapping[str, int]
    embedding_dir: str


def render_card(info: CardInfo) -> str:
    """Return the README.md of the dataset."""
    return "\n".join([_yaml(info), _body(info)])


def _yaml(info: CardInfo) -> str:
    configs = ["- config_name: all\n  data_files: data/all/*.parquet\n  default: true"]
    configs += [
        f"- config_name: {s.config}\n  data_files: data/{s.config}/*.parquet" for s in info.subsets
    ]
    configs.append(f"- config_name: embeddings\n  data_files: {info.embedding_dir}/*.parquet")
    return "\n".join(
        [
            "---",
            "license: other",
            "license_name: per-image-creative-commons",
            "pretty_name: iNaturalist landscape filtering",
            "task_categories:\n- image-classification\n- zero-shot-image-classification",
            "tags:\n- inaturalist\n- landscape\n- land-cover\n- land-use\n- geospatial\n- siglip",
            "configs:",
            *configs,
            "---",
        ]
    )


def _body(info: CardInfo) -> str:
    subsets = "\n".join(f"| `{s.config}` | {s.threshold} | {s.rows} |" for s in info.subsets)
    licenses = "\n".join(f"| {k} | {v} |" for k, v in sorted(info.licenses.items()))
    landscape = ", ".join(f'"{p}"' for p in info.landscape_prompts)
    closeup = ", ".join(f'"{p}"' for p in info.closeup_prompts)
    return f"""
# iNaturalist landscape filtering

iNaturalist photos with a zero-shot score that tells if the photo shows a landscape.
Use the photos for land use and land cover work. Each photo keeps its geolocation.

Code: <https://github.com/NoeFlandre/inat-landscape-filtering>

## Source

- Metadata: iNaturalist Open Data, snapshot `{info.snapshot}`.
- Photos: Plantae observations and observations with no identification.
- Sample: a deterministic sample of {info.total} scored photos (pilot run).

## Configs

| Config | Rows |
|---|---|
| `all` | {info.total} (metadata and scores, no image bytes) |
| `embeddings` | one `{info.model_id}` embedding (float16) per photo |

Threshold configs keep the photos with `score_landscape >= threshold`. They include the
`medium` image (longest side 500 px).

| Config | Threshold | Rows |
|---|---|---|
{subsets}

## Scores

The model is `{info.model_id}`. The pipeline scores the `small` image (longest side
240 px). Each score is `sigmoid(scale * cosine + bias)`, as SigLIP computes it. A score is
not a calibrated probability. Look at the photos before you select a threshold.

- `score_landscape`: mean text embedding of the prompts {landscape}.
- `score_landscape_1` ... `score_landscape_{len(info.landscape_prompts)}`: one score for each
  of these prompts, in the same order.
- `score_closeup`: mean text embedding of the prompts {closeup}.

## Licences

Each photo has its own licence, in the `license` column. The `attribution` column gives
the attribution text that the licence requires. CC-BY-NC photos are not for commercial
use: remove them with a filter on `license` if you need commercial use.

| Licence | Photos |
|---|---|
{licenses}

## Known limits

- Obscured coordinates: iNaturalist moves the location of some observations (threatened
  taxa, or a user choice) to a random point in a cell of about 0.2 degrees. The source
  data has no flag for this. Use `positional_accuracy` with care.
- The coordinates give the position of the observer or of the organism, not of the view.
- Zero-shot scores have no human validation yet.
"""
