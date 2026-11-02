"""The Hugging Face dataset card, generated from the run results."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

REPO_URL = "https://github.com/NoeFlandre/inat-landscape-filtering"
_CELL_AREAS = {5: "about 250 km²", 6: "about 36 km²", 7: "about 5 km²"}
_SIZE_CATEGORIES = (
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
)


@dataclass(frozen=True)
class Subset:
    """One threshold config."""

    config: str
    threshold: float
    rows: int


@dataclass(frozen=True)
class Breakdown:
    """Photos of one category, in total and at or above each threshold."""

    name: str
    rows: int
    kept: tuple[int, ...]


@dataclass(frozen=True)
class Field:
    """One documented column."""

    names: tuple[str, ...]
    type: str
    description: str


@dataclass(frozen=True)
class CardInfo:
    """Everything the card states about one published run."""

    repo_id: str
    snapshot: str
    model_id: str
    landscape_prompts: tuple[str, ...]
    closeup_prompts: tuple[str, ...]
    negative_prompts: tuple[str, ...]
    fraction: float
    candidates: int
    total: int
    subsets: tuple[Subset, ...]
    groups: tuple[Breakdown, ...]
    licenses: tuple[Breakdown, ...]
    quantiles: tuple[tuple[str, float], ...]
    years: tuple[str, str]
    embedding_dir: str
    embedding_dim: int
    embedding_rows: int
    no_location: int
    score: str
    cap_per_cell: int
    embed_cap: int
    h3_res: int


def fields(n_prompts: int) -> tuple[Field, ...]:
    """Return the documented columns of the data configs."""
    per_prompt = tuple(f"score_landscape_{i}" for i in range(1, n_prompts + 1))
    return (
        Field(("photo_id", "photo_uuid"), "int / string", "iNaturalist photo ids"),
        Field(("observation_uuid",), "string", "iNaturalist observation id"),
        Field(("latitude", "longitude"), "float", "Location of the observation (WGS 84)"),
        Field(("positional_accuracy",), "int", "Accuracy of the location, in metres (often empty)"),
        Field(("observed_on",), "string", "Date of the observation (`YYYY-MM-DD`)"),
        Field(
            ("taxon_id", "taxon_name"), "int / string", "Taxon of the observation (empty if none)"
        ),
        Field(("group",), "string", "`plantae` or `unidentified`"),
        Field(("quality_grade",), "string", "`research`, `needs_id` or `casual`"),
        Field(("anomaly_score",), "float", "iNaturalist location anomaly score (often empty)"),
        Field(("position",), "int", "Position of the photo in its observation (0 is the first)"),
        Field(("width", "height", "extension"), "int / string", "Size and format of the original"),
        Field(("license",), "string", "`CC0`, `CC-BY` or `CC-BY-NC`"),
        Field(("attribution",), "string", "Attribution text that the licence requires"),
        Field(("observer_id", "login", "observer_name"), "int / string", "Photographer"),
        Field(
            ("url_small", "url_medium"), "string", "240 px and 500 px photo on the open data bucket"
        ),
        Field(("score_landscape",), "float", "Main score, see [Scores](#scores)"),
        Field(per_prompt, "float", "Score of each landscape prompt, in order"),
        Field(("score_closeup",), "float", "Score of the close-up prompts"),
        Field(("score_contrast",), "float", "Landscape minus specimen prompts: cosine in [-1, 1]"),
        Field(("h3_cell",), "int", "[H3](https://h3geo.org) cell of the location"),
        Field(("sample_rank",), "int", "Position of the photo in the sample order"),
        Field(("image",), "image", "The 500 px photo (threshold configs only)"),
    )


def breakdown(categories: Sequence[str], kept: Sequence[Sequence[bool]]) -> tuple[Breakdown, ...]:
    """Count photos per category, in total and in each kept set, largest category first."""
    if any(len(mask) != len(categories) for mask in kept):
        raise ValueError("each kept mask must have one value per photo")
    names = sorted(set(categories), key=lambda n: (-categories.count(n), n))
    return tuple(
        Breakdown(n, categories.count(n), tuple(_count_kept(categories, m, n) for m in kept))
        for n in names
    )


def _count_kept(categories: Sequence[str], mask: Sequence[bool], name: str) -> int:
    # Lengths are checked in `breakdown`, so the `strict` mutants are equivalent.
    return sum(k for c, k in zip(categories, mask, strict=True) if c == name)  # pragma: no mutate


def size_category(n: int) -> str:
    """Return the Hugging Face size category of a dataset with n rows."""
    return next((label for bound, label in _SIZE_CATEGORIES if n < bound), "n>10M")


def snapshot_label(source: str) -> str:
    """Return the date of an Open Data snapshot (YYYY-MM-DD), or its file name if it has none."""
    match = re.search(r"(\d{4})(\d{2})(\d{2})\.tar\.gz$", source)
    return "-".join(match.groups()) if match else f"`{_file_name(source)}`"


def _file_name(source: str) -> str:
    return source.rpartition("/")[2]


def render_card(info: CardInfo) -> str:
    """Return the README.md of the dataset."""
    return "\n".join(
        [
            _yaml(info),
            _intro(info),
            _quick_start(info),
            _contents(info),
            _fields(info),
            _files(info),
            _scores(info),
            _method(info),
            _license(),
        ]
    )


def _yaml(info: CardInfo) -> str:
    configs = [("all", "data/all/*.parquet", True)]
    configs += [(s.config, f"data/{s.config}/*.parquet", False) for s in info.subsets]
    configs.append(("embeddings", f"{info.embedding_dir}/*.parquet", False))
    blocks = [
        f"- config_name: {name}\n  data_files:\n  - split: train\n    path: {path}"
        + ("\n  default: true" if default else "")
        for name, path, default in configs
    ]
    return "\n".join(
        [
            "---",
            "license: [cc0-1.0, cc-by-4.0, cc-by-nc-4.0]",
            "pretty_name: iNaturalist Landscape Filtering",
            "task_categories: [image-classification, zero-shot-image-classification]",
            "tags: [inaturalist, landscape, land-use, land-cover, geospatial, remote-sensing, "
            "ground-level-imagery, siglip]",
            f"size_categories: [{size_category(info.total)}]",
            "configs:",
            *blocks,
            "---",
        ]
    )


def _intro(info: CardInfo) -> str:
    low = info.subsets[0]
    return f"""
![iNaturalist Landscape Filtering: a deterministic sample of iNaturalist photos of plants and \
unidentified observations is scored for landscape content by SigLIP2; photos above each \
threshold keep their image](assets/hero.png)

# iNaturalist Landscape Filtering

**{info.total:,} iNaturalist photos of plants and unidentified observations \
({info.total - info.no_location:,} with a location), each with a zero-shot SigLIP2 score \
that tells if the photo shows a landscape.**

**{low.rows:,} photos reach the lowest threshold ({low.threshold} on `{info.score}`, at most \
{info.cap_per_cell} per H3 cell) and come with their 500 px image. All {info.total:,} come \
with a {info.embedding_dim}-dimension \
[`{info.model_id}`](https://huggingface.co/{info.model_id}) image embedding.**

The photos come from the [iNaturalist Open Data]\
(https://github.com/inaturalist/inaturalist-open-data) snapshot of \
{snapshot_label(info.snapshot)}, observed from {info.years[0]} to {info.years[1]}. Every row is \
one photo with its location, date, taxon, licence and scores. The goal is a set of \
ground-level photos that show land use and land cover, to pair with remote sensing. \
{_progress(info)} The scores have no human validation yet.

![How full the H3 cells are against the cap of {info.embed_cap} photos, with the full sample \
in grey](assets/map.png)
"""


def _quick_start(info: CardInfo) -> str:
    repo, top = info.repo_id, info.subsets[-1].config
    return f"""## Quick start

```python
from datasets import load_dataset

photos = load_dataset("{repo}", "{top}", split="train")  # strictest threshold, with images
scores = load_dataset("{repo}", split="train")  # all photos: metadata and scores
emb = load_dataset("{repo}", "embeddings", split="train")  # photo_id, embedding
```

Pick your own threshold, or join the embeddings, with DuckDB:

```sql
SELECT a.photo_id, a.latitude, a.longitude, a.score_landscape, e.embedding
FROM 'hf://datasets/{repo}/data/all/*.parquet' a
JOIN 'hf://datasets/{repo}/{info.embedding_dir}/*.parquet' e USING (photo_id)
WHERE a.score_landscape >= 0.001 AND a.license <> 'CC-BY-NC';
```
"""  # noqa: S608 - documentation text, not a query


def _contents(info: CardInfo) -> str:
    return f"""## Contents

{_configs_table(info)}

![Distribution of {info.score}, with the thresholds](assets/scores.png)

Score quantiles, over all {info.total:,} photos:

{_quantiles_table(info)}

{_breakdown_table("Group", info.groups, info.subsets)}

{_breakdown_table("Licence", info.licenses, info.subsets)}
"""


def _configs_table(info: CardInfo) -> str:
    rows = [f"| `all` (default) | {info.total:,} | Metadata and scores, no image |"]
    rows += [
        f"| `{s.config}` | {s.rows:,} | `{info.score}` ≥ {s.threshold}, best "
        f"{info.cap_per_cell} per cell, with the 500 px image |"
        for s in info.subsets
    ]
    rows.append(
        f"| `embeddings` | {info.embedding_rows:,} | `photo_id` and the SigLIP2 image embedding "
        f"({info.embedding_dim} float16 values) |"
    )
    return "| Config | Rows | Content |\n|---|---:|---|\n" + "\n".join(rows)


def _quantiles_table(info: CardInfo) -> str:
    names = " | ".join(name for name, _ in info.quantiles)
    rule = "|".join("---:" for _ in info.quantiles)
    values = " | ".join(f"{v:.2g}" for _, v in info.quantiles)
    return f"| {names} |\n|{rule}|\n| {values} |"


def _breakdown_table(title: str, items: tuple[Breakdown, ...], subsets: tuple[Subset, ...]) -> str:
    heads = " | ".join(f"≥ {s.threshold}" for s in subsets)
    rule = "|".join("---:" for _ in subsets)
    return f"| {title} | Photos | {heads} |\n|---|---:|{rule}|\n{_breakdown_rows(items)}"


def _breakdown_rows(items: tuple[Breakdown, ...]) -> str:
    return "\n".join(
        f"| {b.name} | {b.rows:,} | " + " | ".join(f"{k:,}" for k in b.kept) + " |" for b in items
    )


def _fields(info: CardInfo) -> str:
    rows = "\n".join(
        f"| {', '.join(f'`{n}`' for n in f.names)} | {f.type} | {f.description} |"
        for f in fields(len(info.landscape_prompts))
    )
    return f"""## Fields

| Field | Type | Description |
|---|---|---|
{rows}

The `embeddings` config has one row per photo: `photo_id` and `embedding` (list of \
{info.embedding_dim} float16). Join it to the other configs on `photo_id`.
"""


def _files(info: CardInfo) -> str:
    return f"""## Files

```
data/all/*.parquet                      all photos: metadata and scores
data/<config>/*.parquet                 photos at or above a threshold, with the 500 px image
{info.embedding_dir}/*.parquet   photo_id and SigLIP2 embedding
assets/                                 figures of this card
```

Configs: `all` (default), {", ".join(f"`{s.config}`" for s in info.subsets)}, `embeddings`.
"""


def _scores(info: CardInfo) -> str:
    landscape = "\n".join(f"   {i}. `{p}`" for i, p in enumerate(info.landscape_prompts, 1))
    closeup = ", ".join(f"`{p}`" for p in info.closeup_prompts)
    negative = ", ".join(f"`{p}`" for p in info.negative_prompts)
    return f"""## Scores

Each score is `sigmoid(scale * cosine + bias)` between the image embedding and a text \
embedding, as SigLIP computes it. A prompt set uses the mean of its unit text embeddings.

* `score_landscape` uses these prompts; `score_landscape_<i>` is the score of prompt `i` alone:
{landscape}
* `score_closeup` uses {closeup}.
* `score_contrast` is the cosine between the image and the direction "mean landscape prompt \
minus mean specimen prompt", with the specimen prompts {negative}. It is high for a view of \
the land and low for a close view of one organism or of its traces.

`score_landscape` is a SigLIP sigmoid. It is not a calibrated probability, and it is small: \
most photos score below 0.0001. `score_contrast` is a cosine. For both, the order is what \
matters. A visual check of random photos in each score band (November 2026) gave: at or above \
0.002 on `score_landscape` or 0.08 on `score_contrast`, almost all photos are scene views, \
with a few flower close-ups; between 0.04 and 0.06 on `score_contrast`, about half are close \
views of vegetation. `score_contrast` drops close views of the ground that `score_landscape` \
keeps, and adds views of forests, trees and water that `score_landscape` misses.
"""


def _method(info: CardInfo) -> str:
    return f"""## Methodology

1. **Source.** The metadata dump [`{_file_name(info.snapshot)}`]({info.snapshot}) \
is read once as a stream (36.7 GB compressed); it is not stored.
2. **Sample.** An observation is in the sample when the first 8 hex digits of its UUID are \
below `{info.fraction} * 16^8`. UUID v4 values are random, so the sample is random and \
deterministic. Photos are in (observation UUID, photo position, photo id) order. Within each \
[H3](https://h3geo.org) cell (resolution {info.h3_res}, \
{_CELL_AREAS.get(info.h3_res, "see H3")}), only the first {info.embed_cap:,} photos in that \
order are kept, so dense areas do not dominate; a larger cap keeps a superset. Photos with no \
location are left out. This gave {info.candidates:,} photos.
3. **Taxa.** Only Plantae (taxon 47126 and its descendants) and observations with no taxon \
or the root taxon Life. Habitat and vegetation photos are frequent there.
4. **Licences.** Only photos under CC0, CC-BY or CC-BY-NC, the licences of the open data \
bucket.
5. **Embeddings.** The `small` photo (240 px longest side) goes through \
[`{info.model_id}`](https://huggingface.co/{info.model_id}). Download and decode cost more \
than the model at scale; the model input is 224 px anyway.
6. **Scores.** Zero-shot, from the stored embeddings (see [Scores](#scores)).
7. **Thresholds.** {", ".join(str(s.threshold) for s in info.subsets)} on `{info.score}`. \
Each threshold config keeps at most {info.cap_per_cell} photos per cell, the best scored, with \
their `medium` image (500 px).

Code, with the decisions and their tradeoffs: [{REPO_URL.removeprefix("https://")}]\
({REPO_URL}).

## Known limits

* **Obscured locations.** iNaturalist moves some locations (threatened taxa, user choice) \
to a random point in a cell of about 0.2°. The source has no flag for this.
{_missing_locations(info.no_location)}* **What the location means.** It is where the observer or \
the organism was, not the area in view.
* **No validation.** The thresholds come from a visual check, not from labels.
* **Bias.** iNaturalist users are most active in North America and Europe; the map shows it.
"""


def _progress(info: CardInfo) -> str:
    if info.total < info.candidates:
        return (
            f"The run is in progress: {info.total:,} of the {info.candidates:,} sample "
            "photos are scored."
        )
    return f"All {info.candidates:,} sample photos are scored."


def _missing_locations(count: int) -> str:
    if count == 0:
        return ""
    return (
        f"* **Missing locations.** {count:,} photos have no coordinates. Filter on "
        "`latitude IS NOT NULL`.\n"
    )


def _license() -> str:
    return """## License

Each photo keeps its own licence, in `license`, and the attribution it requires, in \
`attribution`:

* `CC0`: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), no rights reserved.
* `CC-BY`: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
* `CC-BY-NC`: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), no \
commercial use. Remove these rows with a filter on `license` for commercial use.

Metadata: iNaturalist Open Data. Scores and embeddings: [CC BY 4.0]\
(https://creativecommons.org/licenses/by/4.0/). To have a photo removed, open a discussion \
on this repository.
"""
