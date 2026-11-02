"""Command line interface: one command per pipeline stage.

Exit codes: 0 success, 1 runtime error (message on stderr), 2 usage error.
"""

import argparse
import json
import sys
import urllib.error
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import httpx
from huggingface_hub.errors import HfHubHTTPError

from inat_landscape.adapters.hub import REPO_ID, HfHub, Hub, OfflineHub
from inat_landscape.adapters.images import fetch
from inat_landscape.adapters.inat_dump import SNAPSHOT_URL, DumpError, open_source
from inat_landscape.domain.scoring import MODEL_ID
from inat_landscape.domain.stats import cap_report, key_quantiles, summarize
from inat_landscape.domain.thresholds import parse_thresholds
from inat_landscape.pipeline.embed import run_embed
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.ingest import IngestParams, run_index, run_ingest
from inat_landscape.pipeline.ports import Encoder, Fetcher
from inat_landscape.pipeline.publish import PublishSpec, run_publish
from inat_landscape.pipeline.sample import run_sample
from inat_landscape.pipeline.score import SCORE, run_score
from inat_landscape.pipeline.status import run_status

_ERRORS = (StageError, DumpError, HfHubHTTPError, httpx.HTTPError, urllib.error.URLError, OSError)


def _real_encoder(model_id: str, batch_size: int) -> Encoder:
    from inat_landscape.adapters.encoder import SiglipEncoder  # noqa: PLC0415 - torch is slow to import

    return SiglipEncoder(model_id, batch_size)


def _real_hub(repo_id: str, offline: bool) -> Hub:
    return OfflineHub() if offline else HfHub(repo_id)


@dataclass(frozen=True)
class Deps:
    """The outside world, replaceable in tests."""

    opener: Callable[[str], BinaryIO] = open_source
    encoder: Callable[[str, int], Encoder] = _real_encoder
    hub: Callable[[str, bool], Hub] = _real_hub
    fetcher: Fetcher = fetch


def main(argv: Sequence[str] | None = None, deps: Deps | None = None) -> int:
    """Run one command and return its exit code."""
    args = _parser().parse_args(argv)
    try:
        args.handler(args, deps or Deps())
    except _ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _caps(raw: str) -> tuple[int, ...]:
    try:
        return tuple(sorted({int(part) for part in raw.split(",")}))
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad cap list: {raw!r}") from None


def _thresholds(raw: str) -> tuple[float, ...]:
    try:
        return parse_thresholds(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inat-landscape",
        description="Filter iNaturalist photos to keep landscape views. Run the commands in order: "
        "ingest, sample, embed, score, publish.",
    )
    parser.add_argument(
        "--workdir", type=Path, default=Path("work"), help="local work directory (default: work)"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    _add_measure(sub)
    _add_ingest(sub)
    _add_sample(sub)
    _add_embed(sub)
    _add_score(sub)
    _add_publish(sub)
    status = sub.add_parser("status", help="print the row count of each stage output as JSON")
    status.add_argument("--model", default=MODEL_ID, help=f"model id (default: {MODEL_ID})")
    status.set_defaults(handler=_status)
    return parser


def _dump_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default=SNAPSHOT_URL, help="dump URL or local path")
    p.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="share of observations to keep, in (0, 1] (default: 1)",
    )
    p.add_argument("--res", type=int, default=5, help="H3 resolution of the cells (default: 5)")
    p.add_argument(
        "--cap",
        type=int,
        default=1000,
        help="photos kept per cell before embedding (default: 1000)",
    )
    p.add_argument("--force", action="store_true", help="read the dump again even if done")


def _add_ingest(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("ingest", help="stream the metadata dump and keep the capped candidates")
    _dump_options(p)
    p.set_defaults(handler=_ingest)


def _add_measure(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("measure", help="count eligible photos per H3 cell (metadata only)")
    _dump_options(p)
    p.add_argument(
        "--caps",
        type=_caps,
        default=(100, 300, 1000),
        help="caps to report, comma-separated (default: 100,300,1000)",
    )
    p.set_defaults(handler=_measure)


def _add_sample(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sample", help="select the first N candidate photos in sample order")
    p.add_argument("--n", type=int, default=None, help="number of photos (default: all)")
    p.set_defaults(handler=_sample)


def _hub_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo-id", default=REPO_ID, help=f"HF dataset repo (default: {REPO_ID})")
    p.add_argument("--offline", action="store_true", help="do not read or write the HF repo")
    p.add_argument("--model", default=MODEL_ID, help=f"model id (default: {MODEL_ID})")


def _add_embed(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("embed", help="embed sampled photos that have no embedding yet")
    _hub_options(p)
    p.add_argument("--chunk", type=int, default=1000, help="photos per shard (default: 1000)")
    p.add_argument("--batch-size", type=int, default=32, help="model batch size (default: 32)")
    p.set_defaults(handler=_embed)


def _add_score(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("score", help="compute zero-shot scores and print their distribution")
    _hub_options(p)
    p.add_argument(
        "--thresholds",
        type=_thresholds,
        default=(0.0005, 0.001, 0.002, 0.005, 0.01),
        help="thresholds to count in the report, comma-separated",
    )
    p.set_defaults(handler=_score)


def _add_publish(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("publish", help="build the dataset configs and card, and upload them")
    _hub_options(p)
    p.add_argument(
        "--thresholds",
        type=_thresholds,
        required=True,
        help="score thresholds, comma-separated, one config each",
    )
    p.add_argument(
        "--score",
        choices=("score_landscape", "score_contrast"),
        default="score_landscape",
        help="score that defines the configs",
    )
    p.add_argument(
        "--cap-per-cell",
        type=int,
        default=100,
        help="best-scored photos kept per H3 cell in each config (default: 100)",
    )
    p.set_defaults(handler=_publish)


def _params(args: argparse.Namespace) -> IngestParams:
    return IngestParams(source=args.source, fraction=args.fraction, res=args.res, cap=args.cap)


def _measure(args: argparse.Namespace, deps: Deps) -> None:
    index = run_index(args.workdir, _params(args), force=args.force, opener=deps.opener)
    print(cap_report(index.counts["photos"].to_pylist(), args.caps))


def _ingest(args: argparse.Namespace, deps: Deps) -> None:
    table = run_ingest(args.workdir, _params(args), force=args.force, opener=deps.opener)
    print(f"candidates={table.num_rows}")


def _sample(args: argparse.Namespace, _deps: Deps) -> None:
    print(f"sample={run_sample(args.workdir, args.n).num_rows}")


def _embed(args: argparse.Namespace, deps: Deps) -> None:
    report = run_embed(
        args.workdir,
        deps.encoder(args.model, args.batch_size),
        deps.hub(args.repo_id, args.offline),
        fetcher=deps.fetcher,
        chunk=args.chunk,
    )
    print(" ".join(f"{k}={v}" for k, v in vars(report).items()))


def _score(args: argparse.Namespace, deps: Deps) -> None:
    table = run_score(
        args.workdir, deps.encoder(args.model, 32), deps.hub(args.repo_id, args.offline)
    )
    print(summarize(table[SCORE].to_pylist(), args.thresholds))
    quantiles = key_quantiles(table["score_contrast"].to_pylist())
    print("score_contrast: " + " ".join(f"{name}={value:.4f}" for name, value in quantiles))


def _publish(args: argparse.Namespace, deps: Deps) -> None:
    spec = PublishSpec(args.thresholds, score=args.score, cap_per_cell=args.cap_per_cell)
    report = run_publish(
        args.workdir,
        spec,
        deps.hub(args.repo_id, args.offline),
        model_id=args.model,
        fetcher=deps.fetcher,
    )
    print(f"all={report.total} images_failed={report.images_failed}")
    print(" ".join(f"{s.config}={s.rows}" for s in report.subsets))


def _status(args: argparse.Namespace, _deps: Deps) -> None:
    print(json.dumps(run_status(args.workdir, args.model)))


if __name__ == "__main__":
    sys.exit(main())
