"""Download photos over HTTP with bounded concurrency and retries, and decode them."""

import asyncio
import io
import multiprocessing
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache

import httpx
from PIL import Image, UnidentifiedImageError

_PERMANENT = frozenset({400, 401, 403, 404, 410})
_OK = 200


@dataclass(frozen=True, slots=True)
class Fetched:
    """The result of one download."""

    photo_id: int
    data: bytes | None
    status: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.data is not None

    @property
    def permanent(self) -> bool:
        """Tell if a retry later cannot succeed."""
        return self.status in _PERMANENT


def fetch(
    urls: Mapping[int, str],
    *,
    concurrency: int = 32,
    retries: int = 3,
    backoff: float = 1.0,
    timeout: float = 30.0,
    processes: int = 1,
) -> list[Fetched]:
    """Download each URL. Results come in the order of `urls`.

    One Python process handles about 70 HTTPS downloads per second, whatever `concurrency`
    is (the client work runs in one thread). With `processes` > 1, the URLs are split over a
    pool of processes, each with its own `concurrency`.
    """
    if not urls:
        return []
    if processes <= 1:
        return asyncio.run(_fetch_all(urls, concurrency, retries, backoff, timeout))
    return _fetch_in_processes(urls, processes, (concurrency, retries, backoff, timeout))


def _fetch_in_processes(
    urls: Mapping[int, str], processes: int, options: tuple[int, int, float, float]
) -> list[Fetched]:
    items = list(urls.items())
    args = [(dict(items[k::processes]), *options) for k in range(min(processes, len(items)))]
    found = {r.photo_id: r for part in _pool(processes).map(_fetch_part, args) for r in part}
    return [found[photo_id] for photo_id in urls]


@cache
def _pool(processes: int) -> ProcessPoolExecutor:
    """Return a pool of fetch processes. They are started with `spawn`: the parent can hold
    CUDA, which a fork must not copy."""
    return ProcessPoolExecutor(processes, mp_context=multiprocessing.get_context("spawn"))


def _fetch_part(args: tuple[dict[int, str], int, int, float, float]) -> list[Fetched]:
    urls, concurrency, retries, backoff, timeout = args
    return asyncio.run(_fetch_all(urls, concurrency, retries, backoff, timeout))


async def _fetch_all(
    urls: Mapping[int, str], concurrency: int, retries: int, backoff: float, timeout: float
) -> list[Fetched]:
    limit = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:

        async def one(photo_id: int, url: str) -> Fetched:
            async with limit:
                return await _fetch_one(client, photo_id, url, retries, backoff)

        return await asyncio.gather(*(one(i, u) for i, u in urls.items()))


async def _fetch_one(
    client: httpx.AsyncClient, photo_id: int, url: str, retries: int, backoff: float
) -> Fetched:
    result = Fetched(photo_id, None, error="not tried")
    for attempt in range(retries):
        result = await _try(client, photo_id, url)
        if result.ok or result.permanent:
            return result
        await asyncio.sleep(backoff * 2**attempt)
    return result


async def _try(client: httpx.AsyncClient, photo_id: int, url: str) -> Fetched:
    try:
        response = await client.get(url)
    except httpx.HTTPError as exc:
        return Fetched(photo_id, None, error=f"{type(exc).__name__}: {exc}")
    if response.status_code == _OK:
        return Fetched(photo_id, response.content, status=_OK)
    return Fetched(photo_id, None, status=response.status_code, error=response.reason_phrase)


def decode(data: bytes) -> Image.Image | None:
    """Decode image bytes to RGB, or return None if the bytes are not a valid image."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None
