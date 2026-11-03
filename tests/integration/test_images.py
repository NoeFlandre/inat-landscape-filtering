"""Image fetching and decoding."""

from inat_landscape.adapters.images import decode, fetch
from tests.conftest import jpeg


def test_fetch_returns_bytes_and_permanent_errors(server):
    _, handler, base = server
    handler.routes["/a.jpg"] = jpeg((255, 0, 0))
    got = {f.photo_id: f for f in fetch({1: f"{base}/a.jpg", 2: f"{base}/missing.jpg"})}
    assert got[1].data == handler.routes["/a.jpg"]
    assert got[1].ok
    assert got[2].data is None
    assert got[2].status == 404
    assert got[2].permanent


def test_fetch_retries_server_errors(server):
    _, handler, base = server
    handler.routes["/b.jpg"] = b"x"
    handler.flaky["/b.jpg"] = 2
    [got] = fetch({7: f"{base}/b.jpg"}, retries=3, backoff=0.0)
    assert got.data == b"x"


def test_fetch_gives_up_after_retries_and_marks_transient(server):
    _, handler, base = server
    handler.routes["/c.jpg"] = b"x"
    handler.flaky["/c.jpg"] = 10
    [got] = fetch({8: f"{base}/c.jpg"}, retries=2, backoff=0.0)
    assert got.status == 503
    assert not got.permanent


def test_fetch_reports_connection_errors_as_transient():
    [got] = fetch({9: "http://127.0.0.1:9/none.jpg"}, retries=1, backoff=0.0)
    assert got.status is None
    assert got.error
    assert not got.permanent


def test_fetch_empty_input():
    assert fetch({}) == []


def test_decode_gives_rgb_image():
    image = decode(jpeg((0, 255, 0)))
    assert image is not None
    assert image.mode == "RGB"
    assert image.size == (24, 18)


def test_decode_returns_none_for_corrupt_bytes():
    assert decode(b"not an image") is None


def test_fetch_in_several_processes_gives_the_same_results_in_order(server):
    _, handler, base = server
    for i in range(10):
        handler.routes[f"/{i}.jpg"] = bytes([i]) * 5
    urls = {i: f"{base}/{i}.jpg" for i in range(10)}
    urls[99] = f"{base}/missing.jpg"
    one = fetch(urls, backoff=0.0)
    many = fetch(urls, backoff=0.0, processes=3)
    assert [(r.photo_id, r.data, r.status) for r in many] == [
        (r.photo_id, r.data, r.status) for r in one
    ]
    assert [r.photo_id for r in many] == list(urls)
