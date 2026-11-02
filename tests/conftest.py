"""A local HTTP server that serves fixture images and error codes."""

import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest
from PIL import Image


def jpeg(color: tuple[int, int, int], size=(24, 18)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


class _Handler(BaseHTTPRequestHandler):
    routes: ClassVar[dict[str, bytes]] = {}
    flaky: ClassVar[dict[str, int]] = {}

    def do_GET(self):
        if self.flaky.get(self.path, 0) > 0:
            self.flaky[self.path] -= 1
            self.send_response(503)
            self.end_headers()
            return
        body = self.routes.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def server():
    _Handler.routes = {}
    _Handler.flaky = {}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, _Handler, f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
