"""Serve an exported bundle on this computer.

This is a plain static file server from the standard library: the search itself runs in
the browser. A server with a live search API is planned on top of the same web app.
"""

from __future__ import annotations

import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class _Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        # A re-export must show up on reload, so nothing is cached.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


def make_server(directory: Path | str, port: int = 8000) -> ThreadingHTTPServer:
    """A server for ``directory`` on localhost; ``port=0`` picks a free port."""
    handler = partial(_Handler, directory=str(directory))
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def serve(directory: Path | str, port: int = 8000, open_browser: bool = False) -> None:
    """Serve ``directory`` until interrupted."""
    with make_server(directory, port) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(f"serving {directory} at {url}  (Ctrl+C to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()
