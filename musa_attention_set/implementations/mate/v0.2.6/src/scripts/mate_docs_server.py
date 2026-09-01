#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
from pathlib import Path
from urllib.parse import urlparse


class DocsHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        path = urlparse(self.path).path
        if path.endswith((".html", ".htm")) or path.endswith("/"):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve MATE docs with no-cache HTML headers"
    )
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--directory", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handler = lambda *handler_args, **handler_kwargs: DocsHandler(  # noqa: E731
        *handler_args,
        directory=str(args.directory),
        **handler_kwargs,
    )
    server = http.server.ThreadingHTTPServer((args.bind, args.port), handler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
